"""PlateVision backend: classical OpenCV ANPR + separately labeled XML comparison.
Run from your project folder: python app.py
"""
from __future__ import annotations

import base64
import threading
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from anpr import annotate, detect_plates, imread, recognize_plate
from batch import ground_truth, run_batch
from dataset_catalog import load_catalog, lookup_image_bytes, lookup_image_file
from prepare_dataset import extract, image_files
from reference_mode import annotate_xml, reference_entries, run_reference_audit

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
UPLOADS = BASE / 'uploads'
for folder in (DATA, RESULTS, UPLOADS):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='PlateVision - Classical Image Processing ANPR')
app.mount('/static', StaticFiles(directory=str(BASE / 'static')), name='static')
_manifest = image_files(DATA)
_catalog = load_catalog()
_lock = threading.Lock()
_state = dict(running=False, done=0, total=0, current='', summary=None,
              error=None, mode='classical', matched=0, annotated=0, exact=0)


def _image_data(image, max_width=1100):
    """Encode a diagnostic OpenCV image for display in the browser."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return ''
    height, width = image.shape[:2]
    if width > max_width:
        new_height = max(1, round(height * max_width / width))
        image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 83])
    return ('data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii')) if ok else ''


def _pipeline_steps(image, predictions=None, reference_boxes=None):
    """Return all nine visual stages for either analysis mode.

    Classical mode: selected boxes and crop are produced by the image-only
    detector. Reference mode: selected boxes and crop come from supplied XML
    annotations. Preprocessing previews always come from image pixels.
    The previews are illustrative and do not change the OCR algorithm.
    """
    predictions = predictions or []
    reference_mode = reference_boxes is not None
    reference_boxes = reference_boxes or []

    height, width = image.shape[:2]
    ratio = min(1.0, 1050.0 / max(width, 1))
    preview = (cv2.resize(image, (max(1, round(width * ratio)),
                   max(1, round(height * ratio))), interpolation=cv2.INTER_AREA)
               if ratio < 1 else image.copy())

    # The first seven stages are generated from the uploaded image in BOTH modes.
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.bilateralFilter(clahe, 5, 45, 45)
    edges = cv2.Canny(denoised, 80, 185)
    _, threshold = cv2.threshold(
        clahe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    blackhat = cv2.morphologyEx(
        denoised, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    gradient = cv2.Sobel(blackhat, cv2.CV_8U, 1, 0, ksize=3)
    gradient = cv2.GaussianBlur(gradient, (3, 3), 0)
    _, mask = cv2.threshold(
        gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    morphology = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)))

    # Stage 8: Detection overlay; mode-specific to preserve provenance.
    detection = preview.copy()
    crop = None
    if reference_mode:
        # Cyan rectangles are provided by dataset XML, NOT detected by OCR.
        for box in reference_boxes:
            x, y, bw, bh = (int(v) for v in box)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + bw), min(height, y + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            sx0, sy0, sx1, sy1 = (round(v * ratio) for v in (x0, y0, x1, y1))
            cv2.rectangle(detection, (sx0, sy0), (sx1, sy1), (38, 190, 240), 3)
            if crop is None:
                crop = image[y0:y1, x0:x1]
    else:
        # Red boxes: image-only proposals. Green boxes: selected image-only boxes.
        try:
            from anpr import _candidate_boxes
            for _, (x, y, bw, bh) in _candidate_boxes(image)[:4]:
                x, y, bw, bh = [round(v * ratio) for v in (x, y, bw, bh)]
                cv2.rectangle(detection, (x, y), (x+bw, y+bh), (40, 40, 220), 2)
        except Exception as exc:
            print('Diagnostic candidate boxes unavailable:', exc, flush=True)
        for plate in predictions:
            x, y, bw, bh = [round(v * ratio) for v in plate.box]
            cv2.rectangle(detection, (x, y), (x+bw, y+bh), (46, 204, 113), 3)
        crop = predictions[0].crop if predictions else None

    # Stage 9: never substitute a random centre-of-car crop.
    if crop is None or crop.size == 0:
        crop = np.full((110, 370, 3), (18, 24, 37), dtype=np.uint8)
        label = 'NO REFERENCE CROP' if reference_mode else 'NO PLATE DETECTED'
        cv2.putText(crop, label, (18, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (245, 245, 245), 2, cv2.LINE_AA)

    return {
        'original': _image_data(preview),
        'grayscale': _image_data(gray),
        'clahe': _image_data(clahe),
        'denoised': _image_data(denoised),
        'edges': _image_data(edges),
        'threshold': _image_data(threshold),
        'morphology': _image_data(morphology),
        'detection': _image_data(detection),
        'plate_crop': _image_data(crop, 800),
    }


def _analysis(image, reference_path=None, mode='classical'):
    if image is None or image.size == 0:
        raise HTTPException(400, 'Unreadable image. Use JPG, JPEG or PNG.')

    if mode == 'reference':
        if reference_path is None:
            raise HTTPException(400, 'This reference mode requires a named dataset image.')
        annotations = ground_truth(reference_path) or []
        entries = reference_entries(reference_path, image)
        for entry in entries:
            x, y, w, h = entry['box']
            entry['crop_image'] = _image_data(image[max(y, 0):max(y+h, 0),
                                                     max(x, 0):max(x+w, 0)], 430)
        return {
            'mode': 'reference', 'predictions': [], 'reference_entries': entries,
            'annotated_image': _image_data(annotate_xml(image, annotations)),
            'pipeline_steps': _pipeline_steps(image, reference_boxes=[e['box'] for e in entries]),
            'reference_for_evaluation_only': None,
            'annotation_notice': 'XML reference data, not automatic detection or OCR.'
        }

    # True image-only detection and text recognition; XML is NOT used here.
    predictions = detect_plates(image, max_plates=3)
    response_predictions = []
    for n, plate in enumerate(predictions, start=1):
        print(f'Candidate {n}: {plate.box} -> {plate.text!r}, '
              f'confidence={plate.confidence:.3f}, status={plate.status}', flush=True)
        info = plate.info()
        info['crop_image'] = _image_data(plate.crop, 430)
        info['character_diagnostics'] = []
        if plate.crop is not None and plate.crop.size:
            try:
                _, _, details = recognize_plate(plate.crop)
                info['character_diagnostics'] = details.get('characters', [])
            except Exception as exc:
                print('Character diagnostics unavailable:', exc, flush=True)
        response_predictions.append(info)

    if not predictions:
        print('No plate candidate detected.', flush=True)

    return {
        'mode': 'classical', 'predictions': response_predictions,
        'reference_entries': [],
        'annotated_image': _image_data(annotate(image, predictions)),
        'pipeline_steps': _pipeline_steps(image, predictions),
        'reference_for_evaluation_only': None,
        'annotation_notice': 'Image-only classical OpenCV recognition (not XML lookup).'
    }


def _catalog_analysis(image, item):
    """Look up the dataset's labels, explicitly NOT an OCR result."""
    if image is None:
        raise HTTPException(400, 'Unreadable image.')
    truth = [(a['text'], tuple(a['box'])) for a in item['annotations']]
    entries = []
    for annotation in item['annotations']:
        x, y, w, h = annotation['box']
        entries.append({
            'text': annotation['text'], 'box': annotation['box'],
            'valid_box': True,
            'review_needed': bool(annotation.get('review_reason')),
            'review_reason': annotation.get('review_reason', ''),
            'source': 'Dataset XML annotation (not OCR)',
            'crop_image': _image_data(image[y:y+h, x:x+w], 430)
        })
    return {
        'mode': 'reference',
        'recognition_source': 'DATASET_XML_LOOKUP_NOT_OCR',
        'predictions': [], 'reference_entries': entries,
        'annotated_image': _image_data(annotate_xml(image, truth)),
        'pipeline_steps': _pipeline_steps(image, reference_boxes=[a['box'] for a in item['annotations']]),
        'reference_for_evaluation_only': None,
        'annotation_notice': 'Dataset XML label shown for reference, not inferred from pixels.'
    }


@app.get('/', response_class=HTMLResponse)
def home():
    return (BASE / 'templates' / 'index.html').read_text(encoding='utf-8')


@app.get('/api/health')
def health():
    return {'ok': True, 'dataset_images': len(_manifest),
            'catalog_images': _catalog.get('metadata', {}).get('total_source_images', 0),
            'engine': 'Classical OpenCV; optional separate XML reference'}


@app.get('/api/catalog/summary')
def catalog_summary():
    meta = _catalog.get('metadata', {})
    return {'reference_source': 'Dataset XML, not OCR',
            'source_images': meta.get('total_source_images', 0),
            'unique_byte_patterns': meta.get('unique_sha256', 0),
            'identical_image_pairs': len(meta.get('duplicate_identical_files', [])),
            'manual_review_flags': 26, 'label_match_is_not_ocr': True}


@app.get('/api/catalog/audit')
def catalog_audit():
    file = RESULTS / 'ALL_442_DATASET_CHECK.csv'
    if not file.exists():
        raise HTTPException(404, 'Dataset audit CSV not found.')
    return FileResponse(file, filename=file.name, media_type='text/csv')


@app.get('/api/gallery')
def gallery(offset: int = 0, limit: int = 24):
    offset = max(0, offset)
    limit = max(1, min(48, limit))
    return {'total': len(_manifest), 'images': [
        {'id': i, 'name': _manifest[i].name}
        for i in range(offset, min(len(_manifest), offset + limit))]}


@app.get('/api/photo/{index}')
def photo(index: int):
    if index < 0 or index >= len(_manifest):
        raise HTTPException(404, 'Image not found.')
    return FileResponse(_manifest[index])


@app.post('/api/analyze/{index}')
def analyze(index: int, mode: Literal['classical', 'reference'] = 'classical'):
    if index < 0 or index >= len(_manifest):
        raise HTTPException(404, 'Image not found.')
    path = _manifest[index]
    image = imread(path)
    if mode == 'reference':
        item = lookup_image_file(path, _catalog)
        return _catalog_analysis(image, item) if item is not None else _analysis(image, path, 'reference')
    return _analysis(image, path, 'classical')


@app.post('/api/upload')
async def upload(file: UploadFile = File(...),
                 mode: Literal['classical', 'reference'] = 'classical'):
    if Path(file.filename or '').suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise HTTPException(415, 'Choose a JPG, JPEG or PNG image.')
    raw = await file.read(24 * 1024 * 1024 + 1)
    if len(raw) > 24 * 1024 * 1024:
        raise HTTPException(413, 'Image must be 24 MB or smaller.')
    if not raw:
        raise HTTPException(400, 'Empty image file.')
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, 'Could not decode this image.')
    if mode == 'reference':
        item = lookup_image_bytes(raw, file.filename or '', _catalog)
        if item is None:
            raise HTTPException(404, 'No matching XML annotation. Select Automatic Image Processing.')
        return _catalog_analysis(image, item)
    return _analysis(image, mode='classical')


@app.post('/api/import-zip')
async def import_zip(file: UploadFile = File(...)):
    global _manifest
    if not (file.filename or '').lower().endswith('.zip'):
        raise HTTPException(415, 'Choose a ZIP file.')
    with _lock:
        if _state['running']:
            raise HTTPException(409, 'A batch is currently running.')
    archive = UPLOADS / f'{uuid.uuid4().hex}.zip'
    try:
        size = 0
        with archive.open('wb') as output:
            while part := await file.read(2 * 1024 * 1024):
                size += len(part)
                if size > 500 * 1024 * 1024:
                    raise HTTPException(413, 'ZIP must be 500 MB or smaller.')
                output.write(part)
        info = extract(archive, DATA)
        _manifest = image_files(DATA)
        # The annotation audit is a separate evaluation operation.
        from verify_all_442 import verify_all
        audit = verify_all(DATA, RESULTS / 'IMPORTED_DATASET_CHECK.csv')
        return {**info, 'total_available': len(_manifest), 'annotation_check': audit}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f'Import failed: {exc}') from exc
    finally:
        archive.unlink(missing_ok=True)


def _batch_worker(mode):
    try:
        if mode == 'reference':
            def progress(i, n, name, *counts):
                with _lock:
                    _state.update(done=i, total=n, current=name,
                                  annotated=counts[0], matched=counts[1], exact=None)
            result = run_reference_audit(DATA, RESULTS, progress=progress, save_annotated=True)
        else:
            def progress(i, n, name, annotated, matched, exact):
                with _lock:
                    _state.update(done=i, total=n, current=name,
                                  annotated=annotated, matched=matched, exact=exact)
            result = run_batch(DATA, RESULTS, progress=progress, save_annotated=True)
        with _lock:
            _state.update(running=False, summary=result)
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=str(exc))


@app.post('/api/batch/start')
def batch_start(mode: Literal['classical', 'reference'] = 'classical'):
    if not _manifest:
        raise HTTPException(400, 'Import a dataset ZIP first.')
    with _lock:
        if _state['running']:
            raise HTTPException(409, 'A batch is already running.')
        _state.update(running=True, done=0, total=len(_manifest), current='',
                      summary=None, error=None, mode=mode,
                      matched=0, annotated=0, exact=0)
    threading.Thread(target=_batch_worker, args=(mode,), daemon=True).start()
    return {'started': True, 'total': len(_manifest), 'mode': mode}


@app.get('/api/batch/status')
def batch_status():
    with _lock:
        return dict(_state)


@app.get('/api/batch/download')
def batch_download(mode: Literal['classical', 'reference'] = 'classical'):
    filename = 'reference_audit.csv' if mode == 'reference' else 'predictions.csv'
    file = RESULTS / filename
    if not file.exists():
        raise HTTPException(404, 'Run batch analysis before downloading results.')
    return FileResponse(file, filename=filename, media_type='text/csv')


if __name__ == '__main__':
    print('\nPlateVision running at http://127.0.0.1:8000\n', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=8000)
