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
from anpr import imread, detect_plates, annotate, recognize_plate
from batch import run_batch, ground_truth
from reference_mode import reference_entries, annotate_xml, run_reference_audit
from dataset_catalog import lookup_image_bytes, lookup_image_file, load_catalog
from prepare_dataset import extract, image_files

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
TEMP = BASE / 'uploads'
for folder in (DATA, RESULTS, TEMP):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='PlateVision: License Plate Recognition')
app.mount('/static', StaticFiles(directory=str(BASE / 'static')), name='static')
_manifest = image_files(DATA)
_catalog = load_catalog()
_lock = threading.Lock()
_state = {'running': False, 'done': 0, 'total': 0, 'current': '', 'summary': None, 'error': None, 'mode': 'classical'}


def _image_data(image, max_width=1150):
    if image is None or image.size == 0:
        return ''
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode() if ok else ''


# Generates base64 visual outputs for each image processing pipeline stage
def _pipeline_steps(image, predictions):
    steps = {}
    h, w = image.shape[:2]

    # Step 1: Original input image
    steps['original'] = _image_data(image)

    # Step 2: Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps['grayscale'] = _image_data(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # Step 3: Local contrast enhancement (CLAHE)
    clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    steps['clahe'] = _image_data(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR))

    # Step 4: Bilateral filter noise reduction
    denoised = cv2.bilateralFilter(clahe_img, 5, 45, 45)
    steps['denoised'] = _image_data(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR))

    # Step 5: Canny edge contour detection
    edges = cv2.Canny(denoised, 80, 185)
    steps['edges'] = _image_data(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    # Step 6: Otsu global threshold binarization
    _, thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    steps['threshold'] = _image_data(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # Step 7: Morphological closing over gradient features
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, rect)
    grad = cv2.Sobel(blackhat, cv2.CV_8U, 1, 0, ksize=3)
    grad = cv2.GaussianBlur(grad, (3, 3), 0)
    _, grad_thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(grad_thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)))
    steps['morphology'] = _image_data(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR))

    # Step 8: Plate localization & bounding box proposals
    det_img = image.copy()
    try:
        from anpr import _candidate_boxes
        candidates = _candidate_boxes(image)
        for _, (cx, cy, cw, ch) in candidates[:4]:
            cv2.rectangle(det_img, (cx, cy), (cx + cw, cy + ch), (40, 40, 220), 2)
    except Exception:
        pass
    for res in predictions:
        bx, by, bw, bh = res.box
        cv2.rectangle(det_img, (bx, by), (bx + bw, by + bh), (46, 204, 113), 3)
    steps['detection'] = _image_data(det_img)

    # Step 9: Plate crop & character extraction
    if predictions and predictions[0].crop is not None and predictions[0].crop.size > 0:
        steps['plate_crop'] = _image_data(predictions[0].crop, 800)
    else:
        steps['plate_crop'] = _image_data(image[h // 3:2 * h // 3, w // 4:3 * w // 4], 800)

    return steps


def _analysis(image, reference_path=None, mode='classical'):
    if image is None:
        raise HTTPException(400, 'Unreadable image. Use JPG, JPEG or PNG.')
    if mode == 'reference':
        if reference_path is None:
            raise HTTPException(400, 'XML reference mode only works on named dataset samples.')
        gt = ground_truth(reference_path) or []
        entries = reference_entries(reference_path, image)
        for e in entries:
            x, y, w, h = e['box']
            e['crop_image'] = _image_data(image[max(0, y):max(0, y + h), max(0, x):max(0, x + w)], 430)
        return {
            'mode': 'reference',
            'predictions': [],
            'reference_entries': entries,
            'annotated_image': _image_data(annotate_xml(image, gt)),
            'reference_for_evaluation_only': None,
            'annotation_notice': 'Retrieved from dataset annotation for comparison.'
        }
    predictions = detect_plates(image, max_plates=3)
    return {
        'mode': 'classical',
        'predictions': [{**res.info(), 'crop_image': _image_data(res.crop, 430),
                        'character_diagnostics': recognize_plate(res.crop)[2].get('characters', [])}
                       for res in predictions],
        'reference_entries': [],
        'annotated_image': _image_data(annotate(image, predictions)),
        'pipeline_steps': _pipeline_steps(image, predictions),
        'reference_for_evaluation_only': None,
        'annotation_notice': 'Automatic computer vision detection complete.'
    }


def _catalog_analysis(image, item):
    if image is None:
        raise HTTPException(400, 'Unreadable JPG/JPEG/PNG file.')
    truth = [(a['text'], tuple(a['box'])) for a in item['annotations']]
    entries = []
    for annotation in item['annotations']:
        x, y, w, h = annotation['box']
        entries.append({
            'text': annotation['text'], 'box': annotation['box'],
            'valid_box': True, 'review_needed': bool(annotation['review_reason']),
            'review_reason': annotation['review_reason'],
            'source': 'Dataset annotation',
            'crop_image': _image_data(image[y:y + h, x:x + w], 430)
        })
    return {
        'mode': 'reference', 'recognition_source': 'DATASET_LOOKUP',
        'predictions': [], 'reference_entries': entries,
        'annotated_image': _image_data(annotate_xml(image, truth)),
        'reference_for_evaluation_only': None,
        'annotation_notice': 'Retrieved matching dataset annotation.'
    }


@app.get('/', response_class=HTMLResponse)
def home():
    return (BASE / 'templates' / 'index.html').read_text(encoding='utf-8')


@app.get('/api/health')
def health():
    return {
        'ok': True,
        'dataset_images': len(_manifest),
        'catalog_images': _catalog.get('metadata', {}).get('total_source_images', 0),
        'engine': 'OpenCV Computer Vision Pipeline'
    }


@app.get('/api/catalog/summary')
def catalog_summary():
    m = _catalog.get('metadata', {})
    return {
        'reference_source': 'Dataset Reference',
        'source_images': m.get('total_source_images', 0),
        'unique_byte_patterns': m.get('unique_sha256', 0),
        'identical_image_pairs': len(m.get('duplicate_identical_files', [])),
        'manual_review_flags': 26
    }


@app.get('/api/catalog/audit')
def catalog_audit():
    path = RESULTS / 'ALL_442_DATASET_CHECK.csv'
    if not path.exists():
        raise HTTPException(404, 'Audit CSV not found.')
    return FileResponse(path, filename='ALL_442_DATASET_CHECK.csv', media_type='text/csv')


@app.get('/api/gallery')
def gallery(offset: int = 0, limit: int = 24):
    offset = max(0, offset)
    limit = max(1, min(48, limit))
    return {
        'total': len(_manifest),
        'images': [{'id': i, 'name': _manifest[i].name} for i in range(offset, min(len(_manifest), offset + limit))]
    }


@app.get('/api/photo/{index}')
def photo(index: int):
    if index < 0 or index >= len(_manifest):
        raise HTTPException(404, 'Image not found')
    return FileResponse(_manifest[index])


@app.post('/api/analyze/{index}')
def analyze(index: int, mode: Literal['classical', 'reference'] = 'classical'):
    if index < 0 or index >= len(_manifest):
        raise HTTPException(404, 'Image not found')
    p = _manifest[index]
    img = imread(p)
    if mode == 'reference':
        item = lookup_image_file(p, _catalog)
        if item is not None:
            return _catalog_analysis(img, item)
        return _analysis(img, p, 'reference')
    return _analysis(img, p, 'classical')


@app.post('/api/upload')
async def upload(file: UploadFile = File(...), mode: Literal['classical', 'reference'] = 'classical'):
    if Path(file.filename or '').suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise HTTPException(415, 'Choose JPG, JPEG, or PNG.')
    contents = await file.read()
    if len(contents) > 24 * 1024 * 1024:
        raise HTTPException(413, 'Image exceeds 24 MB')
    if not contents:
        raise HTTPException(400, 'Empty image file')
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, 'Could not decode the image.')
    if mode == 'reference':
        item = lookup_image_bytes(contents, file.filename or '', _catalog)
        if item is not None:
            return _catalog_analysis(img, item)
        raise HTTPException(404, 'No matching annotation found for this image.')
    return _analysis(img, mode='classical')


@app.post('/api/import-zip')
async def import_zip(file: UploadFile = File(...)):
    global _manifest
    if not (file.filename or '').lower().endswith('.zip'):
        raise HTTPException(415, 'Select a ZIP archive.')
    with _lock:
        if _state['running']:
            raise HTTPException(409, 'Finish current batch before importing.')
    path = TEMP / (uuid.uuid4().hex + '.zip')
    try:
        size = 0
        with path.open('wb') as stream:
            while part := await file.read(2 * 1024 * 1024):
                size += len(part)
                if size > 500 * 1024 * 1024:
                    raise HTTPException(413, 'ZIP exceeds 500 MB')
                stream.write(part)
        info = extract(path, DATA)
        _manifest = image_files(DATA)
        from verify_all_442 import verify_all
        check = verify_all(DATA, RESULTS / 'IMPORTED_DATASET_CHECK.csv')
        return {**info, 'total_available': len(_manifest), 'annotation_check': check}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f'Import failed: {exc}') from exc
    finally:
        path.unlink(missing_ok=True)


def _batch_worker(mode):
    try:
        if mode == 'reference':
            def progress(i, n, name, *c):
                with _lock:
                    _state.update(done=i, total=n, current=name, annotated=c[0], matched=c[1], exact=None)
            results = run_reference_audit(DATA, RESULTS, progress=progress, save_annotated=True)
        else:
            def progress(i, n, name, ann, loc, exact):
                with _lock:
                    _state.update(done=i, total=n, current=name, annotated=ann, matched=loc, exact=exact)
            results = run_batch(DATA, RESULTS, progress=progress, save_annotated=True)
        with _lock:
            _state.update(running=False, summary=results)
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=str(exc))


@app.post('/api/batch/start')
def batch_start(mode: Literal['classical', 'reference'] = 'classical'):
    if not _manifest:
        raise HTTPException(400, 'Import your dataset ZIP first.')
    with _lock:
        if _state['running']:
            raise HTTPException(409, 'A batch is already running.')
        _state.update(running=True, done=0, total=len(_manifest), current='', summary=None,
                      error=None, matched=0, annotated=0, exact=0, mode=mode)
    threading.Thread(target=_batch_worker, args=(mode,), daemon=True).start()
    return {'started': True, 'total': len(_manifest), 'mode': mode}


@app.get('/api/batch/status')
def batch_status():
    with _lock:
        return dict(_state)


@app.get('/api/batch/download')
def batch_download(mode: Literal['classical', 'reference'] = 'classical'):
    file = RESULTS / ('reference_audit.csv' if mode == 'reference' else 'predictions.csv')
    if not file.exists():
        raise HTTPException(404, 'Run that batch mode before downloading.')
    return FileResponse(file, filename=file.name, media_type='text/csv')


if __name__ == '__main__':
    print('\nPlateVision server running: http://127.0.0.1:8000\n')
    uvicorn.run(app, host='127.0.0.1', port=8000)
