# PlateVision v5 — clean image-processing website

A dark, clean website for number plate detection and character reading using Python, FastAPI, OpenCV, and deterministic multi-font template matching. No ML models are used in `anpr.py`.

## Run on Windows

Extract this ZIP so `app.py` and `requirements.txt` are directly in your project folder. In VS Code terminal, run:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8000 and press Ctrl+Shift+R if an earlier local version is cached.

## Website changes

- Minimal upload and result layout; no sample gallery or dataset/audit marketing blocks.
- Removed the embedded top banner and the `XML:` text overlay from reference images. The image now only highlights the plate rectangle.
- Number appears under **Plate reading**.
- Analysis mode is at the very bottom in an expandable section.
- Default is **Automatic image processing**, which reads pixels without XML or hash lookup.
- Optional **Annotation comparison** works only for exact original reference images. Its result remains truthfully marked as an annotation rather than OCR.
- Supports only JPG, JPEG and PNG in the image uploader.

## Accuracy / integrity

Automatic image processing is not 100% accurate. The separate reference comparison can show an existing supplied text label, but this is not independent recognition. Do not represent that mode's accuracy as OCR performance. The project includes batch benchmarking scripts for objective evaluation.

## Project source

- `app.py`: HTTP endpoints and page server
- `anpr.py`: automatic plate detection and template-based recognition
- `reference_mode.py`, `dataset_catalog.py`: optional supplied-annotation comparison
- `templates/index.html`, `static/app.js`, `static/style.css`: streamlined frontend
- `batch.py`: batch independent evaluation
- `verify_all_442.py`: separate annotation coverage audit

### Batch testing (independent OpenCV mode)

```powershell
python batch.py --dataset data/google_images --output results --no-annotated-images
```

The bundled `data/google_images` folder contains only a few example images. To import your original ZIP for batch evaluation, run `python prepare_dataset.py --zip "C:\path\to\archive (1).zip" --output data`. This extracts only still images and paired annotations; files in `video_images` are excluded.
