# PlateVision — Number Plate Recognition System

PlateVision is a computer vision web application for vehicle number plate localization and character recognition built using Python, OpenCV, and FastAPI. It features an interactive UI styled after the Canva Blue & Beige aesthetic, complete with a 9-stage visual image processing pipeline.

---

## Quick Start & Localhost Link

1. **Install dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

2. **Start the application:**
   ```powershell
   python app.py
   ```

3. **Generate benchmark & noise robustness plots:**
   ```powershell
   python generate_benchmark_plots.py
   ```

4. **Open in your browser:**
   ```
   http://127.0.0.1:8000
   ```
   *(Link: [http://127.0.0.1:8000](http://127.0.0.1:8000))*

---

## Project Directory Structure

```
Imageprocessing_numberplates/
├── anpr.py                         # 🧠 Core computer vision & character OCR algorithms
├── app.py                          # 🌐 FastAPI web server & 9-stage visual pipeline generator
├── batch.py                        # 📊 Batch evaluation runner across dataset directories
├── generate_benchmark_plots.py     # 📈 Benchmark & noise robustness plot generator (300 DPI)
├── dataset_catalog.py              # 🔍 SHA256 image lookup helper for ground-truth references
├── dataset_reference_catalog.json  # 📑 Pre-compiled reference annotations catalog
├── prepare_dataset.py              # 📦 Dataset ZIP extractor and image file scanner
├── reference_mode.py               # 📋 Ground-truth XML parser for dataset evaluation
├── verify_all_442.py               # ✅ Dataset annotation integrity verification script
├── requirements.txt                # 📦 Python package dependencies (OpenCV, FastAPI, Uvicorn)
├── README.md                       # 📖 Documentation and usage guide
│
├── templates/
│   └── index.html                  # 🖥️ Webpage structure, dropzone, and 3x3 pipeline grid
│
├── static/
│   ├── style.css                   # 🎨 Canva Blue & Beige color palette and layout styling
│   └── app.js                      # ⚡ Client-side upload handler, pipeline renderer, & lightbox
│
└── data/
    └── google_images/              # 🚗 Sample vehicle images (.png/.jpg) and annotations (.xml)
```

---

## Important Files & Their Role in Image Processing

### 1. `anpr.py` *(Core Computer Vision Engine)*
* **Preprocessing & Edge Filtering:** Applies Grayscale conversion, Contrast Limited Adaptive Histogram Equalization (`cv2.createCLAHE`) to boost local contrast, bilateral edge-preserving smoothing (`cv2.bilateralFilter`), and multi-directional Canny edge detection (`cv2.Canny`).
* **Candidate Region Localization:** Performs multi-scale morphological blackhat/tophat filtering with rectangular structuring elements (`cv2.morphologyEx`) and Sobel vertical gradients, then scores bounding boxes by aspect ratio, contour density, and gradient texture.
* **Character Segmentation & Recognition:** Segments individual character glyphs via Otsu threshold binarization and connected components analysis (`cv2.connectedComponentsWithStats`), then decodes characters using normalized cross-correlation against multi-font synthetic templates with topological hole-count verification.

### 2. `app.py` *(Web Server & Visual Pipeline Generator)*
* **API Endpoints:** Hosts the FastAPI web application, serving the web UI at `/` and accepting vehicle image uploads at `/api/upload`.
* **Visual Pipeline Generator (`_pipeline_steps`):** Automatically captures snapshots of all 9 intermediate image processing transformations, converts them to base64 images, and delivers them to the frontend for real-time visualization.
* **Dual Execution Modes:** Supports both automatic computer vision inference on uploaded images and ground-truth XML annotation comparison on known dataset samples.

### 3. `templates/index.html` *(Frontend UI Structure)*
* **Interactive Studio Layout:** Provides a vehicle photograph upload dropzone with drag-and-drop capability and a detection result panel displaying the recognized license plate.
* **9-Stage Pipeline Showcase:** Features a 3×3 visual grid displaying each step of the OpenCV pipeline with click-to-enlarge lightbox modals.
* **Theme Switching:** Includes an integrated theme selector for toggling between Canva Beige & Navy, Canva Deep Navy, and Dark Cyber aesthetics.

### 4. `static/style.css` & `static/app.js` *(Design & Client-Side Logic)*
* **Canva Aesthetic (`style.css`):** Implements the exact color scheme from Canva template `EAGumw_v_CU` using `#F4EEE2` (Warm Beige) and `#2F386F` (Deep Indigo Navy) with responsive card grids.
* **Interactive Behavior (`app.js`):** Asynchronously dispatches image files to the backend via `fetch()`, dynamically renders the 9 pipeline stages with labels, and manages full-resolution lightbox viewing.

### 5. `batch.py` *(Batch Evaluation Runner)*
* **Dataset Benchmarking:** Scans entire folders of vehicle photos, executing the `anpr.py` pipeline across each file sequentially.
* **Evaluation Metrics:** Calculates Intersection-over-Union (IoU) overlap against ground-truth bounding boxes and verifies character prediction accuracy, outputting results to CSV.

---

## The 9-Stage Visual Processing Pipeline

When any vehicle image is uploaded, the following 9 intermediate image processing stages are computed and displayed:

| Step | Stage Name | Computer Vision Technique Applied |
| :---: | :--- | :--- |
| **01** | **Original** | Raw input vehicle photograph. |
| **02** | **Grayscale** | Eliminates chrominance to focus purely on luminance intensity (`cv2.cvtColor`). |
| **03** | **CLAHE** | Contrast-Limited Adaptive Histogram Equalization enhances faint plate edges. |
| **04** | **Denoised (Bilateral Filter)** | Smooths background textures while keeping glyph edges sharp (`cv2.bilateralFilter`). |
| **05** | **Edges (Canny)** | Multi-scale gradient analysis isolating plate boundaries (`cv2.Canny`). |
| **06** | **Threshold (Otsu)** | Calculates the optimal inter-class variance binarization threshold. |
| **07** | **Morphology (Closed)** | Sobel vertical gradient + rectangular morphological closing to consolidate characters. |
| **08** | **Detection Result** | Visualizes candidate proposals in Red and the best-scoring plate in Green. |
| **09** | **Plate Crop** | High-resolution isolated plate region paired with the recognized registration text. |
