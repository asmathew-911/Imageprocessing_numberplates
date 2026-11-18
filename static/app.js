
/* ============================================================
   PLATEVISION - CLIENT-SIDE APPLICATION
   Image Upload, Recognition Results, 9 Processing Stages
   Works with Automatic OpenCV and XML Reference modes
   ============================================================ */

const $ = id => document.getElementById(id);

let selectedFile = null;
let requestNumber = 0;
let previewUrl = null;


// ============================================================
// 1. THEME MANAGEMENT
// ============================================================

function initThemes() {
    const saved =
        localStorage.getItem('platevision-theme') ||
        'canva-beige';

    setTheme(saved);

    document.querySelectorAll('[data-set-theme]')
        .forEach(btn => {
            btn.addEventListener('click', () => {
                setTheme(btn.getAttribute('data-set-theme'));
            });
        });
}

function setTheme(themeName) {
    document.documentElement.setAttribute(
        'data-theme',
        themeName
    );

    localStorage.setItem(
        'platevision-theme',
        themeName
    );

    document.querySelectorAll('[data-set-theme]')
        .forEach(btn => {
            btn.classList.toggle(
                'active',
                btn.getAttribute('data-set-theme') === themeName
            );
        });
}


// ============================================================
// 2. STATUS AND FEEDBACK
// ============================================================

function setStatus(text, statusClass = 'ready') {
    const badge = $('analysis-status');

    badge.textContent = text;
    badge.className = `status-badge ${statusClass}`;
}

function feedback(message, isError = false) {
    const el = $('upload-feedback');

    el.textContent = message;
    el.classList.toggle('error', isError);
}

function safeText(value, fallback = '—') {
    return typeof value === 'string' && value.trim()
        ? value
        : fallback;
}


// ============================================================
// 3. NINE IMAGE-PROCESSING STAGES
// ============================================================

const PIPELINE_ORDER = [
    {
        key: 'original',
        label: 'Original'
    },
    {
        key: 'grayscale',
        label: 'Grayscale'
    },
    {
        key: 'clahe',
        label: 'CLAHE'
    },
    {
        key: 'denoised',
        label: 'Denoised (Bilateral Filter)'
    },
    {
        key: 'edges',
        label: 'Edges (Canny)'
    },
    {
        key: 'threshold',
        label: 'Threshold (Otsu)'
    },
    {
        key: 'morphology',
        label: 'Morphology (Closed)'
    },
    {
        key: 'detection',
        label: 'Detection Result'
    },
    {
        key: 'plate_crop',
        label: 'Plate Crop'
    }
];


// ============================================================
// 4. RENDER PROCESSING PIPELINE
//    Works in BOTH recognition modes
// ============================================================

function renderPipeline(data) {
    const steps = data.pipeline_steps || {};
    const grid = $('pipeline-grid');
    const section = $('pipeline-section');

    grid.replaceChildren();

    const isReference = data.mode === 'reference';

    // Get the displayed plate number from the selected mode.
    const recognizedText = isReference
        ? ((data.reference_entries || [])[0]?.text || '')
        : ((data.predictions || [])[0]?.text || '');

    let count = 0;

    PIPELINE_ORDER.forEach(item => {
        const imgSrc = steps[item.key];

        if (!imgSrc) return;

        count++;

        let displayLabel = item.label;

        // In XML mode the bounding box comes from annotations.
        if (item.key === 'detection') {
            displayLabel = isReference
                ? 'Reference Plate Detection (XML)'
                : 'Automatic Plate Detection';
        }

        // Set the plate crop label.
        if (item.key === 'plate_crop') {
            if (isReference) {
                displayLabel = recognizedText
                    ? `Reference Plate Crop (XML) - "${recognizedText}"`
                    : 'Reference Plate Crop (XML)';
            } else {
                displayLabel = recognizedText
                    ? `Plate Crop - "${recognizedText}"`
                    : 'Plate Crop';
            }
        }

        // Create processing-stage card.
        const card = document.createElement('div');

        card.className = 'pipeline-card';
        card.setAttribute('tabindex', '0');
        card.setAttribute('role', 'button');
        card.setAttribute('aria-label', `View ${displayLabel}`);

        // Image container.
        const imgWrap = document.createElement('div');

        imgWrap.className = 'pipeline-card-img-wrap';

        const img = document.createElement('img');

        img.src = imgSrc;
        img.alt = displayLabel;
        img.loading = 'lazy';

        imgWrap.append(img);

        // Caption.
        const banner = document.createElement('div');

        banner.className = 'pipeline-card-label';
        banner.textContent = displayLabel;

        card.append(imgWrap, banner);

        // Click to enlarge.
        card.addEventListener('click', () => {
            openLightbox(imgSrc, displayLabel);
        });

        card.addEventListener('keydown', e => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openLightbox(imgSrc, displayLabel);
            }
        });

        grid.append(card);
    });

    // Show the stages in both modes.
    if (count > 0) {
        section.hidden = false;
        section.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest'
        });
    } else {
        section.hidden = true;
    }
}


// ============================================================
// 5. LIGHTBOX - ENLARGE PROCESSING IMAGES
// ============================================================

function openLightbox(src, title) {
    $('lightbox-img').src = src;
    $('lightbox-label').textContent = title;

    $('lightbox').hidden = false;
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    $('lightbox').hidden = true;
    document.body.style.overflow = '';
}

$('lightbox-close').addEventListener(
    'click',
    closeLightbox
);

$('lightbox').addEventListener('click', e => {
    if (e.target === $('lightbox')) {
        closeLightbox();
    }
});

document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('lightbox').hidden) {
        closeLightbox();
    }
});


// ============================================================
// 6. DISPLAY RECOGNIZED PLATE CARD
// ============================================================

function addPredictionCard(result, isReference) {
    const container = $('predictions');

    const card = document.createElement('div');
    card.className = 'prediction';

    // Cropped plate image.
    if (result.crop_image) {
        const cropImg = document.createElement('img');

        cropImg.src = result.crop_image;
        cropImg.alt = 'Number plate crop';

        card.append(cropImg);
    }

    const body = document.createElement('div');
    body.className = 'prediction-body';

    // Display plate number.
    const textEl = document.createElement('div');

    textEl.className = 'result-text';
    textEl.textContent = safeText(
        result.text,
        'UNREADABLE'
    );

    body.append(textEl);

    const desc = document.createElement('div');
    desc.className = 'result-desc';

    if (isReference) {
        desc.textContent =
            'Reference plate number from dataset annotation';

        body.append(desc);
    } else {
        const isConfident = result.status === 'read';

        desc.textContent = isConfident
            ? `Detected plate (${Math.round(
                (result.confidence || 0) * 100
              )}% template match)`
            : 'Uncertain reading - verify with plate crop';

        body.append(desc);

        // Character-level recognition diagnostics.
        if (
            Array.isArray(result.character_diagnostics) &&
            result.character_diagnostics.length
        ) {
            const details = document.createElement('details');
            details.className = 'character-details';

            const summary = document.createElement('summary');
            summary.textContent = 'View character template matches';

            details.append(summary);

            result.character_diagnostics.forEach(char => {
                const line = document.createElement('div');
                line.className = 'character-line';

                const alternatives = (char.alternatives || [])
                    .slice(0, 3)
                    .map(a =>
                        `${a.char} (${Number(a.similarity).toFixed(2)})`
                    )
                    .join(' · ');

                line.textContent =
                    `${char.position}. [${char.predicted || '?'}] ` +
                    alternatives;

                details.append(line);
            });

            body.append(details);
        }
    }

    card.append(body);
    container.append(card);
}


// ============================================================
// 7. DISPLAY ANALYSIS RESULTS
//    Important: Both modes show processing stages
// ============================================================

function displayResults(data) {
    $('result-placeholder').hidden = true;
    $('result-loaded').hidden = false;

    $('result-image').src = data.annotated_image;

    const predList = $('predictions');
    const sourceEl = $('result-source');

    predList.replaceChildren();

    sourceEl.hidden = true;
    sourceEl.textContent = '';

    // --------------------------------------------------------
    // MODE A: DATASET ANNOTATION COMPARISON
    // --------------------------------------------------------

    if (data.mode === 'reference') {
        const entries = data.reference_entries || [];

        if (!entries.length) {
            predList.textContent =
                'No matching reference annotation available.';

            setStatus('NOT FOUND', 'error');
        } else {
            entries.forEach(result => {
                addPredictionCard(result, true);
            });

            setStatus('ANALYZED', 'analyzed');
        }

        sourceEl.textContent =
            'Plate location and text supplied by dataset XML annotations.';

        sourceEl.hidden = false;

        // SHOW ALL PROCESSING STAGES IN XML MODE.
        renderPipeline(data);

        return;
    }

    // --------------------------------------------------------
    // MODE B: AUTOMATIC OPENCV RECOGNITION
    // --------------------------------------------------------

    const predictions = data.predictions || [];

    if (!predictions.length) {
        predList.textContent =
            'No registration plate detected in this image.';

        setStatus('NOT FOUND', 'error');
    } else {
        predictions.forEach(result => {
            addPredictionCard(result, false);
        });

        const anyRead = predictions.some(
            p => p.status === 'read'
        );

        setStatus(
            anyRead ? 'ANALYZED' : 'UNCERTAIN',
            anyRead ? 'analyzed' : 'processing'
        );
    }

    // SHOW ALL PROCESSING STAGES IN AUTOMATIC MODE.
    renderPipeline(data);
}


// ============================================================
// 8. UPLOAD AND ANALYZE CAR IMAGE
// ============================================================

async function analyzeImage(file) {
    if (!file) return;

    // Only allow JPG, JPEG and PNG.
    if (!/\.(jpe?g|png)$/i.test(file.name)) {
        feedback(
            'Please upload a JPG, JPEG, or PNG image.',
            true
        );
        return;
    }

    // Maximum file size: 24 MB.
    if (file.size > 24 * 1024 * 1024) {
        feedback(
            'Image file size must be under 24 MB.',
            true
        );
        return;
    }

    selectedFile = file;

    const currentReq = ++requestNumber;

    // Display the selected image immediately.
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);

    $('result-image').src = previewUrl;
    $('result-placeholder').hidden = true;
    $('result-loaded').hidden = false;

    $('predictions').textContent =
        'Executing OpenCV processing pipeline...';

    $('result-source').hidden = true;
    $('pipeline-section').hidden = true;

    setStatus('PROCESSING', 'processing');

    feedback(
        `Analyzing ${file.name} through the processing stages...`
    );

    // Prepare multipart image upload.
    const formData = new FormData();

    formData.append('file', file);

    try {
        const mode = $('sample-mode').value;

        const response = await fetch(
            `/api/upload?mode=${encodeURIComponent(mode)}`,
            {
                method: 'POST',
                body: formData
            }
        );

        const result = await response.json().catch(
            () => ({})
        );

        // Ignore results from earlier uploads.
        if (currentReq !== requestNumber) {
            return;
        }

        if (!response.ok) {
            throw new Error(
                result.detail ||
                `Server error (${response.status})`
            );
        }

        // Display detection and all processing stages.
        displayResults(result);

        feedback(
            `Analysis complete for ${file.name}`
        );

    } catch (error) {
        if (currentReq !== requestNumber) {
            return;
        }

        setStatus('FAILED', 'error');

        const message = error instanceof TypeError
            ? 'Cannot reach server. Make sure python app.py is running.'
            : String(error.message || error);

        feedback(message, true);

        $('predictions').textContent =
            `Analysis error: ${message}`;
    }
}


// ============================================================
// 9. DRAG AND DROP IMAGE UPLOAD
// ============================================================

const dropzone = $('dropzone');

dropzone.addEventListener('click', () => {
    $('image-upload').click();
});

dropzone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        $('image-upload').click();
    }
});

dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', e => {
    e.preventDefault();

    dropzone.classList.remove('dragover');

    analyzeImage(e.dataTransfer?.files?.[0]);
});


// ============================================================
// 10. FILE SELECTION BUTTON
// ============================================================

$('image-upload').addEventListener('change', e => {
    const file = e.target.files?.[0];

    e.target.value = '';

    analyzeImage(file);
});


// ============================================================
// 11. SWITCH BETWEEN AUTOMATIC AND XML MODES
// ============================================================

$('sample-mode').addEventListener('change', () => {
    const isReference =
        $('sample-mode').value === 'reference';

    $('mode-notice').textContent = isReference
        ? 'Displays the dataset reference annotation and its processing stages.'
        : 'Analyzes image pixels using classical OpenCV algorithms.';

    // Reprocess the selected image in the new mode.
    if (selectedFile) {
        analyzeImage(selectedFile);
    }
});


// ============================================================
// 12. INITIALIZE WEBSITE
// ============================================================

initThemes();
