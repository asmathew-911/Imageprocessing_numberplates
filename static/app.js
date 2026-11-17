/* ============================================================
   PlateVision — Client-Side Application
   Handles theme switching, file upload, pipeline rendering & lightbox
   ============================================================ */

const $ = id => document.getElementById(id);

let selectedFile  = null;
let requestNumber = 0;
let previewUrl    = null;

// ---- Theme Management --------------------------------------
function initThemes() {
  const saved = localStorage.getItem('platevision-theme') || 'canva-beige';
  setTheme(saved);

  document.querySelectorAll('[data-set-theme]').forEach(btn => {
    btn.addEventListener('click', () => {
      const theme = btn.getAttribute('data-set-theme');
      setTheme(theme);
    });
  });
}

function setTheme(themeName) {
  document.documentElement.setAttribute('data-theme', themeName);
  localStorage.setItem('platevision-theme', themeName);

  document.querySelectorAll('[data-set-theme]').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-set-theme') === themeName);
  });
}

// ---- Status helpers ----------------------------------------
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
  return typeof value === 'string' && value.trim() ? value : fallback;
}

// ---- Pipeline Step Definitions (Matching Uploaded Screenshot)
const PIPELINE_ORDER = [
  { key: 'original',   label: 'Original' },
  { key: 'grayscale',  label: 'Grayscale' },
  { key: 'clahe',      label: 'CLAHE' },
  { key: 'denoised',   label: 'Denoised (Bilateral Filter)' },
  { key: 'edges',      label: 'Edges (Canny)' },
  { key: 'threshold',  label: 'Threshold (Otsu)' },
  { key: 'morphology', label: 'Morphology (Closed)' },
  { key: 'detection',  label: 'Detection Result (Red=A, Green=B)' },
  { key: 'plate_crop', label: 'Plate Crop' },
];

function renderPipeline(data) {
  const steps = data.pipeline_steps || {};
  const grid  = $('pipeline-grid');
  grid.replaceChildren();

  let count = 0;

  // Determine recognized plate text for step 9
  let recognizedText = '';
  if (data.predictions && data.predictions.length > 0) {
    recognizedText = data.predictions[0].text || '';
  }

  PIPELINE_ORDER.forEach(item => {
    const imgSrc = steps[item.key];
    if (!imgSrc) return;
    count++;

    let displayLabel = item.label;
    if (item.key === 'plate_crop') {
      displayLabel = recognizedText ? `Plate Crop - "${recognizedText}"` : 'Plate Crop';
    }

    // Pipeline card container
    const card = document.createElement('div');
    card.className = 'pipeline-card';
    card.setAttribute('tabindex', '0');
    card.setAttribute('role', 'button');
    card.setAttribute('aria-label', `View ${displayLabel}`);

    // Image wrapper
    const imgWrap = document.createElement('div');
    imgWrap.className = 'pipeline-card-img-wrap';

    const img = document.createElement('img');
    img.src = imgSrc;
    img.alt = displayLabel;
    img.loading = 'lazy';
    imgWrap.append(img);

    // Caption banner underneath image
    const banner = document.createElement('div');
    banner.className = 'pipeline-card-label';
    banner.textContent = displayLabel;

    card.append(imgWrap, banner);

    // Lightbox triggers
    card.addEventListener('click', () => openLightbox(imgSrc, displayLabel));
    card.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openLightbox(imgSrc, displayLabel);
      }
    });

    grid.append(card);
  });

  const section = $('pipeline-section');
  if (count > 0) {
    section.hidden = false;
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else {
    section.hidden = true;
  }
}

// ---- Lightbox Functionality --------------------------------
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

$('lightbox-close').addEventListener('click', closeLightbox);
$('lightbox').addEventListener('click', e => {
  if (e.target === $('lightbox')) closeLightbox();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !$('lightbox').hidden) closeLightbox();
});

// ---- Result Rendering --------------------------------------
function addPredictionCard(result, isReference) {
  const container = $('predictions');
  const card = document.createElement('div');
  card.className = 'prediction';

  if (result.crop_image) {
    const cropImg = document.createElement('img');
    cropImg.src = result.crop_image;
    cropImg.alt = 'Plate crop preview';
    card.append(cropImg);
  }

  const body = document.createElement('div');
  body.className = 'prediction-body';

  const textEl = document.createElement('div');
  textEl.className = 'result-text';
  textEl.textContent = safeText(result.text, 'UNREADABLE');
  body.append(textEl);

  if (isReference) {
    const desc = document.createElement('div');
    desc.className = 'result-desc';
    desc.textContent = 'Ground truth annotation from dataset XML';
    body.append(desc);
  } else {
    const desc = document.createElement('div');
    desc.className = 'result-desc';
    const isConfident = result.status === 'read';
    desc.textContent = isConfident
      ? `Detected plate (${Math.round((result.confidence || 0) * 100)}% template match)`
      : 'Uncertain reading — please verify with visual crop';
    body.append(desc);

    // Diagnostics details
    if (Array.isArray(result.character_diagnostics) && result.character_diagnostics.length) {
      const details = document.createElement('details');
      details.className = 'character-details';

      const summary = document.createElement('summary');
      summary.textContent = 'View character template matches';
      details.append(summary);

      result.character_diagnostics.forEach(char => {
        const line = document.createElement('div');
        line.className = 'character-line';
        const alts = (char.alternatives || [])
          .slice(0, 3)
          .map(a => `${a.char} (${Number(a.similarity).toFixed(2)})`)
          .join(' · ');
        line.textContent = `${char.position}. [${char.predicted || '?'}]  ${alts}`;
        details.append(line);
      });
      body.append(details);
    }
  }

  card.append(body);
  container.append(card);
}

function displayResults(data) {
  $('result-placeholder').hidden = true;
  $('result-loaded').hidden = false;
  $('result-image').src = data.annotated_image;

  const predList = $('predictions');
  const sourceEl = $('result-source');
  predList.replaceChildren();
  sourceEl.hidden = true;
  sourceEl.textContent = '';

  if (data.mode === 'reference') {
    const entries = data.reference_entries || [];
    if (!entries.length) {
      predList.textContent = 'No matching reference annotation available.';
      setStatus('NO XML', 'error');
    } else {
      entries.forEach(r => addPredictionCard(r, true));
      setStatus('XML MATCH', 'analyzed');
    }
    sourceEl.textContent = 'Ground-truth dataset reference shown for comparison.';
    sourceEl.hidden = false;
    $('pipeline-section').hidden = true;
    return;
  }

  const predictions = data.predictions || [];
  if (!predictions.length) {
    predList.textContent = 'No candidate registration plate identified in this image.';
    setStatus('NOT FOUND', 'error');
  } else {
    predictions.forEach(p => addPredictionCard(p, false));
    const anyRead = predictions.some(p => p.status === 'read');
    setStatus(anyRead ? 'ANALYZED' : 'UNCERTAIN', anyRead ? 'analyzed' : 'processing');
  }

  // Render the 9 major processing pipeline steps!
  renderPipeline(data);
}

// ---- Analysis API Call -------------------------------------
async function analyzeImage(file) {
  if (!file) return;

  if (!/\.(jpe?g|png)$/i.test(file.name)) {
    feedback('Please upload a valid JPG, JPEG, or PNG image.', true);
    return;
  }
  if (file.size > 24 * 1024 * 1024) {
    feedback('Image file size must be under 24 MB.', true);
    return;
  }

  selectedFile = file;
  const currentReq = ++requestNumber;

  // Local object URL for instant preview
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);

  $('result-image').src = previewUrl;
  $('result-placeholder').hidden = true;
  $('result-loaded').hidden = false;
  $('predictions').textContent = 'Executing OpenCV processing pipeline…';
  $('result-source').hidden = true;
  $('pipeline-section').hidden = true;

  setStatus('PROCESSING', 'processing');
  feedback(`Analyzing ${file.name} through 9 pipeline stages…`);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const mode = $('sample-mode').value;
    const res = await fetch(`/api/upload?mode=${encodeURIComponent(mode)}`, {
      method: 'POST',
      body: formData,
    });
    const result = await res.json().catch(() => ({}));

    if (currentReq !== requestNumber) return;

    if (!res.ok) {
      throw new Error(result.detail || `Server error (${res.status})`);
    }

    displayResults(result);
    feedback(`Analysis complete for ${file.name}`);
  } catch (err) {
    if (currentReq !== requestNumber) return;
    setStatus('FAILED', 'error');
    const msg = err instanceof TypeError
      ? 'Cannot reach local server. Ensure "python app.py" is running.'
      : String(err.message || err);
    feedback(msg, true);
    $('predictions').textContent = `Analysis error: ${msg}`;
  }
}

// ---- Event Listeners ---------------------------------------
const dropzone = $('dropzone');

dropzone.addEventListener('click', () => $('image-upload').click());
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
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  analyzeImage(e.dataTransfer?.files?.[0]);
});

$('image-upload').addEventListener('change', e => {
  const file = e.target.files?.[0];
  e.target.value = '';
  analyzeImage(file);
});

$('sample-mode').addEventListener('change', () => {
  const isRef = $('sample-mode').value === 'reference';
  $('mode-notice').textContent = isRef
    ? 'Displays ground-truth annotations from dataset XML if this image matches a dataset sample.'
    : 'Analyzes pixels directly using OpenCV computer vision algorithms.';
  if (selectedFile) analyzeImage(selectedFile);
});

// Initialize on page load
initThemes();
