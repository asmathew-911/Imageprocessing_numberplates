const $ = id => document.getElementById(id);
let selectedFile = null;
let requestNumber = 0;
let previewUrl = null;

function setStatus(value) { $('analysis-status').textContent = value; }
function feedback(value, isError=false) {
  const el = $('upload-feedback');
  el.textContent = value;
  el.classList.toggle('error', isError);
}
function safeText(value, fallback='—') { return typeof value === 'string' && value.trim() ? value : fallback; }

function addResult(result, isReference) {
  const list = $('predictions');
  const card = document.createElement('div');
  card.className = 'prediction';
  if (result.crop_image) {
    const crop = document.createElement('img');
    crop.src = result.crop_image;
    crop.alt = 'Number plate crop';
    card.append(crop);
  }
  const body = document.createElement('div');
  body.className = 'prediction-body';
  const text = document.createElement('div');
  text.className = 'result-text';
  text.textContent = safeText(result.text, 'NOT READABLE');
  body.append(text);
  if (isReference) {
    const source = document.createElement('div');
    source.className = 'result-desc';
    source.textContent = 'Reference annotation · not automatically read from pixels';
    body.append(source);
    if (result.review_needed) {
      const warning = document.createElement('div');
      warning.className = 'review-warning';
      warning.textContent = 'This reference label may need manual review.';
      body.append(warning);
    }
  } else {
    const status = document.createElement('div');
    status.className = 'result-desc';
    status.textContent = result.status === 'read' ? 'Automatic image-processing result · please verify' : 'Uncertain reading · please verify';
    body.append(status);
    if (Array.isArray(result.character_diagnostics) && result.character_diagnostics.length) {
      const more = document.createElement('details');
      more.className = 'character-details';
      const summary = document.createElement('summary');
      summary.textContent = 'Character details';
      more.append(summary);
      for (const c of result.character_diagnostics) {
        const line = document.createElement('div');
        line.className = 'character-line';
        const alternatives = (c.alternatives || []).slice(0, 3).map(a => `${a.char} (${Number(a.similarity).toFixed(2)})`).join(' · ');
        line.textContent = `${c.position}. ${c.predicted || '?'}  —  ${alternatives}`;
        more.append(line);
      }
      body.append(more);
    }
  }
  card.append(body);
  list.append(card);
}

function showResult(data) {
  $('result-placeholder').hidden = true;
  $('result-loaded').hidden = false;
  $('result-image').src = data.annotated_image;
  const list = $('predictions');
  list.replaceChildren();
  const source = $('result-source');
  source.hidden = true;
  source.textContent = '';
  if (data.mode === 'reference') {
    const entries = data.reference_entries || [];
    if (!entries.length) {
      list.textContent = 'No matching reference annotation available.';
      setStatus('NO REFERENCE');
    } else {
      entries.forEach(r => addResult(r, true));
      setStatus('REFERENCE');
    }
    source.textContent = 'This comparison uses an existing annotation, not automatic character recognition.';
    source.hidden = false;
    return;
  }
  const predictions = data.predictions || [];
  if (!predictions.length) {
    list.textContent = 'No reliable number plate found in this image.';
    setStatus('NOT FOUND');
  } else {
    predictions.forEach(r => addResult(r, false));
    setStatus(predictions.some(r => r.status === 'read') ? 'ANALYZED' : 'UNCERTAIN');
  }
}

async function analyze(file) {
  if (!file) return;
  if (!/\.(jpe?g|png)$/i.test(file.name)) { feedback('Please choose a JPG, JPEG, or PNG file.',true); return; }
  if (file.size > 24 * 1024 * 1024) { feedback('Please select an image under 24 MB.',true); return; }
  selectedFile = file;
  const seq = ++requestNumber;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('result-image').src = previewUrl;
  $('result-placeholder').hidden = true;
  $('result-loaded').hidden = false;
  $('predictions').textContent = 'Analyzing the image…';
  $('result-source').hidden = true;
  setStatus('PROCESSING');
  feedback(`Processing ${file.name}…`);
  const form = new FormData();
  form.append('file', file);
  try {
    const mode = $('sample-mode').value;
    const response = await fetch('/api/upload?mode=' + encodeURIComponent(mode), {method:'POST', body:form});
    const data = await response.json().catch(() => ({}));
    if (seq !== requestNumber) return;
    if (!response.ok) throw new Error(data.detail || `Upload error (${response.status})`);
    showResult(data);
    feedback(`Completed: ${file.name}`);
  } catch (err) {
    if (seq !== requestNumber) return;
    setStatus('ERROR');
    const error = err instanceof TypeError ? 'Cannot contact the local Python server. Check that python app.py is running.' : String(err.message || err);
    feedback(error, true);
    $('predictions').textContent = 'Unable to analyze this image. ' + error;
  }
}

const dropzone = $('dropzone');
dropzone.addEventListener('click', () => $('image-upload').click());
dropzone.addEventListener('keydown', event => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('image-upload').click(); }
});
dropzone.addEventListener('dragover', event => { event.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', event => {
  event.preventDefault(); dropzone.classList.remove('dragover');
  analyze(event.dataTransfer?.files?.[0]);
});
$('image-upload').addEventListener('change', event => {
  const file = event.target.files?.[0];
  event.target.value = '';
  analyze(file);
});
$('sample-mode').addEventListener('change', () => {
  const isReference = $('sample-mode').value === 'reference';
  $('mode-notice').textContent = isReference
    ? 'Shows an existing annotation for an exactly matching reference image, not a plate number read from pixels.'
    : 'Analyzes image pixels directly using OpenCV. Accuracy varies with image quality.';
  if (selectedFile) analyze(selectedFile);
});
