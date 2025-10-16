// app.js (UPDATED) — Fixes "Load CSV" button behaviour and makes file-loading robust.
// Replace your existing app.js with this file. It only changes the loading logic.
// Assumes data-loader.js and gru.js remain in the same folder and that index.html
// contains elements with IDs: csv-file, btn-load, btn-train, btn-eval, log, chart.

import DataLoader from './data-loader.js';
import CNNModel from './gru.js';

let loader = null;
let model = null;
let tensors = null;

const fileInput = document.getElementById('csv-file');
const loadBtn = document.getElementById('btn-load');
const trainBtn = document.getElementById('btn-train');
const evalBtn = document.getElementById('btn-eval');
const logDiv = document.getElementById('log');
const chartCanvas = document.getElementById('chart')?.getContext?.('2d');
let chartInstance = null;

function log(msg) {
  const time = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.textContent = `[${time}] ${msg}`;
  logDiv.prepend(line);
}

// Helper: ensure DOM elements exist
if (!fileInput || !loadBtn || !trainBtn || !evalBtn || !logDiv) {
  console.error('Required DOM elements missing. Check index.html IDs: csv-file, btn-load, btn-train, btn-eval, log');
  alert('App initialization error: missing UI elements. See console for details.');
}

// Make buttons disabled until appropriate
trainBtn.disabled = true;
evalBtn.disabled = true;

// Robust file getter (works on click or file input change)
function getSelectedFile() {
  // Prefer file input selection
  if (fileInput && fileInput.files && fileInput.files.length > 0) return fileInput.files[0];
  return null;
}

// Primary load handler
async function handleLoadClick(ev) {
  try {
    loadBtn.disabled = true;
    log('Load CSV: triggered.');
    const file = getSelectedFile();
    if (!file) {
      alert('Please select a CSV file using the file input before clicking Load CSV.');
      log('No file selected.');
      loadBtn.disabled = false;
      return;
    }

    // Basic file type sanity check
    if (!file.name.toLowerCase().endsWith('.csv') && file.type !== 'text/csv') {
      const proceed = confirm('Selected file does not have .csv extension. Proceed anyway?');
      if (!proceed) { loadBtn.disabled = false; return; }
    }

    // Instantiate DataLoader and attempt to load
    loader = new DataLoader({ seqLen: 12, forecastHorizon: 3, testSplitPct: 0.2 });

    // Provide user feedback during potentially long read
    log(`Reading file "${file.name}" (${Math.round(file.size/1024)} KB)...`);
    await loader.loadFile(file); // loader.loadFile uses file.text() internally
    tensors = loader.getTensors();
    log(`CSV loaded. Symbols: ${tensors.symbols.join(', ')}. Samples: train=${tensors.X_train.shape[0]}, test=${tensors.X_test.shape[0]}.`);
    trainBtn.disabled = false;
    evalBtn.disabled = false;
  } catch (err) {
    console.error('Load CSV error:', err);
    alert('Error loading CSV: ' + (err.message || err));
    log('Error loading CSV: ' + (err.message || String(err)));
  } finally {
    loadBtn.disabled = false;
  }
}

// Also handle file input change (auto-enable load button and show file name)
fileInput?.addEventListener('change', (e) => {
  const f = getSelectedFile();
  if (f) {
    log(`File selected: ${f.name} (${Math.round(f.size/1024)} KB)`);
    loadBtn.disabled = false;
  } else {
    log('File input cleared.');
    loadBtn.disabled = false;
  }
});

// Support drag-and-drop onto the file input area (optional)
const fileDropArea = fileInput?.parentElement;
if (fileDropArea) {
  fileDropArea.addEventListener('dragover', (ev) => {
    ev.preventDefault();
    fileDropArea.style.outline = '2px dashed #007bff';
  });
  fileDropArea.addEventListener('dragleave', () => {
    fileDropArea.style.outline = '';
  });
  fileDropArea.addEventListener('drop', (ev) => {
    ev.preventDefault();
    fileDropArea.style.outline = '';
    const dt = ev.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      fileInput.files = dt.files; // populate file input
      const evt = new Event('change');
      fileInput.dispatchEvent(evt);
      log(`File dropped: ${dt.files[0].name}`);
    }
  });
}

// Attach click handler (also support pressing Enter when focused)
loadBtn.addEventListener('click', handleLoadClick);

// Training handler (unchanged logic but safe-guarded)
trainBtn.addEventListener('click', async () => {
  try {
    if (!tensors) { alert('Please load a CSV file first.'); return; }
    trainBtn.disabled = true;
    log('Building model...');
    const seqLen = tensors.seqLen;
    const numFeatures = tensors.X_train.shape[2];
    const nSymbols = tensors.symbols.length;
    model = new CNNModel({
      seqLen,
      numFeatures,
      nSymbols,
      horizon: tensors.forecastHorizon,
      learningRate: 0.001,
    });
    model.build();
    log('Starting training...');
    const epochs = 10;
    const batchSize = 32;
    await model.train(tensors.X_train, tensors.y_train, {
      epochs,
      batchSize,
      onEpoch: (epoch, logs) => {
        log(`Epoch ${epoch + 1}/${epochs}  loss=${(logs.loss||0).toFixed(4)}  val_loss=${(logs.val_loss||0).toFixed(4)}`);
      }
    });
    log('Training finished.');
  } catch (err) {
    console.error('Training error:', err);
    alert('Training error: ' + (err.message || err));
    log('Training error: ' + (err.message || String(err)));
  } finally {
    trainBtn.disabled = false;
  }
});

// Evaluation handler
evalBtn.addEventListener('click', async () => {
  try {
    if (!model || !tensors) { alert('Model or data missing. Train after loading CSV.'); return; }
    evalBtn.disabled = true;
    log('Running evaluation on test set...');
    const results = await model.evaluate(tensors.X_test, tensors.y_test, tensors.symbols, tensors.forecastHorizon);
    log('Evaluation complete. Rendering chart...');
    renderChart(results);
    log('Chart rendered.');
  } catch (err) {
    console.error('Eval error:', err);
    alert('Evaluation error: ' + (err.message || err));
    log('Evaluation error: ' + (err.message || String(err)));
  } finally {
    evalBtn.disabled = false;
  }
});

// Render horizontal accuracy bar chart (sorted best->worst)
function renderChart(results) {
  const sorted = results.slice().sort((a, b) => b.accuracy - a.accuracy);
  const labels = sorted.map(r => r.symbol);
  const data = sorted.map(r => +(r.accuracy * 100).toFixed(2));

  if (!chartCanvas) {
    log('Chart canvas not found; skipping chart rendering.');
    return;
  }

  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  chartInstance = new Chart(chartCanvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Accuracy (%)',
        data,
        backgroundColor: data.map(v => v >= 50 ? 'rgba(76,175,80,0.7)' : 'rgba(244,67,54,0.7)')
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { suggestedMin: 0, suggestedMax: 100 } }
    }
  });
}

// Clean up on unload to avoid TF memory leaks
window.addEventListener('beforeunload', () => {
  try {
    loader?.dispose?.();
    model?.dispose?.();
    if (tensors) {
      ['X_train','y_train','X_test','y_test'].forEach(k => {
        if (tensors[k] && tensors[k].dispose) tensors[k].dispose();
      });
    }
    chartInstance?.destroy();
  } catch (e) { /* ignore */ }
});

// Ensure UI initially reflects that user must select a file
loadBtn.disabled = false;
trainBtn.disabled = true;
evalBtn.disabled = true;
log('Ready — select a CSV and click Load CSV.');
