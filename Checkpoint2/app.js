// app.js – TF.js churn prediction frontend

let tfModel = null;
let preprocessConfig = null;
let rawCsvData = null;

// Utility: small helper to create status tags
function statusTag(text, type = "ok") {
  const cls = type === "ok" ? "tag-ok" : "tag-warn";
  return `<span class="tag ${cls}">${text}</span>`;
}

// 1) Load TF.js model and preprocessing config on page load
async function loadResources() {
  const modelStatusEl = document.getElementById("model-status");
  const preprocessStatusEl = document.getElementById("preprocess-status");

  try {
    modelStatusEl.innerHTML = "Loading model from <code>model/model.json</code>...";
    tfModel = await tf.loadLayersModel("model/model.json");
    modelStatusEl.innerHTML = `${statusTag("MODEL LOADED")} Model ready.`;
  } catch (err) {
    console.error("Error loading model:", err);
    modelStatusEl.innerHTML = `${statusTag("MODEL ERROR", "warn")} Could not load model/model.json`;
  }

  try {
    preprocessStatusEl.innerHTML = "Loading preprocessing_config.json...";
    const resp = await fetch("preprocessing_config.json");
    preprocessConfig = await resp.json();

    preprocessStatusEl.innerHTML =
      `${statusTag("PREPROCESS OK")} ` +
      `Loaded <code>preprocessing_config.json</code> with ` +
      `${preprocessConfig.feature_names.length} features and threshold ${preprocessConfig.threshold}.`;
  } catch (err) {
    console.error("Error loading preprocessing config:", err);
    preprocessStatusEl.innerHTML =
      `${statusTag("PREPROCESS ERROR", "warn")} Could not load preprocessing_config.json`;
  }

  // Enable predict button only if everything is loaded
  const predictBtn = document.getElementById("predict-btn");
  if (tfModel && preprocessConfig) {
    predictBtn.disabled = false;
  }
}

// 2) Handle file upload (CSV)
function setupFileInput() {
  const fileInput = document.getElementById("file-input");
  const uploadStatus = document.getElementById("upload-status");

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) {
      uploadStatus.innerHTML = `${statusTag("NO FILE", "warn")} Please select a CSV file.`;
      return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
      uploadStatus.innerHTML = `${statusTag("INVALID", "warn")} Only .csv files are supported.`;
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      rawCsvData = parseCsv(text); // store parsed data
      if (rawCsvData.rows.length === 0) {
        uploadStatus.innerHTML = `${statusTag("EMPTY", "warn")} CSV has no data rows.`;
      } else {
        uploadStatus.innerHTML =
          `${statusTag("CSV READY")} Loaded ${rawCsvData.rows.length} rows with ` +
          `${rawCsvData.headers.length} columns.`;
      }
    };
    reader.readAsText(file);
    uploadStatus.innerHTML = "Reading CSV...";
  });
}

// 3) Simple CSV parser (no external libraries)
// Returns { headers: [...], rows: [ {col: val, ...}, ... ] }
function parseCsv(csvText) {
  const lines = csvText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length === 0) {
    return { headers: [], rows: [] };
  }

  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length === 0) continue;
    const rowObj = {};
    headers.forEach((h, idx) => {
      rowObj[h] = parts[idx] !== undefined ? parts[idx].trim() : "";
    });
    rows.push(rowObj);
  }

  return { headers, rows };
}

// 4) Preprocess a single row using medians + scaler params from config
function preprocessRow(rowObj) {
  const { feature_names, medians, scaler_mean, scaler_scale } = preprocessConfig;
  const featureVector = [];

  for (const feat of feature_names) {
    let rawVal = rowObj[feat];

    // If missing, use median
    if (rawVal === undefined || rawVal === null || rawVal === "") {
      rawVal = medians[feat];
    }

    let x = parseFloat(rawVal);
    if (Number.isNaN(x)) {
      // again fall back to median if parse failed
      x = parseFloat(medians[feat]) || 0.0;
    }

    // Scale: (x - mean) / scale
    const mean = scaler_mean[feat];
    const scale = scaler_scale[feat] || 1.0;
    const xScaled = (x - mean) / scale;

    featureVector.push(xScaled);
  }

  return featureVector;
}

// 5) Run prediction on all uploaded rows
async function runPrediction() {
  const resultsDiv = document.getElementById("results");
  const uploadStatus = document.getElementById("upload-status");

  if (!tfModel || !preprocessConfig) {
    alert("Model or preprocessing config not loaded yet.");
    return;
  }
  if (!rawCsvData || rawCsvData.rows.length === 0) {
    alert("Please upload a CSV file first.");
    return;
  }

  const { rows } = rawCsvData;
  const threshold = preprocessConfig.threshold;

  // Build feature matrix
  const featureMatrix = rows.map((row) => preprocessRow(row));
  const tensor = tf.tensor2d(featureMatrix);

  // Predict probabilities
  const probsTensor = tfModel.predict(tensor);
  const probs = await probsTensor.data();

  tensor.dispose();
  probsTensor.dispose();

  // Build result table
  let html = "";
  html += `<p>Predictions for ${rows.length} customers (threshold = ${threshold}):</p>`;
  html += `<table><thead><tr>
      <th>#</th>
      <th>Churn Probability</th>
      <th>Prediction</th>
  </tr></thead><tbody>`;

  for (let i = 0; i < rows.length; i++) {
    const p = probs[i];
    const label = p >= threshold ? 1 : 0;
    const probPercent = (p * 100).toFixed(2) + "%";
    const cls = label === 1 ? "pred-high" : "pred-low";
    const labelText = label === 1 ? "Churn" : "Not Churn";

    html += `<tr>
        <td>${i + 1}</td>
        <td>${probPercent}</td>
        <td class="${cls}">${labelText}</td>
    </tr>`;
  }

  html += `</tbody></table>`;

  resultsDiv.innerHTML = html;
  uploadStatus.innerHTML += "<br/>Prediction complete.";
}

// Init
window.addEventListener("DOMContentLoaded", () => {
  loadResources();
  setupFileInput();

  const predictBtn = document.getElementById("predict-btn");
  predictBtn.addEventListener("click", () => {
    runPrediction().catch((err) => {
      console.error("Error during prediction:", err);
      alert("Prediction failed. Check console for details.");
    });
  });
});
