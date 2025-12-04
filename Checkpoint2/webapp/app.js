// app.js – model logic, uses UI helpers from ui.js

const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const predictSingleBtn = document.getElementById("predictSingleBtn");

let tfModel = null;
let preprocessingConfig = null;
let rawData = null;

const { 
  setStatus, 
  renderBatchResults, 
  buildSingleCustomerForm, 
  getSingleCustomerValues, 
  showSinglePrediction 
} = window.UI;

// =========== LOAD MODEL & CONFIG ON PAGE LOAD ===========
window.addEventListener("DOMContentLoaded", async () => {
  try {
    setStatus("Loading model...");
    // index.html is in /webapp, model is in /models
    tfModel = await tf.loadLayersModel("../models/tfjs_model/model.json");

    const resp = await fetch("../models/preprocessing_config.json");
    preprocessingConfig = await resp.json();

    setStatus("Model & preprocessing loaded. Ready.");
    buildSingleCustomerForm(
      preprocessingConfig.feature_names,
      preprocessingConfig.means
    );
    predictSingleBtn.disabled = false;
  } catch (err) {
    console.error("Error loading model/config:", err);
    setStatus("Error loading model/config. See console.");
  }
});

// ===================== CSV UPLOAD =======================
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  setStatus("Parsing CSV...");

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      rawData = results.data;
      setStatus(`Loaded ${rawData.length} rows. Click "Run Prediction".`);
      predictBtn.disabled = false;
    },
    error: (err) => {
      console.error("CSV parse error:", err);
      setStatus("Error parsing CSV. See console.");
    }
  });
});

// ===================== BATCH PREDICTION ==================
predictBtn.addEventListener("click", async () => {
  if (!tfModel || !preprocessingConfig || !rawData) {
    setStatus("Model/config/data missing.");
    return;
  }

  setStatus("Running batch prediction...");
  predictBtn.disabled = true;

  try {
    const X = buildInputTensor(rawData, preprocessingConfig);
    const preds = await tfModel.predict(X).data();
    X.dispose();

    const results = rawData.map((row, i) => ({
      ...row,
      probability: preds[i],
      prediction: preds[i] >= 0.5 ? "Churned" : "Not Churned"
    }));

    renderBatchResults(results);
    setStatus("Prediction completed.");
  } catch (err) {
    console.error("Prediction error:", err);
    setStatus("Error during prediction. See console.");
  } finally {
    predictBtn.disabled = false;
  }
});

function buildInputTensor(rows, config) {
  const features = config.feature_names;
  const means = config.means;
  const stds = config.stds;

  const numRows = rows.length;
  const numFeatures = features.length;

  const data = new Float32Array(numRows * numFeatures);

  for (let i = 0; i < numRows; i++) {
    const row = rows[i];
    for (let j = 0; j < numFeatures; j++) {
      const col = features[j];
      let value = parseFloat(row[col]);
      if (Number.isNaN(value)) value = means[j];
      const scaled = (value - means[j]) / (stds[j] || 1.0);
      data[i * numFeatures + j] = scaled;
    }
  }

  return tf.tensor2d(data, [numRows, numFeatures]);
}

// ================== SINGLE CUSTOMER PREDICTION ==========
predictSingleBtn.addEventListener("click", async () => {
  if (!tfModel || !preprocessingConfig) {
    setStatus("Model or preprocessing not ready.");
    return;
  }

  try {
    const features = preprocessingConfig.feature_names;
    const means = preprocessingConfig.means;
    const stds = preprocessingConfig.stds;

    const rawVals = getSingleCustomerValues(features, means);
    const scaled = new Float32Array(features.length);

    for (let i = 0; i < features.length; i++) {
      scaled[i] = (rawVals[i] - means[i]) / (stds[i] || 1.0);
    }

    const X = tf.tensor2d(scaled, [1, features.length]);
    const preds = await tfModel.predict(X).data();
    X.dispose();

    const prob = preds[0];
    showSinglePrediction(prob);
  } catch (err) {
    console.error("Single prediction error:", err);
    setStatus("Error during single-customer prediction.");
  }
});
