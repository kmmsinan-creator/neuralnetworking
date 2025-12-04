// app.js

let tfModel = null;
let preprocessingConfig = null;
let rawData = null;  // rows from CSV
let header = null;   // columns from CSV

const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const resultsContainer = document.getElementById("resultsContainer");

const singleForm = document.getElementById("singleForm");
const predictSingleBtn = document.getElementById("predictSingleBtn");
const singleResult = document.getElementById("singleResult");

// ===================== 1) LOAD MODEL + CONFIG ON PAGE LOAD =====================
window.addEventListener("DOMContentLoaded", async () => {
  try {
    statusEl.textContent = "Loading TF.js model...";
    tfModel = await tf.loadLayersModel("tfjs_model/model.json");
    statusEl.textContent = "Model loaded. Loading preprocessing_config.json...";

    const resp = await fetch("preprocessing_config.json");
    preprocessingConfig = await resp.json();

    statusEl.textContent = "Model & preprocessing ready. Upload a CSV or use Single Customer section.";

    // Build single-customer dynamic form
    buildSingleCustomerForm(preprocessingConfig);
    predictSingleBtn.disabled = false;
  } catch (err) {
    console.error("Error loading model or config:", err);
    statusEl.textContent = "Error loading model or preprocessing_config.json. Check console.";
  }
});

// ===================== 2) CSV UPLOAD & BATCH PREDICTION =======================

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  statusEl.textContent = "Parsing CSV...";
  rawData = null;
  header = null;
  predictBtn.disabled = true;

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    complete: function (results) {
      rawData = results.data;
      header = results.meta.fields;

      statusEl.textContent = `Loaded ${rawData.length} rows. Click "Run Churn Prediction".`;
      predictBtn.disabled = false;
    },
    error: function (err) {
      console.error("Error parsing CSV:", err);
      statusEl.textContent = "Error parsing CSV. Check console.";
    }
  });
});

predictBtn.addEventListener("click", async () => {
  if (!tfModel || !preprocessingConfig || !rawData) {
    statusEl.textContent = "Model/config/data missing.";
    return;
  }

  statusEl.textContent = "Preparing data and running predictions...";
  predictBtn.disabled = true;

  try {
    const inputTensor = buildInputTensor(rawData, preprocessingConfig);
    const preds = await tfModel.predict(inputTensor).data();
    inputTensor.dispose();

    // Build results with probabilities + labels
    const results = [];
    for (let i = 0; i < rawData.length; i++) {
      const p = preds[i];
      const predictedLabel = p >= 0.5 ? 1 : 0;
      results.push({
        __row: i + 1,
        churn_probability: p,
        churn_pred: predictedLabel,
        ...rawData[i]  // keep original columns if needed
      });
    }

    renderResultsTable(results);
    statusEl.textContent = "Prediction completed.";
  } catch (err) {
    console.error("Error during prediction:", err);
    statusEl.textContent = "Error during prediction. Check console.";
  } finally {
    predictBtn.disabled = false;
  }
});

// Build input tensor using feature_names + means + stds
function buildInputTensor(rows, config) {
  const featureNames = config.feature_names;
  const means = config.means;
  const stds = config.stds;

  const numRows = rows.length;
  const numFeatures = featureNames.length;

  const data = new Float32Array(numRows * numFeatures);

  for (let i = 0; i < numRows; i++) {
    const row = rows[i];

    for (let j = 0; j < numFeatures; j++) {
      const col = featureNames[j];
      let value = parseFloat(row[col]);

      if (Number.isNaN(value)) {
        // missing -> use training mean
        value = means[j];
      }
      const std = stds[j] || 1.0;
      const scaled = (value - means[j]) / std;
      data[i * numFeatures + j] = scaled;
    }
  }

  return tf.tensor2d(data, [numRows, numFeatures]);
}

// Smart-ish display: show row index + ID-like columns + prob + label
function renderResultsTable(results) {
  if (!results.length) {
    resultsContainer.innerHTML = "<p>No data to display.</p>";
    return;
  }

  const allKeys = Object.keys(results[0]);

  // Detect ID-like columns from original CSV
  const idCandidates = allKeys.filter(k => {
    const lower = k.toLowerCase();
    return (
      lower.includes("id") ||
      lower.includes("customer") ||
      lower.includes("account") ||
      lower.includes("user")
    );
  }).filter(k => !["__row", "churn_probability", "churn_pred"].includes(k));

  const colsToShow = ["__row", ...idCandidates, "churn_probability", "churn_pred"];

  let html = "<table><thead><tr>";
  for (const c of colsToShow) {
    if (c === "__row") {
      html += "<th>#</th>";
    } else if (c === "churn_probability") {
      html += "<th>Churn Probability</th>";
    } else if (c === "churn_pred") {
      html += "<th>Prediction</th>";
    } else {
      html += `<th>${c}</th>`;
    }
  }
  html += "</tr></thead><tbody>";

  for (const row of results) {
    html += "<tr>";
    for (const c of colsToShow) {
      if (c === "churn_probability") {
        html += `<td>${row[c].toFixed(4)}</td>`;
      } else if (c === "churn_pred") {
        const isChurn = row[c] === 1 || row[c] === "1";
        const label = isChurn ? "Churned" : "Not Churned";
        const cls = isChurn ? "tag tag-churn" : "tag tag-nochurn";
        html += `<td><span class="${cls}">${label}</span></td>`;
      } else {
        html += `<td>${row[c] !== undefined ? row[c] : ""}</td>`;
      }
    }
    html += "</tr>";
  }

  html += "</tbody></table>";
  resultsContainer.innerHTML = html;
}

// ===================== 3) SINGLE CUSTOMER PREDICTION ==========================

function buildSingleCustomerForm(config) {
  const featureNames = config.feature_names;
  const means = config.means;

  singleForm.innerHTML = "";

  featureNames.forEach((fname, idx) => {
    const fieldDiv = document.createElement("div");
    fieldDiv.className = "form-field";

    const label = document.createElement("label");
    label.textContent = fname;

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = means[idx].toFixed(3); // default to mean

    // Use a safe ID for inputs
    const safeId = "feat_" + idx;
    input.id = safeId;
    input.dataset.featureIndex = idx;

    fieldDiv.appendChild(label);
    fieldDiv.appendChild(input);
    singleForm.appendChild(fieldDiv);
  });
}

predictSingleBtn.addEventListener("click", async () => {
  if (!tfModel || !preprocessingConfig) {
    singleResult.textContent = "Model or preprocessing not loaded yet.";
    return;
  }

  try {
    const featureNames = preprocessingConfig.feature_names;
    const means = preprocessingConfig.means;
    const stds = preprocessingConfig.stds;

    const numFeatures = featureNames.length;
    const data = new Float32Array(numFeatures);

    const inputs = singleForm.querySelectorAll("input[data-feature-index]");

    inputs.forEach((inp) => {
      const idx = parseInt(inp.dataset.featureIndex, 10);
      let value = parseFloat(inp.value);

      if (Number.isNaN(value)) {
        value = means[idx]; // fallback
      }
      const std = stds[idx] || 1.0;
      const scaled = (value - means[idx]) / std;
      data[idx] = scaled;
    });

    const inputTensor = tf.tensor2d(data, [1, numFeatures]);
    const preds = await tfModel.predict(inputTensor).data();
    inputTensor.dispose();

    const prob = preds[0];
    const label = prob >= 0.5 ? "Churned" : "Not Churned";

    singleResult.innerHTML = `
      <p>
        <span class="pill">Churn Probability: ${prob.toFixed(4)}</span>
        <span class="pill">Prediction: ${label}</span>
      </p>
    `;
  } catch (err) {
    console.error("Error during single-customer prediction:", err);
    singleResult.textContent = "Error during prediction. Check console.";
  }
});
