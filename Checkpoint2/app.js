// app.js – complete, self-contained logic for Churn TF.js app

// ---------- DOM elements ----------
const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const predictSingleBtn = document.getElementById("predictSingleBtn");
// This button is optional – only exists if you added it in index.html
const useSampleBtn = document.getElementById("useSampleBtn");

const statusEl = document.getElementById("status");
const resultsContainer = document.getElementById("resultsContainer");
const singleForm = document.getElementById("singleForm");
const singleResult = document.getElementById("singleResult");

// ---------- Global state ----------
let tfModel = null;
let preprocessingConfig = null;
let rawData = null;

// ---------- Small UI helpers ----------
function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg;
}

function renderBatchResults(rows) {
  if (!rows || !rows.length) {
    resultsContainer.innerHTML = "<p>No results to show.</p>";
    return;
  }

  let html = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Churn Probability</th>
          <th>Prediction</th>
        </tr>
      </thead>
      <tbody>
  `;

  rows.forEach((r, i) => {
    const prob = r.probability || 0;
    const label = r.prediction || "Not Churned";
    const isChurn = label === "Churned";
    const cls = isChurn ? "tag tag-churn" : "tag tag-nochurn";

    html += `
      <tr>
        <td>${i + 1}</td>
        <td>${prob.toFixed(4)}</td>
        <td><span class="${cls}">${label}</span></td>
      </tr>
    `;
  });

  html += "</tbody></table>";
  resultsContainer.innerHTML = html;
}

function buildSingleCustomerForm(featureNames, means) {
  singleForm.innerHTML = "";
  featureNames.forEach((fname, idx) => {
    const row = document.createElement("div");
    row.className = "form-row";

    const label = document.createElement("label");
    label.textContent = fname;

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = Number(means[idx]).toFixed(3);
    input.id = `feat_${idx}`;

    row.appendChild(label);
    row.appendChild(input);
    singleForm.appendChild(row);
  });
}

function getSingleCustomerValues(featureNames, means) {
  const vals = new Float32Array(featureNames.length);
  featureNames.forEach((_, idx) => {
    const inp = document.getElementById(`feat_${idx}`);
    let v = parseFloat(inp?.value);
    if (Number.isNaN(v)) v = means[idx];
    vals[idx] = v;
  });
  return vals;
}

function showSinglePrediction(prob) {
  const label = prob >= 0.5 ? "Churned" : "Not Churned";
  singleResult.innerHTML = `
    <p>
      <span class="pill">Probability: ${prob.toFixed(4)}</span>
      <span class="pill">Prediction: ${label}</span>
    </p>
  `;
}

// ---------- Load model + config on page load ----------
window.addEventListener("load", async () => {
  try {
    setStatus("Loading model...");

    // index.html is in /Checkpoint2, model is in /Checkpoint2/models/...
    tfModel = await tf.loadLayersModel("models/tfjs_model/model.json");

    const resp = await fetch("models/preprocessing_config.json");
    preprocessingConfig = await resp.json();

    setStatus("Model & preprocessing loaded. Ready.");

    // Build single-customer form using feature names + means
    buildSingleCustomerForm(
      preprocessingConfig.feature_names,
      preprocessingConfig.means
    );

    predictSingleBtn.disabled = false;
  } catch (err) {
    console.error("Error loading model/config:", err);
    setStatus("Error loading model/config. Open browser console (F12 → Console) for details.");
  }
});

// ---------- CSV upload from user ----------
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  setStatus("Parsing uploaded CSV...");

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      rawData = results.data;
      setStatus(`Loaded ${rawData.length} rows from uploaded file. Click "Run Prediction".`);
      predictBtn.disabled = false;
    },
    error: (err) => {
      console.error("CSV parse error:", err);
      setStatus("Error parsing uploaded CSV. See console.");
    }
  });
});

// ---------- Optional: use sample CSV from /data/ ----------
if (useSampleBtn) {
  useSampleBtn.addEventListener("click", async () => {
    try {
      setStatus("Loading sample dataset from /data/E_Commerce_Dataset.csv ...");

      const response = await fetch("data/E_Commerce_Dataset.csv");
      const csvText = await response.text();

      Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          rawData = results.data;
          setStatus(`Loaded ${rawData.length} rows from sample dataset. Click "Run Prediction".`);
          predictBtn.disabled = false;
        },
        error: (err) => {
          console.error("Sample CSV parse error:", err);
          setStatus("Error parsing sample CSV. See console.");
        }
      });
    } catch (err) {
      console.error("Error fetching sample dataset:", err);
      setStatus("Error loading sample dataset. See console.");
    }
  });
}

// ---------- Batch prediction ----------
predictBtn.addEventListener("click", async () => {
  // Detailed checks so you see *what* is missing
  if (!tfModel) {
    setStatus("ERROR: Model not loaded (tfModel is null). Check models/tfjs_model/model.json path.");
    return;
  }
  if (!preprocessingConfig) {
    setStatus("ERROR: preprocessing_config.json not loaded.");
    return;
  }
  if (!rawData || !rawData.length) {
    setStatus("ERROR: No data loaded. Did the CSV parse correctly?");
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

// ---------- Single-customer prediction ----------
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
    setStatus("Error during single-customer prediction. See console.");
  }
});
