let tfModel = null;
let preprocessingConfig = null;
let rawData = null;

const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const resultsContainer = document.getElementById("resultsContainer");
const fileInput = document.getElementById("fileInput");

const singleForm = document.getElementById("singleForm");
const predictSingleBtn = document.getElementById("predictSingleBtn");

// ================= LOAD MODEL + PREPROCESSING CONFIG =================
window.addEventListener("DOMContentLoaded", async () => {
  try {
    statusEl.textContent = "Loading model...";
    tfModel = await tf.loadLayersModel("models/tfjs_model/model.json");

    const resp = await fetch("models/preprocessing_config.json");
    preprocessingConfig = await resp.json();

    statusEl.textContent = "Model loaded. Ready.";
    buildSingleForm(preprocessingConfig);
    predictSingleBtn.disabled = false;

  } catch (err) {
    console.error("Error loading model:", err);
    statusEl.textContent = "Error loading model. Check console.";
  }
});

// ================= CSV UPLOAD =================
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  statusEl.textContent = "Parsing CSV...";

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      rawData = results.data;
      statusEl.textContent = `CSV loaded: ${rawData.length} rows`;
      predictBtn.disabled = false;
    }
  });
});

// ================= BATCH PREDICTION =================
predictBtn.addEventListener("click", async () => {
  if (!rawData) return;

  statusEl.textContent = "Predicting...";

  const X = buildInputTensor(rawData, preprocessingConfig);
  const preds = await tfModel.predict(X).data();
  X.dispose();

  const output = rawData.map((r, i) => ({
    ...r,
    probability: preds[i],
    prediction: preds[i] >= 0.5 ? "Churned" : "Not Churned"
  }));

  renderResults(output);
  statusEl.textContent = "Done.";
});

// Build input tensor using standardized values
function buildInputTensor(rows, config) {
  const features = config.feature_names;
  const means = config.means;
  const stds = config.stds;

  const numRows = rows.length;
  const numFeatures = features.length;

  const arr = new Float32Array(numRows * numFeatures);

  rows.forEach((row, i) => {
    features.forEach((f, j) => {
      let val = parseFloat(row[f]);
      if (isNaN(val)) val = means[j];
      arr[i * numFeatures + j] = (val - means[j]) / (stds[j] || 1);
    });
  });

  return tf.tensor2d(arr, [numRows, numFeatures]);
}

// Render HTML table for results
function renderResults(rows) {
  let html = "<table><tr><th>Probability</th><th>Prediction</th></tr>";

  rows.forEach(r => {
    html += `
      <tr>
        <td>${r.probability.toFixed(4)}</td>
        <td>${r.prediction}</td>
      </tr>`;
  });

  html += "</table>";
  resultsContainer.innerHTML = html;
}

// =============== SINGLE CUSTOMER FORM =================
function buildSingleForm(config) {
  const features = config.feature_names;
  const means = config.means;

  let html = "";
  features.forEach((f, i) => {
    html += `
      <div>
        <label>${f}</label>
        <input type="number" id="feat_${i}" value="${means[i].toFixed(3)}" step="any" />
      </div>
    `;
  });

  singleForm.innerHTML = html;
}

predictSingleBtn.addEventListener("click", async () => {
  const features = preprocessingConfig.feature_names;
  const means = preprocessingConfig.means;
  const stds = preprocessingConfig.stds;

  const arr = new Float32Array(features.length);

  features.forEach((f, i) => {
    let val = parseFloat(document.getElementById(`feat_${i}`).value);
    if (isNaN(val)) val = means[i];
    arr[i] = (val - means[i]) / (stds[i] || 1);
  });

  const X = tf.tensor2d(arr, [1, features.length]);
  const pred = await tfModel.predict(X).data();
  X.dispose();

  const p = pred[0];
  document.getElementById("singleResult").innerHTML =
    `<p><b>Probability:</b> ${p.toFixed(4)} <br>
     <b>Prediction:</b> ${p >= 0.5 ? "Churned" : "Not Churned"}</p>`;
});
