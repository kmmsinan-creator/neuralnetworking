/* ===================================================
   GLOBAL CONFIG
=================================================== */

const CSV_URL =
 const CSV_URL =
  "https://raw.githubusercontent.com/kmmsinan-creator/neuralnetworking/main/checkpoint/data/customers_scaled_with_id.csv";

/* ===================================================
   GLOBAL STATE
=================================================== */
let model = null;
let rawData = [];
let featureRows = [];
let customerIds = [];
let probsGlobal = [];
let numFeaturesModel = 30;
let histChart = null;

/* ===================================================
   DOM ELEMENTS
=================================================== */
const logEl = document.getElementById("log");
const loadBtn = document.getElementById("loadBtn");
const fileNameSpan = document.getElementById("fileName");
const predictBtn = document.getElementById("predictBtn");
const modelStatus = document.getElementById("modelStatus");
const resultsTable = document.getElementById("resultsTable");
const resultCard = document.getElementById("resultCard");
const filterPanel = document.getElementById("filterPanel");
const summarySection = document.getElementById("summarySection");
const histSection = document.getElementById("histSection");

const sumTotal = document.getElementById("sumTotal");
const sumChurn = document.getElementById("sumChurn");
const sumRate = document.getElementById("sumRate");

/* ===================================================
   LOG HELPER
=================================================== */
function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

/* ===================================================
   BUILD TFJS MODEL
=================================================== */
function buildModel() {
  const m = tf.sequential();

  m.add(tf.layers.dense({ units: 64, activation: "relu", inputShape: [numFeaturesModel] }));
  m.add(tf.layers.dropout({ rate: 0.3 }));

  m.add(tf.layers.dense({ units: 32, activation: "relu" }));
  m.add(tf.layers.dropout({ rate: 0.2 }));

  m.add(tf.layers.dense({ units: 16, activation: "relu" }));
  m.add(tf.layers.dropout({ rate: 0.1 }));

  m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  return m;
}

/* ===================================================
   LOAD MODEL WEIGHTS
=================================================== */
async function loadModel() {
  try {
    log("Loading neural network weights…");

    model = buildModel();

    const res = await fetch("./model/weights.json");
    const data = await res.json();

    const tensors = data.weights.map((w, i) => tf.tensor(w, data.shapes[i]));
    model.setWeights(tensors);

    modelStatus.textContent = "Model loaded ✔";
    modelStatus.classList.add("status-ready");
    log("✅ Model loaded successfully.");

  } catch (err) {
    log("❌ Error loading model: " + err);
  }
}

/* ===================================================
   LOAD CSV WHEN BUTTON IS CLICKED
=================================================== */
loadBtn.addEventListener("click", () => {
  log("Fetching CSV from GitHub…");

  Papa.parse(CSV_URL, {
    download: true,
    header: true,
    skipEmptyLines: true,

    complete: (results) => {
      rawData = results.data;

      if (rawData.length === 0) {
        log("❌ CSV is empty!");
        return;
      }

      processCSVRows(rawData);

      fileNameSpan.textContent = "Loaded from GitHub ✔";
      predictBtn.disabled = false;

      log(`Loaded ${rawData.length} rows from GitHub.`);
    },

    error: (err) => {
      log("❌ Error loading CSV: " + err);
    }
  });
});

/* ===================================================
   PROCESS CSV ROWS
=================================================== */
function processCSVRows(rows) {
  featureRows = [];
  customerIds = [];

  let keys = Object.keys(rows[0]);
  let featureKeys = keys.filter(k => k !== "CustomerID");

  rows.forEach((row) => {
    const numeric = featureKeys.map((k) => Number(row[k]));
    if (numeric.some(v => isNaN(v))) return;

    featureRows.push(numeric);
    customerIds.push(row["CustomerID"]);
  });

  log(`Valid rows: ${featureRows.length}`);
}

/* ===================================================
   RUN PREDICTION
=================================================== */
predictBtn.addEventListener("click", async () => {
  if (!model || featureRows.length === 0) {
    log("❌ Model or CSV not ready.");
    return;
  }

  log("Running predictions…");

  const input = tf.tensor2d(featureRows);
  const out = model.predict(input);
  const probs = await out.data();

  probsGlobal = Array.from(probs);

  input.dispose();
  out.dispose();

  renderTable(probsGlobal);
  updateSummary(probsGlobal);
  renderHistogram(probsGlobal);

  resultCard.classList.remove("hidden");
  filterPanel.classList.remove("hidden");
  summarySection.classList.remove("hidden");
  histSection.classList.remove("hidden");

  document.getElementById("downloadBtn").disabled = false;

  log("✅ Predictions completed.");
});

/* ===================================================
   TABLE RENDERING
=================================================== */
function renderTable(probs) {
  let html = `
    <thead>
      <tr>
        <th>#</th>
        <th>CustomerID</th>
        <th>Churn Probability</th>
        <th>Class</th>
      </tr>
    </thead>
    <tbody>
  `;

  probs.forEach((p, i) => {
    const probPercent = (p * 100).toFixed(2);
    const label = p >= 0.5 ? "Churned" : "Not Churned";
    const rowClass = p >= 0.5 ? "row-churn" : "row-safe";
    const badgeClass = p >= 0.5 ? "badge churn" : "badge not-churn";

    html += `
      <tr class="${rowClass}">
        <td>${i + 1}</td>
        <td>${customerIds[i]}</td>
        <td>${probPercent}%</td>
        <td><span class="${badgeClass}">${label}</span></td>
      </tr>
    `;
  });

  html += "</tbody>";
  resultsTable.innerHTML = html;
}

/* ===================================================
   SEARCH FUNCTION
=================================================== */
function searchCustomerID() {
  const term = document.getElementById("searchInput").value.toLowerCase();

  const filteredProbs = [];
  const filteredIds = [];

  probsGlobal.forEach((p, i) => {
    const id = String(customerIds[i]).toLowerCase();
    if (id.includes(term)) {
      filteredProbs.push(p);
      filteredIds.push(customerIds[i]);
    }
  });

  const backupIds = [...customerIds];
  const backupProbs = [...probsGlobal];

  customerIds = filteredIds;
  probsGlobal = filteredProbs;

  renderTable(filteredProbs);

  customerIds = backupIds;
  probsGlobal = backupProbs;
}

/* ===================================================
   SORTING
=================================================== */
function sortTable(type) {
  let combined = probsGlobal.map((p, i) => ({ prob: p, id: customerIds[i] }));

  if (type === "prob") combined.sort((a, b) => a.prob - b.prob);
  if (type === "probDesc") combined.sort((a, b) => b.prob - a.prob);

  probsGlobal = combined.map(r => r.prob);
  customerIds = combined.map(r => r.id);

  renderTable(probsGlobal);
}

/* ===================================================
   FILTER
=================================================== */
function filterChurn(type) {
  const filteredProbs = [];
  const filteredIds = [];

  probsGlobal.forEach((p, i) => {
    if (type === "churn" && p < 0.5) return;
    if (type === "not" && p >= 0.5) return;
    filteredProbs.push(p);
    filteredIds.push(customerIds[i]);
  });

  const backupIds = [...customerIds];
  const backupProbs = [...probsGlobal];

  customerIds = filteredIds;
  probsGlobal = filteredProbs;

  renderTable(filteredProbs);

  customerIds = backupIds;
  probsGlobal = backupProbs;
}

/* ===================================================
   SUMMARY CARDS
=================================================== */
function updateSummary(probs) {
  const total = probs.length;
  const churn = probs.filter(p => p >= 0.5).length;

  sumTotal.textContent = total;
  sumChurn.textContent = churn;
  sumRate.textContent = ((churn / total) * 100).toFixed(2) + "%";
}

/* ===================================================
   HISTOGRAM
=================================================== */
function renderHistogram(probs) {
  if (histChart) histChart.destroy();

  const bins = new Array(10).fill(0);
  probs.forEach(p => bins[Math.min(9, Math.floor(p * 10))]++);

  histChart = new Chart(document.getElementById("histChart"), {
    type: "bar",
    data: {
      labels: ["0–10%","10–20%","20–30%","30–40%","40–50%","50–60%","60–70%","70–80%","80–90%","90–100%"],
      datasets: [{
        label: "Customers",
        data: bins,
        backgroundColor: "#4ea1ff"
      }]
    },
    options: { responsive: true }
  });
}

/* ===================================================
   DOWNLOAD CSV
=================================================== */
function downloadCSV() {
  let rows = [["CustomerID", "Probability", "Label"]];

  probsGlobal.forEach((p, i) => {
    rows.push([
      customerIds[i],
      p.toFixed(6),
      p >= 0.5 ? "Churned" : "Not Churned"
    ]);
  });

  const csv = Papa.unparse(rows);
  const blob = new Blob([csv], { type: "text/csv" });

  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "churn_predictions.csv";
  a.click();
}

/* ===================================================
   INITIALIZATION
=================================================== */
document.addEventListener("DOMContentLoaded", () => {
  log("Initializing app…");
  loadModel();       // Load NN model weights
});
