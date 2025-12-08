/* ===================================================
   CONFIG
=================================================== */

// YOUR REAL CSV PATH IN GITHUB
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
let histChart = null;
let numFeaturesModel = 30;

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
   HELPERS
=================================================== */

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

/* ===================================================
   BUILD MODEL (MUST MATCH TRAINED NN)
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

      processCSV(rawData);

      fileNameSpan.textContent = "Loaded from GitHub ✔";
      predictBtn.disabled = false;

      log(`Loaded ${rawData.length} rows from GitHub.`);
    },

    error: (err, file, inputElem, reason) => {
      console.error("PapaParse error:", err, reason);
      let msg = err?.message || reason || "Unknown error";
      log("❌ Error loading CSV: " + msg);
    }
  });
});

/* ===================================================
   PROCESS CSV ROWS
=================================================== */

function processCSV(rows) {
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

  renderTable();
  updateSummary();
  renderHistogram();

  resultCard.classList.remove("hidden");
  filterPanel.classList.remove("hidden");
  summarySection.classList.remove("hidden");
  histSection.classList.remove("hidden");

  document.getElementById("downloadBtn").disabled = false;

  log("✅ Predictions completed.");
});

/* ===================================================
   RENDER TABLE WITH ROW ANIMATIONS
=================================================== */

function renderTable() {
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

  probsGlobal.forEach((p, i) => {
    const prob = (p * 100).toFixed(2);
    const label = p >= 0.5 ? "Churned" : "Not Churned";
    const rowClass = p >= 0.5 ? "row-churn" : "row-safe";
    const badgeClass = p >= 0.5 ? "badge churn" : "badge not-churn";

    html += `
      <tr class="${rowClass}">
        <td>${i + 1}</td>
        <td>${customerIds[i]}</td>
        <td>${prob}%</td>
        <td><span class="${badgeClass}">${label}</span></td>
      </tr>
    `;
  });

  html += "</tbody>";
  resultsTable.innerHTML = html;
}

/* ===================================================
   SEARCH BY CUSTOMERID
=================================================== */

function searchCustomerID() {
  const term = document.getElementById("searchInput").value.toLowerCase();

  const filteredIndices = customerIds
    .map((id, i) => (String(id).toLowerCase().includes(term) ? i : -1))
    .filter(i => i !== -1);

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

  filteredIndices.forEach((i) => {
    const p = probsGlobal[i];
    const prob = (p * 100).toFixed(2);
    const label = p >= 0.5 ? "Churned" : "Not Churned";
    const rowClass = p >= 0.5 ? "row-churn" : "row-safe";
    const badgeClass = p >= 0.5 ? "badge churn" : "badge not-churn";

    html += `
      <tr class="${rowClass}">
        <td>${i + 1}</td>
        <td>${customerIds[i]}</td>
        <td>${prob}%</td>
        <td><span class="${badgeClass}">${label}</span></td>
      </tr>
    `;
  });

  html += "</tbody>";
  resultsTable.innerHTML = html;
}

/* ===================================================
   SORTING
=================================================== */

function sortTable(type) {
  let arr = probsGlobal.map((p, i) => ({
    prob: p,
    id: customerIds[i]
  }));

  if (type === "prob") arr.sort((a, b) => a.prob - b.prob);
  if (type === "probDesc") arr.sort((a, b) => b.prob - a.prob);

  probsGlobal = arr.map(x => x.prob);
  customerIds = arr.map(x => x.id);

  renderTable();
}

/* ===================================================
   FILTER CHURN / NOT CHURN
=================================================== */

function filterChurn(type) {
  const indices = probsGlobal
    .map((p, i) => {
      if (type === "churn" && p >= 0.5) return i;
      if (type === "not" && p < 0.5) return i;
      if (type === "all") return i;
      return -1;
    })
    .filter(i => i !== -1);

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

  indices.forEach((i) => {
    const p = probsGlobal[i];
    const prob = (p * 100).toFixed(2);
    const label = p >= 0.5 ? "Churned" : "Not Churned";
    const rowClass = p >= 0.5 ? "row-churn" : "row-safe";
    const badgeClass = p >= 0.5 ? "badge churn" : "badge not-churn";

    html += `
      <tr class="${rowClass}">
        <td>${i + 1}</td>
        <td>${customerIds[i]}</td>
        <td>${prob}%</td>
        <td><span class="${badgeClass}">${label}</span></td>
      </tr>
    `;
  });

  html += "</tbody>";
  resultsTable.innerHTML = html;
}

/* ===================================================
   SUMMARY CARDS
=================================================== */

function updateSummary() {
  const total = probsGlobal.length;
  const churnCount = probsGlobal.filter(p => p >= 0.5).length;

  sumTotal.textContent = total;
  sumChurn.textContent = churnCount;
  sumRate.textContent = ((churnCount / total) * 100).toFixed(2) + "%";
}

/* ===================================================
   HISTOGRAM
=================================================== */

function renderHistogram() {
  if (histChart) histChart.destroy();

  const bins = Array(10).fill(0);
  probsGlobal.forEach(p => bins[Math.min(9, Math.floor(p * 10))]++);

  histChart = new Chart(document.getElementById("histChart"), {
    type: "bar",
    data: {
      labels: [
        "0–10%", "10–20%", "20–30%", "30–40%", "40–50%",
        "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"
      ],
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
   INIT
=================================================== */

document.addEventListener("DOMContentLoaded", () => {
  log("Initializing app…");
  loadModel();
});
