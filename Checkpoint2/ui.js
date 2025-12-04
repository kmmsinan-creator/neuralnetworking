// ui.js – UI helpers. Exposed via window.UI

const statusEl = document.getElementById("status");
const resultsContainer = document.getElementById("resultsContainer");
const singleForm = document.getElementById("singleForm");
const singleResult = document.getElementById("singleResult");

function setStatus(message) {
  if (statusEl) statusEl.textContent = message;
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

  rows.forEach((r, idx) => {
    const prob = r.probability || 0;
    const label = r.prediction || "Not Churned";
    const isChurn = label === "Churned";
    const cls = isChurn ? "tag tag-churn" : "tag tag-nochurn";

    html += `
      <tr>
        <td>${idx + 1}</td>
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
    input.dataset.index = idx;

    row.appendChild(label);
    row.appendChild(input);
    singleForm.appendChild(row);
  });
}

function getSingleCustomerValues(featureNames, means) {
  const values = new Float32Array(featureNames.length);

  featureNames.forEach((_, idx) => {
    const inp = document.getElementById(`feat_${idx}`);
    let v = parseFloat(inp?.value);
    if (Number.isNaN(v)) v = means[idx];
    values[idx] = v;
  });

  return values;
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

window.UI = {
  setStatus,
  renderBatchResults,
  buildSingleCustomerForm,
  getSingleCustomerValues,
  showSinglePrediction
};
