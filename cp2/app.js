let model = null;
let dataRows = [];        // numeric rows from CSV
let featureNames = [];    // header columns from CSV
let numFeaturesModel = null;

const fileInput = document.getElementById('fileInput');
const fileNameSpan = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBtn');
const modelStatus = document.getElementById('modelStatus');
const logEl = document.getElementById('log');
const resultCard = document.getElementById('resultCard');
const resultsTable = document.getElementById('resultsTable');
const totalCustomersSpan = document.getElementById('totalCustomers');
const totalChurnersSpan = document.getElementById('totalChurners');

function log(message) {
  console.log(message);
  logEl.textContent += message + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

/* ------------ Load TF.js model ------------ */

async function loadModel() {
  try {
    const modelUrl = './model/model.json';
    log(`Loading TF.js model from ${modelUrl} …`);

    model = await tf.loadLayersModel(modelUrl);
    numFeaturesModel = model.inputs[0].shape[1];

    log(`✅ Model loaded. Input features expected: ${numFeaturesModel}`);
    modelStatus.textContent = 'Model loaded';
    modelStatus.classList.remove('status-loading');
    modelStatus.classList.add('status-ready');
    predictBtn.disabled = false;
  } catch (err) {
    console.error(err);
    const msg = err && err.message ? err.message : String(err);
    log('❌ Error loading model: ' + msg);
    modelStatus.textContent = 'Model load failed';
  }
}

/* ------------ CSV upload handling (multi-row) ------------ */

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) {
    fileNameSpan.textContent = 'No file selected';
    dataRows = [];
    return;
  }

  fileNameSpan.textContent = file.name;
  log(`Selected file: ${file.name}`);

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      const rows = results.data;

      if (!rows || rows.length === 0) {
        log('❌ CSV appears to be empty.');
        dataRows = [];
        return;
      }

      featureNames = Object.keys(rows[0]);
      log(`CSV feature columns (${featureNames.length}): ${featureNames.join(', ')}`);

      const numericRows = [];
      let skipped = 0;

      rows.forEach((row, idx) => {
        const values = featureNames.map((name) => row[name]);

        const allEmpty = values.every(
          (v) => v === '' || v === null || v === undefined
        );
        if (allEmpty) {
          skipped++;
          return;
        }

        const numeric = values.map((v) => Number(v));
        const hasNaN = numeric.some((v) => Number.isNaN(v));

        if (hasNaN) {
          skipped++;
          log(`⚠️ Skipping row ${idx + 2} (non-numeric value detected).`);
          return;
        }

        numericRows.push(numeric);
      });

      if (numericRows.length === 0) {
        log('❌ No valid numeric rows found in CSV. Ensure data is already encoded & scaled.');
        dataRows = [];
        return;
      }

      if (numFeaturesModel !== null && featureNames.length !== numFeaturesModel) {
        log(
          `⚠️ WARNING: Model expects ${numFeaturesModel} features, ` +
          `but CSV has ${featureNames.length}. Check preprocessing and column order.`
        );
      }

      dataRows = numericRows;
      log(`✅ Parsed ${dataRows.length} valid customer rows (skipped ${skipped}).`);

      // Reset previous results
      resultCard.classList.add('hidden');
      resultsTable.innerHTML = '';
      totalCustomersSpan.textContent = '0';
      totalChurnersSpan.textContent = '0';
    },
    error: (err) => {
      console.error(err);
      log('❌ Error parsing CSV file.');
    },
  });
});

/* ------------ Predict for ALL rows ------------ */

predictBtn.addEventListener('click', async () => {
  if (!model) {
    log('❌ Model not loaded yet.');
    return;
  }
  if (!dataRows || dataRows.length === 0) {
    log('❌ No valid rows parsed. Please upload a proper preprocessed CSV.');
    return;
  }

  try {
    log(`Running prediction for ${dataRows.length} customers …`);

    const inputTensor = tf.tensor2d(dataRows); // shape [N, num_features]
    const output = model.predict(inputTensor);
    const probs = await output.data();        // Float32Array of length N

    inputTensor.dispose();
    output.dispose();

    let html = `
      <thead>
        <tr>
          <th class="col-idx">#</th>
          <th class="col-prob">Churn Prob.</th>
          <th class="col-label">Class</th>
        </tr>
      </thead>
      <tbody>
    `;

    let churnCount = 0;

    for (let i = 0; i < probs.length; i++) {
      const prob = probs[i];
      const probPercent = (prob * 100).toFixed(2);
      const label = prob >= 0.5 ? 'Churned' : 'Not Churned';
      const badgeClass = label === 'Churned' ? 'badge churn' : 'badge not-churn';

      if (label === 'Churned') churnCount++;

      html += `
        <tr>
          <td>${i + 1}</td>
          <td>${probPercent}%</td>
          <td><span class="${badgeClass}">${label}</span></td>
        </tr>
      `;
    }

    html += '</tbody>';
    resultsTable.innerHTML = html;

    totalCustomersSpan.textContent = String(probs.length);
    totalChurnersSpan.textContent = String(churnCount);

    resultCard.classList.remove('hidden');

    log(`✅ Prediction complete. ${churnCount} out of ${probs.length} customers predicted to churn.`);
  } catch (err) {
    console.error(err);
    const msg = err && err.message ? err.message : String(err);
    log('❌ Error during prediction: ' + msg);
  }
});

/* ------------ Init ------------ */

document.addEventListener('DOMContentLoaded', () => {
  log('Initializing app …');
  loadModel();
});
