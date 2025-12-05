let model = null;
let dataRows = [];       // numeric feature rows
let featureNames = [];   // CSV header
let customerIDs = [];    // first column from CSV
let numFeaturesModel = 30; // your model was trained with 30 features

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

/* ------------ Build the model architecture in JS ------------ */

function buildModel() {
  const m = tf.sequential();

  // Must match your Python model architecture EXACTLY:
  // Dense(64, relu, input_shape=(30,))
  m.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
    inputShape: [numFeaturesModel],
    name: 'dense_11'
  }));

  m.add(tf.layers.dropout({
    rate: 0.3,
    name: 'dropout_8'
  }));

  m.add(tf.layers.dense({
    units: 32,
    activation: 'relu',
    name: 'dense_12'
  }));

  m.add(tf.layers.dropout({
    rate: 0.2,
    name: 'dropout_9'
  }));

  m.add(tf.layers.dense({
    units: 16,
    activation: 'relu',
    name: 'dense_13'
  }));

  m.add(tf.layers.dropout({
    rate: 0.1,
    name: 'dropout_10'
  }));

  m.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
    name: 'output'
  }));

  return m;
}

/* ------------ Load weights.json and set model weights ------------ */

async function loadModel() {
  try {
    log('Loading model weights from ./model/weights.json …');

    // Build JS model with same architecture
    model = buildModel();

    // Load weights json
    const response = await fetch('./model/weights.json');
    if (!response.ok) {
      throw new Error('Could not fetch weights.json');
    }
    const data = await response.json();

    const shapes = data.shapes;
    const weights = data.weights;

    if (!Array.isArray(shapes) || !Array.isArray(weights) || shapes.length !== weights.length) {
      throw new Error('Invalid weights.json format');
    }

    // Convert loaded weights to tensors with correct shapes
    const weightTensors = weights.map((wArr, idx) => {
      const shape = shapes[idx];
      return tf.tensor(wArr, shape);
    });

    model.setWeights(weightTensors);

    log(`✅ Model weights loaded. Input features expected: ${numFeaturesModel}`);
    modelStatus.textContent = 'Model loaded';
    modelStatus.classList.remove('status-loading');
    modelStatus.classList.add('status-ready');
    predictBtn.disabled = false;
  } catch (err) {
    console.error(err);
    log('❌ Error loading model: ' + (err.message || String(err)));
    modelStatus.textContent = 'Model load failed';
  }
}

/* ------------ CSV upload handling (multi-row, with CustomerID) ------------ */

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) {
    fileNameSpan.textContent = 'No file selected';
    dataRows = [];
    customerIDs = [];
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
        customerIDs = [];
        return;
      }

      featureNames = Object.keys(rows[0]);
      log(`CSV columns (${featureNames.length}): ${featureNames.join(', ')}`);

      const numericRows = [];
      const ids = [];
      let skipped = 0;

      rows.forEach((row, idx) => {
        // Extract values in header order
        const values = featureNames.map((name) => row[name]);

        // Row completely empty?
        const allEmpty = values.every(
          (v) => v === '' || v === null || v === undefined
        );
        if (allEmpty) {
          skipped++;
          return;
        }

        // First column: CustomerID (keep as string)
        const customerId = values[0];

        // Remaining columns: numeric features used by the model
        const featureVals = values.slice(1).map((v) => Number(v));
        const hasNaN = featureVals.some((v) => Number.isNaN(v));

        if (hasNaN) {
          skipped++;
          log(`⚠️ Skipping row ${idx + 2} (non-numeric feature value detected).`);
          return;
        }

        ids.push(customerId);
        numericRows.push(featureVals);
      });

      if (numericRows.length === 0) {
        log('❌ No valid numeric rows found in CSV. Make sure features are already encoded & scaled.');
        dataRows = [];
        customerIDs = [];
        return;
      }

      if (featureNames.length - 1 !== numFeaturesModel) {
        log(
          `⚠️ WARNING: Model expects ${numFeaturesModel} feature columns, ` +
          `but CSV has ${featureNames.length - 1} feature columns (excluding CustomerID). ` +
          `Check preprocessing and column order.`
        );
      }

      dataRows = numericRows;
      customerIDs = ids;

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

    const inputTensor = tf.tensor2d(dataRows); // [N, numFeaturesModel]
    const output = model.predict(inputTensor);
    const probs = await output.data();

    inputTensor.dispose();
    output.dispose();

    let html = `
      <thead>
        <tr>
          <th class="col-idx">Customer ID</th>
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
          <td>${customerIDs[i]}</td>
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
    log('❌ Error during prediction: ' + (err.message || String(err)));
  }
});

/* ------------ Init ------------ */

document.addEventListener('DOMContentLoaded', () => {
  log('Initializing app …');
  loadModel();
});
