let model = null;
let inputRow = null; // numeric feature vector
let numFeatures = null;

const fileInput = document.getElementById('fileInput');
const fileNameSpan = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBtn');
const modelStatus = document.getElementById('modelStatus');
const logEl = document.getElementById('log');
const resultCard = document.getElementById('resultCard');
const predictedLabelSpan = document.getElementById('predictedLabel');
const predictedProbSpan = document.getElementById('predictedProb');

/* ------------------ Logging helper ------------------ */

function log(message) {
  console.log(message);
  logEl.textContent += message + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

/* ------------------ Model loading ------------------ */

async function loadModel() {
  try {
    log('Loading TF.js model from model/model.json …');
    // If your model folder name is different, adjust this path
    model = await tf.loadLayersModel('model/model.json');
    numFeatures = model.inputs[0].shape[1];
    log(`✅ Model loaded successfully. Input features: ${numFeatures}`);
    modelStatus.textContent = 'Model loaded';
    modelStatus.classList.remove('status-loading');
    modelStatus.classList.add('status-ready');
    predictBtn.disabled = false;
  } catch (err) {
    console.error(err);
    log('❌ Error loading model. Check console and model path.');
    modelStatus.textContent = 'Model load failed';
  }
}

/* ------------------ CSV upload handling ------------------ */

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) {
    fileNameSpan.textContent = 'No file selected';
    inputRow = null;
    return;
  }

  fileNameSpan.textContent = file.name;
  log(`Selected file: ${file.name}`);

  Papa.parse(file, {
    complete: (results) => {
      const rows = results.data.filter((r) => r && r.length > 0);

      if (rows.length < 2) {
        log('❌ CSV must contain a header row and at least one data row.');
        inputRow = null;
        return;
      }

      // We assume:
      // row[0] = header, row[1] = single data row
      let row = rows[1];

      // Remove any empty strings at the end
      row = row.filter((v) => v !== '');

      const numericRow = row.map((v) => Number(v));
      if (numericRow.some((v) => Number.isNaN(v))) {
        log('❌ Some values are not numeric. Ensure the CSV is preprocessed numeric features.');
        inputRow = null;
        return;
      }

      if (numFeatures !== null && numericRow.length !== numFeatures) {
        log(
          `❌ Feature length mismatch. Model expects ${numFeatures} features, but CSV row has ${numericRow.length}.`
        );
        inputRow = null;
        return;
      }

      inputRow = numericRow;
      log('✅ Parsed CSV row successfully. Ready to predict.');
    },
    header: false,
    skipEmptyLines: true,
  });
});

/* ------------------ Prediction ------------------ */

predictBtn.addEventListener('click', async () => {
  if (!model) {
    log('❌ Model not loaded yet.');
    return;
  }
  if (!inputRow) {
    log('❌ No valid input row. Please upload a proper CSV.');
    return;
  }

  try {
    log('Running prediction …');

    // [1, num_features] tensor
    const inputTensor = tf.tensor2d([inputRow]);

    const output = model.predict(inputTensor);
    const data = await output.data();
    const prob = data[0];

    inputTensor.dispose();
    output.dispose();

    const probPercent = (prob * 100).toFixed(2);
    const label = prob >= 0.5 ? 'Churned' : 'Not Churned';

    // Update UI
    resultCard.classList.remove('hidden');
    predictedProbSpan.textContent = `${probPercent}%`;

    if (label === 'Churned') {
      predictedLabelSpan.textContent = 'Churned';
      predictedLabelSpan.classList.remove('not-churn');
      predictedLabelSpan.classList.add('churn');
    } else {
      predictedLabelSpan.textContent = 'Not Churned';
      predictedLabelSpan.classList.remove('churn');
      predictedLabelSpan.classList.add('not-churn');
    }

    log(`✅ Prediction complete. Label: ${label}, Probability: ${probPercent}%`);
  } catch (err) {
    console.error(err);
    log('❌ Error during prediction. See console for details.');
  }
});

/* ------------------ Init ------------------ */

document.addEventListener('DOMContentLoaded', () => {
  log('Initializing app …');
  loadModel();
});
