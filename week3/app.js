// Full updated app.js — includes robust preprocess & createModel fixes

// ==================== GLOBALS ====================
let trainData = null, testData = null;
let preprocessedTrainData = null, preprocessedTestData = null;
let model = null, trainingHistory = null;
let validationData = null, validationLabels = null, validationPredictions = null, testPredictions = null;
let visorOpened = false;

const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';

// ==================== FILE LOADING ====================
async function loadData() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];
  const statusDiv = document.getElementById('data-status');

  if (!trainFile || !testFile) {
    alert('Please upload both train.csv and test.csv');
    return;
  }

  statusDiv.textContent = 'Loading data...';

  try {
    const trainText = await readFile(trainFile);
    const testText = await readFile(testFile);

    trainData = parseCSV(trainText);
    testData = parseCSV(testText);

    statusDiv.innerHTML = `✅ Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
    document.getElementById('inspect-btn').disabled = false;

    console.log('Sample train row:', trainData[0]);
    console.log('Sample test row:', testData[0]);
  } catch (err) {
    statusDiv.innerHTML = `❌ Error loading data: ${err.message}`;
    console.error(err);
  }
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = e => resolve(e.target.result);
    fr.onerror = () => reject(new Error('Failed to read file'));
    fr.readAsText(file);
  });
}

// ==================== CSV PARSING (PapaParse) ====================
function parseCSV(text) {
  const results = Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true
  });
  if (results.errors && results.errors.length) {
    console.warn('PapaParse errors/warnings:', results.errors);
  }
  return results.data;
}

// ==================== INSPECT & VISUALIZE ====================
function inspectData() {
  if (!trainData || trainData.length === 0) {
    alert('Please load data first.');
    return;
  }

  const previewDiv = document.getElementById('data-preview');
  previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
  previewDiv.appendChild(createTable(trainData.slice(0, 10)));

  // stats
  const statsDiv = document.getElementById('data-stats');
  const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
  const survivalCount = trainData.filter(r => r[TARGET_FEATURE] === 1).length;
  const survivalRate = ((survivalCount / trainData.length) * 100).toFixed(2);
  let missingHtml = '<h4>Missing Values Percentage:</h4><ul>';
  Object.keys(trainData[0]).forEach(k => {
    const missing = trainData.filter(r => r[k] === null || r[k] === undefined).length;
    missingHtml += `<li>${k}: ${((missing / trainData.length) * 100).toFixed(2)}%</li>`;
  });
  missingHtml += '</ul>';

  statsDiv.innerHTML = `<p>${shapeInfo}</p><p>Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)</p>${missingHtml}`;

  createVisualizations();
  document.getElementById('preprocess-btn').disabled = false;
}

function createTable(data) {
  const table = document.createElement('table');
  const headerTr = document.createElement('tr');
  Object.keys(data[0]).forEach(k => {
    const th = document.createElement('th');
    th.textContent = k;
    headerTr.appendChild(th);
  });
  table.appendChild(headerTr);

  data.forEach(row => {
    const tr = document.createElement('tr');
    Object.values(row).forEach(val => {
      const td = document.createElement('td');
      td.textContent = (val === null || val === undefined) ? 'NULL' : String(val);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });

  return table;
}

function createVisualizations() {
  if (!trainData) return;

  const chartsDiv = document.getElementById('charts');
  chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';

  // Survival by Sex
  const bySex = {};
  trainData.forEach(r => {
    if (r.Sex !== undefined && r.Survived !== undefined) {
      if (!bySex[r.Sex]) bySex[r.Sex] = { surv: 0, total: 0 };
      bySex[r.Sex].total++;
      if (r.Survived === 1) bySex[r.Sex].surv++;
    }
  });
  const sexData = Object.entries(bySex).map(([k, v]) => ({ x: k, y: (v.surv / v.total) * 100 }));

  tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Charts' }, sexData, { xLabel: 'Sex', yLabel: 'Survival Rate (%)' });

  // Survival by Pclass
  const byClass = {};
  trainData.forEach(r => {
    if (r.Pclass !== undefined && r.Survived !== undefined) {
      if (!byClass[r.Pclass]) byClass[r.Pclass] = { surv: 0, total: 0 };
      byClass[r.Pclass].total++;
      if (r.Survived === 1) byClass[r.Pclass].surv++;
    }
  });
  const classData = Object.entries(byClass).map(([k, v]) => ({ x: `Class ${k}`, y: (v.surv / v.total) * 100 }));
  tfvis.render.barchart({ name: 'Survival Rate by Passenger Class', tab: 'Charts' }, classData, { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' });

  chartsDiv.innerHTML += '<p>Charts are shown in the tfjs-vis visor (floating panel). Press <b>Shift+V</b> to toggle if hidden.</p>';

  // ensure visor appears
  try {
    if (!visorOpened && tfvis && tfvis.visor) {
      tfvis.visor().open();
      visorOpened = true;
    }
  } catch (err) {
    console.warn('tfvis visor open failed:', err);
  }
}

// ==================== PREPROCESSING (with validation) ====================
function preprocessData() {
  const out = document.getElementById('preprocessing-output');
  out.innerHTML = 'Preprocessing data...';

  try {
    if (!trainData || !testData) {
      out.innerHTML = '<p style="color:red">❌ Data not loaded. Please load train/test CSV first.</p>';
      return;
    }
    // compute imputation values
    const ageVals = trainData.map(r => r.Age).filter(v => v !== null && v !== undefined && !isNaN(v));
    const fareVals = trainData.map(r => r.Fare).filter(v => v !== null && v !== undefined && !isNaN(v));
    const embVals = trainData.map(r => r.Embarked).filter(v => v !== null && v !== undefined);

    const ageMedian = calculateMedian(ageVals);
    const fareMedian = calculateMedian(fareVals);
    const embarkedMode = calculateMode(embVals);

    // build features arrays (raw JS arrays first)
    const trainFeatures = [];
    const trainLabels = [];
    for (let i = 0; i < trainData.length; i++) {
      const r = trainData[i];
      const feats = extractFeatures(r, ageMedian, fareMedian, embarkedMode);
      trainFeatures.push(feats);
      // ensure label present
      trainLabels.push(Number((r[TARGET_FEATURE] !== undefined && r[TARGET_FEATURE] !== null) ? r[TARGET_FEATURE] : 0));
    }

    // Validate feature lengths are consistent
    const lengths = trainFeatures.map(f => f.length);
    const uniqueLengths = [...new Set(lengths)];
    if (uniqueLengths.length > 1) {
      // produce informative message
      const counts = {};
      lengths.forEach((l, idx) => { counts[l] = (counts[l] || 0) + 1; });
      let msg = `<p style="color:red">❌ Inconsistent feature vector lengths detected after preprocessing.</p>`;
      msg += `<p>Different lengths: ${uniqueLengths.join(', ')}. Occurrences: ${JSON.stringify(counts)}</p>`;
      msg += `<p>Example row lengths (first 10): ${lengths.slice(0, 10).join(', ')}</p>`;
      msg += `<p>Please inspect raw CSV or ensure categorical values are consistent (e.g., Embarked values C/Q/S).</p>`;
      out.innerHTML = msg;
      console.error('Feature length inconsistency', counts);
      return;
    }

    // everything consistent → convert to tensors
    preprocessedTrainData = {
      features: tf.tensor2d(trainFeatures),
      labels: tf.tensor1d(trainLabels, 'float32')
    };

    // preprocess test
    const testFeatures = [];
    const testIds = [];
    for (let i = 0; i < testData.length; i++) {
      const r = testData[i];
      testFeatures.push(extractFeatures(r, ageMedian, fareMedian, embarkedMode));
      testIds.push(r[ID_FEATURE]);
    }
    preprocessedTestData = { features: testFeatures, passengerIds: testIds };

    out.innerHTML = `<p style="color:green">✅ Preprocessing completed!</p>
      <p>Training features shape: [${preprocessedTrainData.features.shape}]</p>
      <p>Training labels shape: [${preprocessedTrainData.labels.shape}]</p>
      <p>Test features: ${preprocessedTestData.features.length} samples</p>`;

    document.getElementById('create-model-btn').disabled = false;
  } catch (err) {
    out.innerHTML = `<p style="color:red">❌ Error during preprocessing: ${err.message}</p>`;
    console.error('Preprocess error:', err);
  }
}

// ==================== FEATURE EXTRACTION & HELPERS ====================
function extractFeatures(r, ageMedian, fareMedian, embarkedMode) {
  const age = (r.Age !== undefined && r.Age !== null && !isNaN(r.Age)) ? r.Age : ageMedian;
  const fare = (r.Fare !== undefined && r.Fare !== null && !isNaN(r.Fare)) ? r.Fare : fareMedian;
  const embarked = (r.Embarked !== undefined && r.Embarked !== null) ? r.Embarked : embarkedMode;

  const ageStd = (age - ageMedian) / (calculateStdDev(trainData.map(x => x.Age).filter(v => v != null)) || 1);
  const fareStd = (fare - fareMedian) / (calculateStdDev(trainData.map(x => x.Fare).filter(v => v != null)) || 1);

  let feats = [ageStd, fareStd, (r.SibSp || 0), (r.Parch || 0)];
  // Pclass one-hot [1,2,3]
  feats = feats.concat(oneHotEncode(r.Pclass, [1, 2, 3]));
  // Sex one-hot
  feats = feats.concat(oneHotEncode(r.Sex, ['male', 'female']));
  // Embarked one-hot
  feats = feats.concat(oneHotEncode(embarked, ['C', 'Q', 'S']));

  // family features
  if (document.getElementById('add-family-features')?.checked) {
    const familySize = (r.SibSp || 0) + (r.Parch || 0) + 1;
    const isAlone = familySize === 1 ? 1 : 0;
    feats.push(familySize, isAlone);
  }

  return feats;
}

function calculateMedian(arr) {
  if (!arr || arr.length === 0) return 0;
  const a = arr.slice().sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 === 0 ? (a[mid - 1] + a[mid]) / 2 : a[mid];
}
function calculateMode(arr) {
  if (!arr || arr.length === 0) return null;
  const count = {};
  arr.forEach(v => count[v] = (count[v] || 0) + 1);
  let best = null, bestCount = 0;
  for (const k in count) {
    if (count[k] > bestCount) { best = k; bestCount = count[k]; }
  }
  return best;
}
function calculateStdDev(arr) {
  if (!arr || arr.length === 0) return 0;
  const n = arr.length;
  const mean = arr.reduce((s, v) => s + v, 0) / n;
  const variance = arr.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / n;
  return Math.sqrt(variance);
}
function oneHotEncode(value, categories) {
  return categories.map(c => (c === value ? 1 : 0));
}

// ==================== CREATE MODEL (ROBUST) ====================
function createModel() {
  const summaryDiv = document.getElementById('model-summary');
  summaryDiv.innerHTML = '<p>Creating model...</p>';

  try {
    if (!preprocessedTrainData || !preprocessedTrainData.features) {
      summaryDiv.innerHTML = '<p style="color:red">❌ No preprocessed training data found. Run Preprocess first.</p>';
      return;
    }

    // ensure features tensor exists
    let featuresTensor = preprocessedTrainData.features;
    if (!featuresTensor || (Array.isArray(featuresTensor) && featuresTensor.length === 0)) {
      summaryDiv.innerHTML = '<p style="color:red">❌ Training features are empty or invalid.</p>';
      return;
    }

    // if features stored as JS array (should be tensor already), convert now
    if (!tf.Tensor.isTensor ? !(featuresTensor instanceof tf.Tensor) : !(featuresTensor instanceof tf.Tensor)) {
      // defensive: if it looks like an array of arrays, convert
      if (Array.isArray(featuresTensor)) {
        // check consistent lengths
        const firstLen = featuresTensor[0]?.length;
        if (featuresTensor.some(f => !Array.isArray(f) || f.length !== firstLen)) {
          summaryDiv.innerHTML = `<p style="color:red">❌ Inconsistent feature vector sizes detected. Cannot create model.</p>
            <p>Example lengths (first 10): ${featuresTensor.slice(0, 10).map(f => (Array.isArray(f) ? f.length : 'N/A')).join(', ')}</p>`;
          return;
        }
        featuresTensor = tf.tensor2d(featuresTensor);
        preprocessedTrainData.features = featuresTensor;
      } else {
        summaryDiv.innerHTML = '<p style="color:red">❌ Unexpected feature data type.</p>';
        return;
      }
    }

    // labels validation
    const labelsTensor = preprocessedTrainData.labels;
    if (!labelsTensor || labelsTensor.shape[0] !== featuresTensor.shape[0]) {
      summaryDiv.innerHTML = `<p style="color:red">❌ Labels length (${labelsTensor ? labelsTensor.shape[0] : 'N/A'}) does not match features (${featuresTensor.shape[0]}).</p>`;
      return;
    }

    const inputDim = featuresTensor.shape[1];
    if (!inputDim || isNaN(inputDim)) {
      summaryDiv.innerHTML = '<p style="color:red">❌ Invalid input dimension for model.</p>';
      return;
    }

    // build model
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [inputDim] }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    // show summary
    let html = '<h3>Model Summary</h3><ul>';
    model.layers.forEach((layer, i) => {
      html += `<li>Layer ${i + 1}: ${layer.getClassName()} — Output shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    html += `</ul><p>Total params: ${model.countParams()}</p>`;
    summaryDiv.innerHTML = html;

    document.getElementById('train-btn').disabled = false;
    summaryDiv.innerHTML += '<p style="color:green">✅ Model created. Click Train Model.</p>';
  } catch (err) {
    summaryDiv.innerHTML = `<p style="color:red">❌ Error creating model: ${err.message}</p>`;
    console.error('CreateModel error:', err);
  }
}

// ==================== TRAIN (kept typical) ====================
async function trainModel() {
  const statusDiv = document.getElementById('training-status');
  statusDiv.innerHTML = 'Training...';

  try {
    if (!model || !preprocessedTrainData) {
      alert('Please create model and preprocess data first.');
      return;
    }

    const total = preprocessedTrainData.features.shape[0];
    const nFeatures = preprocessedTrainData.features.shape[1];
    const splitIndex = Math.floor(total * 0.8);

    const trainX = preprocessedTrainData.features.slice([0, 0], [splitIndex, nFeatures]);
    const trainY = preprocessedTrainData.labels.slice([0], [splitIndex]);
    const valX = preprocessedTrainData.features.slice([splitIndex, 0], [total - splitIndex, nFeatures]);
    const valY = preprocessedTrainData.labels.slice([splitIndex], [total - splitIndex]);

    validationData = valX;
    validationLabels = valY;

    trainingHistory = await model.fit(trainX, trainY, {
      epochs: 30,
      batchSize: 32,
      validationData: [valX, valY],
      callbacks: [
        tfvis.show.fitCallbacks({ name: 'Training Performance' }, ['loss', 'accuracy', 'val_loss', 'val_accuracy'], { callbacks: ['onEpochEnd'] }),
        {
          onEpochEnd: (epoch, logs) => {
            const acc = logs.accuracy ?? logs.acc ?? 'N/A';
            const valAcc = logs.val_accuracy ?? logs.val_acc ?? 'N/A';
            statusDiv.innerHTML = `Epoch ${epoch + 1}: loss=${logs.loss?.toFixed(4) ?? 'N/A'}, acc=${(acc !== 'N/A') ? acc.toFixed(4) : acc}, val_loss=${logs.val_loss?.toFixed(4) ?? 'N/A'}, val_acc=${(valAcc !== 'N/A') ? valAcc.toFixed(4) : valAcc}`;
          }
        }
      ]
    });

    statusDiv.innerHTML += '<p style="color:green">✅ Training finished.</p>';
    validationPredictions = model.predict(validationData);

    document.getElementById('threshold-slider').disabled = false;
    const slider = document.getElementById('threshold-slider');
    slider.removeEventListener('input', updateMetrics);
    slider.addEventListener('input', updateMetrics);

    document.getElementById('predict-btn').disabled = false;
    await updateMetrics();
  } catch (err) {
    statusDiv.innerHTML = `<p style="color:red">❌ Error during training: ${err.message}</p>`;
    console.error('Train error:', err);
  }
}

// ==================== METRICS, ROC, PREDICT, EXPORT (unchanged logic) ====================
async function updateMetrics() {
  if (!validationPredictions || !validationLabels) return;
  const threshold = parseFloat(document.getElementById('threshold-slider').value || 0.5);
  document.getElementById('threshold-value').textContent = threshold.toFixed(2);

  const predRaw = validationPredictions.arraySync();
  const trueRaw = validationLabels.arraySync();
  const preds = predRaw.map(p => Array.isArray(p) ? p[0] : p);
  const trues = trueRaw.map(v => Array.isArray(v) ? v[0] : v);

  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < preds.length; i++) {
    const p = preds[i] >= threshold ? 1 : 0;
    const a = trues[i];
    if (p === 1 && a === 1) tp++;
    else if (p === 0 && a === 0) tn++;
    else if (p === 1 && a === 0) fp++;
    else if (p === 0 && a === 1) fn++;
  }

  const cmDiv = document.getElementById('confusion-matrix');
  cmDiv.innerHTML = `
    <table>
      <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
      <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
      <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
    </table>`;

  const precision = (tp + fp) ? tp / (tp + fp) : 0;
  const recall = (tp + fn) ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) ? 2 * (precision * recall) / (precision + recall) : 0;
  const accuracy = (tp + tn + fp + fn) ? (tp + tn) / (tp + tn + fp + fn) : 0;

  const metricsDiv = document.getElementById('performance-metrics');
  metricsDiv.innerHTML = `
    <p>Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
    <p>Precision: ${precision.toFixed(4)}</p>
    <p>Recall: ${recall.toFixed(4)}</p>
    <p>F1 Score: ${f1.toFixed(4)}</p>`;

  await plotROC(trues, preds);
}

async function plotROC(trueLabels, predictions) {
  const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
  const roc = [];
  thresholds.forEach(t => {
    let tp = 0, fn = 0, fp = 0, tn = 0;
    for (let i = 0; i < predictions.length; i++) {
      const p = predictions[i] >= t ? 1 : 0;
      if (trueLabels[i] === 1) { if (p === 1) tp++; else fn++; }
      else { if (p === 1) fp++; else tn++; }
    }
    const tpr = (tp + fn) ? tp / (tp + fn) : 0;
    const fpr = (fp + tn) ? fp / (fp + tn) : 0;
    roc.push({ x: fpr, y: tpr });
  });

  // approximate AUC
  let auc = 0;
  for (let i = 1; i < roc.length; i++) {
    const x1 = roc[i - 1].x, x2 = roc[i].x;
    const y1 = roc[i - 1].y, y2 = roc[i].y;
    auc += (x2 - x1) * (y1 + y2) / 2;
  }

  tfvis.render.linechart({ name: 'ROC Curve', tab: 'Evaluation' }, { values: roc }, { xLabel: 'FPR', yLabel: 'TPR' });
  const metricsDiv = document.getElementById('performance-metrics');
  metricsDiv.innerHTML += `<p>AUC: ${auc.toFixed(4)}</p>`;
}

async function predict() {
  if (!model || !preprocessedTestData) {
    alert('Please train model and preprocess data first.');
    return;
  }
  const out = document.getElementById('prediction-output');
  out.innerHTML = 'Predicting...';
  try {
    const testX = tf.tensor2d(preprocessedTestData.features);
    testPredictions = model.predict(testX);
    const raw = testPredictions.arraySync().map(p => Array.isArray(p) ? p[0] : p);
    const results = preprocessedTestData.passengerIds.map((id, i) => ({ PassengerId: id, Survived: raw[i] >= 0.5 ? 1 : 0, Probability: raw[i] }));
    out.innerHTML = '<h3>Predictions (first 10)</h3>';
    out.appendChild(createPredictionTable(results.slice(0, 10)));
    document.getElementById('export-btn').disabled = false;
  } catch (err) {
    out.innerHTML = `<p style="color:red">❌ Prediction error: ${err.message}</p>`;
    console.error(err);
  }
}

function createPredictionTable(rows) {
  const t = document.createElement('table');
  const head = document.createElement('tr');
  ['PassengerId', 'Survived', 'Probability'].forEach(h => { const th = document.createElement('th'); th.textContent = h; head.appendChild(th); });
  t.appendChild(head);
  rows.forEach(r => {
    const tr = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(k => {
      const td = document.createElement('td');
      td.textContent = (k === 'Probability') ? r[k].toFixed(4) : r[k];
      tr.appendChild(td);
    });
    t.appendChild(tr);
  });
  return t;
}

async function exportResults() {
  if (!testPredictions || !preprocessedTestData) {
    alert('Please predict first.');
    return;
  }
  const statusDiv = document.getElementById('export-status');
  statusDiv.innerHTML = 'Exporting...';
  try {
    const raw = testPredictions.arraySync().map(p => Array.isArray(p) ? p[0] : p);
    let csvSub = 'PassengerId,Survived\n';
    preprocessedTestData.passengerIds.forEach((id, i) => csvSub += `${id},${raw[i] >= 0.5 ? 1 : 0}\n`);

    let csvProb = 'PassengerId,Probability\n';
    preprocessedTestData.passengerIds.forEach((id, i) => csvProb += `${id},${raw[i].toFixed(6)}\n`);

    const a1 = document.createElement('a');
    a1.href = URL.createObjectURL(new Blob([csvSub], { type: 'text/csv' }));
    a1.download = 'submission.csv';
    a1.click();

    const a2 = document.createElement('a');
    a2.href = URL.createObjectURL(new Blob([csvProb], { type: 'text/csv' }));
    a2.download = 'probabilities.csv';
    a2.click();

    await model.save('downloads://titanic-tfjs-model');

    statusDiv.innerHTML = '<p style="color:green">✅ Export complete (CSV files & model saved).</p>';
  } catch (err) {
    statusDiv.innerHTML = `<p style="color:red">❌ Export error: ${err.message}</p>`;
    console.error(err);
  }
}
