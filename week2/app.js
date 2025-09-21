// app.js - Titanic Survival Classifier
// Uses TensorFlow.js + tfjs-vis, runs entirely in the browser

// --- Globals ---
let trainData = [], testData = [];
let xTrain, yTrain, xVal, yVal, testX;
let featureNames = [];
let model;
let stopTraining = false;

// --- Helpers ---
const $ = id => document.getElementById(id);
const setDisabled = (id, state) => { $(id).disabled = state; };

// --- CSV Loading ---
$('load-data-btn').addEventListener('click', async () => {
  try {
    trainData = await loadCSV($('trainFile'));
    testData = await loadCSV($('testFile'));
    $('data-status').textContent = `Train: ${trainData.length} rows, Test: ${testData.length} rows`;
    setDisabled('inspect-btn', false);
  } catch (err) {
    alert("Error loading CSV: " + err);
  }
});

function loadCSV(input) {
  return new Promise((resolve, reject) => {
    if (!input.files[0]) return reject("No file chosen");
    Papa.parse(input.files[0], {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: res => resolve(res.data),
      error: err => reject(err)
    });
  });
}

// --- Inspect ---
$('inspect-btn').addEventListener('click', () => {
  const preview = trainData.slice(0, 5);
  $('data-preview').textContent = JSON.stringify(preview, null, 2);

  const cols = Object.keys(trainData[0]);
  const missing = cols.map(c => ({
    col: c,
    missing: trainData.filter(r => r[c] === null || r[c] === "").length
  }));
  $('data-stats').textContent = JSON.stringify(missing, null, 2);

  // Survival by Sex
  const survBySex = {};
  trainData.forEach(r => {
    if (r.Survived != null) {
      survBySex[r.Sex] = survBySex[r.Sex] || { surv: 0, total: 0 };
      survBySex[r.Sex].total++;
      if (r.Survived === 1) survBySex[r.Sex].surv++;
    }
  });
  const sexValues = Object.keys(survBySex).map(k => ({
    x: k,
    y: survBySex[k].surv / survBySex[k].total
  }));
  tfvis.render.barchart({ name: 'Survival by Sex', tab: 'Inspect' }, sexValues);

  setDisabled('preprocess-btn', false);
});

// --- Preprocess ---
$('preprocess-btn').addEventListener('click', () => {
  const addFamily = $('toggle-family').checked;

  const imputeMedian = (arr, key) => {
    const vals = arr.map(r => r[key]).filter(v => v != null);
    vals.sort((a, b) => a - b);
    return vals[Math.floor(vals.length / 2)];
  };
  const imputeMode = (arr, key) => {
    const counts = {};
    arr.forEach(r => { counts[r[key]] = (counts[r[key]] || 0) + 1; });
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  };

  const ageMed = imputeMedian(trainData, 'Age');
  const embMode = imputeMode(trainData, 'Embarked');
  trainData.forEach(r => {
    if (!r.Age) r.Age = ageMed;
    if (!r.Embarked) r.Embarked = embMode;
  });
  testData.forEach(r => {
    if (!r.Age) r.Age = ageMed;
    if (!r.Embarked) r.Embarked = embMode;
  });

  // Features
  featureNames = ['Age', 'Fare'];
  if (addFamily) { featureNames.push('FamilySize', 'IsAlone'); }
  ['Pclass', 'Sex', 'Embarked'].forEach(cat => {
    const uniq = [...new Set(trainData.map(r => r[cat]))];
    uniq.forEach(u => featureNames.push(`${cat}_${u}`));
  });

  const proc = rows => rows.map(r => {
    let feats = [];
    let age = (r.Age - ageMed) / ageMed;
    let fare = (r.Fare - (r.Fare || 0)) / (r.Fare || 1);
    feats.push(age, fare);
    if (addFamily) {
      const fs = r.SibSp + r.Parch + 1;
      feats.push(fs, fs === 1 ? 1 : 0);
    }
    ['Pclass', 'Sex', 'Embarked'].forEach(cat => {
      const uniq = [...new Set(trainData.map(rr => rr[cat]))];
      uniq.forEach(u => feats.push(r[cat] === u ? 1 : 0));
    });
    return feats;
  });

  const X = proc(trainData);
  const Y = trainData.map(r => r.Survived);
  const N = X.length, split = Math.floor(N * 0.8);
  xTrain = tf.tensor2d(X.slice(0, split));
  yTrain = tf.tensor2d(Y.slice(0, split), [split, 1]);
  xVal = tf.tensor2d(X.slice(split));
  yVal = tf.tensor2d(Y.slice(split), [N - split, 1]);

  testX = tf.tensor2d(proc(testData));

  $('preprocessing-output').textContent = `Features: ${featureNames.join(", ")}\nTrain shape: ${xTrain.shape}`;
  setDisabled('create-model-btn', false);
});

// --- Model ---
$('create-model-btn').addEventListener('click', () => {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [featureNames.length] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  $('model-summary').textContent = "Model created.";
  setDisabled('train-btn', false);
});

// --- Train ---
$('train-btn').addEventListener('click', async () => {
  stopTraining = false;
  setDisabled('stop-train-btn', false);
  await model.fit(xTrain, yTrain, {
    epochs: 50, batchSize: 32, validationData: [xVal, yVal],
    callbacks: tfvis.show.fitCallbacks({ name: 'Training', tab: 'Training' }, ['loss', 'val_loss', 'acc', 'val_acc'], { callbacks: ['onEpochEnd'] })
  });
  setDisabled('stop-train-btn', true);
  setDisabled('predict-btn', false);
  setDisabled('export-btn', false);
  setDisabled('save-model-btn', false);
});

// --- Stop ---
$('stop-train-btn').addEventListener('click', () => stopTraining = true);

// --- Predict ---
$('predict-btn').addEventListener('click', async () => {
  const preds = model.predict(testX).arraySync().map(v => v[0]);
  $('prediction-output').textContent = preds.slice(0, 10).map(p => p.toFixed(3)).join(", ") + "...";
});

// --- Export ---
$('export-btn').addEventListener('click', () => {
  let csv = "PassengerId,Survived\n";
  testData.forEach((r, i) => {
    const p = model.predict(tf.tensor2d([testX.arraySync()[i]])).arraySync()[0][0];
    csv += `${r.PassengerId},${p >= 0.5 ? 1 : 0}\n`;
  });
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'submission.csv';
  a.click();
});

// --- Save Model ---
$('save-model-btn').addEventListener('click', async () => {
  await model.save('downloads://titanic-tfjs');
});
