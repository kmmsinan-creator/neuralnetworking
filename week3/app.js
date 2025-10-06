// app.js — Browser-based Titanic binary classifier with TensorFlow.js

let trainData = [], testData = [];
let xTrain, yTrain, xVal, yVal, testX;
let featureNames = [];
let model;

// --- Helpers ---
const $ = id => document.getElementById(id);
const setDisabled = (id, state) => { $(id).disabled = state; };

// --- CSV Preview ---
$('btnPreview').addEventListener('click', async () => {
  try {
    const trainFile = $('trainFile').files[0];
    const testFile = $('testFile').files[0];
    if (!trainFile || !testFile) return alert('Please upload both train.csv and test.csv');
    trainData = await parseCSV(trainFile);
    testData = await parseCSV(testFile);
    $('dataPreview').textContent = JSON.stringify(trainData.slice(0, 5), null, 2);
    $('dataInfo').textContent = `Train rows: ${trainData.length}, Test rows: ${testData.length}`;
    setDisabled('btnPreprocess', false);
  } catch (err) {
    alert('CSV load error: ' + err);
  }
});

function parseCSV(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true, dynamicTyping: true, skipEmptyLines: true,
      complete: res => resolve(res.data), error: err => reject(err)
    });
  });
}

// --- Preprocessing ---
$('btnPreprocess').addEventListener('click', () => {
  const addFamily = $('addFamily').checked;

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
  trainData.forEach(r => { if (!r.Age) r.Age = ageMed; if (!r.Embarked) r.Embarked = embMode; });
  testData.forEach(r => { if (!r.Age) r.Age = ageMed; if (!r.Embarked) r.Embarked = embMode; });

  featureNames = ['Age', 'Fare'];
  if (addFamily) featureNames.push('FamilySize', 'IsAlone');
  ['Pclass', 'Sex', 'Embarked'].forEach(cat => {
    [...new Set(trainData.map(r => r[cat]))].forEach(u => featureNames.push(`${cat}_${u}`));
  });

  const proc = rows => rows.map(r => {
    let feats = [];
    feats.push((r.Age - ageMed) / ageMed);
    feats.push((r.Fare - (r.Fare || 0)) / ((r.Fare || 1)));
    if (addFamily) {
      const fs = r.SibSp + r.Parch + 1;
      feats.push(fs, fs === 1 ? 1 : 0);
    }
    ['Pclass', 'Sex', 'Embarked'].forEach(cat => {
      [...new Set(trainData.map(rr => rr[cat]))].forEach(u => feats.push(r[cat] === u ? 1 : 0));
    });
    return feats;
  });

  const X = proc(trainData), Y = trainData.map(r => r.Survived);
  const N = X.length, split = Math.floor(N * 0.8);
  xTrain = tf.tensor2d(X.slice(0, split));
  yTrain = tf.tensor2d(Y.slice(0, split), [split, 1]);
  xVal = tf.tensor2d(X.slice(split));
  yVal = tf.tensor2d(Y.slice(split), [N - split, 1]);
  testX = tf.tensor2d(proc(testData));

  $('preprocessOutput').textContent =
    `Features: ${featureNames.join(', ')}\nTrain shape: ${xTrain.shape}`;
  setDisabled('btnBuild', false);
});

// --- Build Model ---
$('btnBuild').addEventListener('click', () => {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [featureNames.length] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  model.summary();
  $('modelSummary').textContent = 'Model built with 16 hidden units (ReLU) → 1 sigmoid output.';
  setDisabled('btnTrain', false);
});

// --- Train Model ---
$('btnTrain').addEventListener('click', async () => {
  $('trainStatus').textContent = 'Training...';
  const history = await model.fit(xTrain, yTrain, {
    epochs: 50, batchSize: 32, validationData: [xVal, yVal],
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance', tab: 'Training' },
      ['loss', 'val_loss', 'acc', 'val_acc'], { callbacks: ['onEpochEnd'] })
  });
  $('trainStatus').textContent = 'Training complete.';
  setDisabled('btnPredict', false);
  setDisabled('btnExport', false);
  setDisabled('btnSaveModel', false);
});

// --- Predict ---
$('btnPredict').addEventListener('click', () => {
  const preds = model.predict(testX).arraySync().map(r => r[0]);
  $('predictionOutput').textContent =
    preds.slice(0, 10).map(p => p.toFixed(3)).join(', ') + ' ...';
});

// --- Export ---
$('btnExport').addEventListener('click', () => {
  const preds = model.predict(testX).arraySync().map(r => r[0]);
  let csv = 'PassengerId,Survived\n';
  testData.forEach((r, i) => {
    csv += `${r.PassengerId},${preds[i] >= 0.5 ? 1 : 0}\n`;
  });
  downloadCSV(csv, 'submission.csv');
});

// --- Save Model ---
$('btnSaveModel').addEventListener('click', async () => {
  await model.save('downloads://titanic-tfjs');
});

// --- Download Helper ---
function downloadCSV(csv, name) {
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}
