// app.js - Titanic Binary Classifier in TensorFlow.js

const $ = id => document.getElementById(id);
function setDisabled(id, state) {
  const el = $(id);
  if (el) el.disabled = state;
}

// Globals
let rawTrain = null, rawTest = null;
let trainData = [], testData = [];
let featureNames = [];
let trainXs, trainYs, valXs, valYs, testXs;
let model, stopTrainingFlag = false;
let valPredProbs = null;

// ------------------------------
// CSV Loading
// ------------------------------
function parseFileInput(fileInput) {
  return new Promise((resolve, reject) => {
    const file = fileInput.files[0];
    if (!file) return reject("No file selected");
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: results => resolve(results.data),
      error: err => reject(err)
    });
  });
}

$('trainFile').addEventListener('change', async () => {
  try {
    rawTrain = await parseFileInput($('trainFile'));
    $('previewInfo').textContent = `Loaded train.csv — ${rawTrain.length} rows.`;
    setDisabled('btnPreview', false);
    setDisabled('btnInspect', false);
  } catch (err) {
    alert("Error loading train.csv: " + err);
  }
});

$('testFile').addEventListener('change', async () => {
  try {
    rawTest = await parseFileInput($('testFile'));
    $('previewInfo').textContent += `\nLoaded test.csv — ${rawTest.length} rows.`;
  } catch (err) {
    alert("Error loading test.csv: " + err);
  }
});

// ------------------------------
// Preview
// ------------------------------
$('btnPreview').addEventListener('click', () => {
  if (!rawTrain) return alert("Load train.csv first.");
  $('previewInfo').textContent = JSON.stringify(rawTrain.slice(0, 5), null, 2);
  setDisabled('btnPreprocess', false);
});

// Inspect survival by Sex & Pclass
$('btnInspect').addEventListener('click', () => {
  if (!rawTrain) return;
  const survivalBySex = {};
  rawTrain.forEach(r => {
    if (!r.Survived) return;
    const s = r.Sex;
    if (!survivalBySex[s]) survivalBySex[s] = { survived: 0, total: 0 };
    survivalBySex[s].total++;
    if (r.Survived == 1) survivalBySex[s].survived++;
  });
  $('previewInfo').textContent = "Survival by Sex: " + JSON.stringify(survivalBySex, null, 2);
});

// ------------------------------
// Preprocessing
// ------------------------------
function median(values) {
  const nums = values.filter(v => v != null).sort((a, b) => a - b);
  const mid = Math.floor(nums.length / 2);
  return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
}

function mode(values) {
  const freq = {};
  values.forEach(v => { if (v) freq[v] = (freq[v] || 0) + 1; });
  return Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];
}

function preprocessData(data, isTrain = true) {
  // Impute
  const ageMed = median(data.map(r => r.Age));
  const embMode = mode(data.map(r => r.Embarked));
  data.forEach(r => {
    if (!r.Age) r.Age = ageMed;
    if (!r.Embarked) r.Embarked = embMode;
  });

  // Feature engineering
  data.forEach(r => {
    r.FamilySize = (r.SibSp || 0) + (r.Parch || 0) + 1;
    r.IsAlone = r.FamilySize === 1 ? 1 : 0;
  });

  // One-hot encode
  const sexVals = ["male", "female"];
  const pclassVals = [1, 2, 3];
  const embVals = ["C", "Q", "S"];

  function oneHotEncode(val, categories) {
    return categories.map(c => (val === c ? 1 : 0));
  }

  const features = data.map(r => {
    const sexOH = oneHotEncode(r.Sex, sexVals);
    const pclassOH = oneHotEncode(r.Pclass, pclassVals);
    const embOH = oneHotEncode(r.Embarked, embVals);
    return [
      (r.Age - 30) / 15, // scaled
      (r.Fare - 32) / 50,
      r.SibSp,
      r.Parch,
      r.FamilySize,
      r.IsAlone,
      ...sexOH,
      ...pclassOH,
      ...embOH
    ];
  });

  featureNames = [
    "Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone",
    ...sexVals.map(s => "Sex_" + s),
    ...pclassVals.map(p => "Pclass_" + p),
    ...embVals.map(e => "Embarked_" + e)
  ];

  return tf.tensor2d(features);
}

$('btnPreprocess').addEventListener('click', () => {
  if (!rawTrain) return;
  trainData = rawTrain.filter(r => r.Survived != null);
  const labels = trainData.map(r => r.Survived);

  const xs = preprocessData(trainData);
  const ys = tf.tensor2d(labels, [labels.length, 1]);

  // Split 80/20
  const split = Math.floor(xs.shape[0] * 0.8);
  trainXs = xs.slice([0, 0], [split, xs.shape[1]]);
  trainYs = ys.slice([0, 0], [split, 1]);
  valXs = xs.slice([split, 0], [xs.shape[0] - split, xs.shape[1]]);
  valYs = ys.slice([split, 0], [ys.shape[0] - split, 1]);

  if (rawTest) testXs = preprocessData(rawTest, false);

  $('featureInfo').textContent = "Features: " + featureNames.join(", ") +
    `\nTrain shape: ${trainXs.shape}, Val shape: ${valXs.shape}`;
  setDisabled('btnBuild', false);
  setDisabled('btnShowFeatures', false);
});

$('btnShowFeatures').addEventListener('click', () => {
  $('featureInfo').textContent = "Final feature names:\n" + featureNames.join(", ");
});

// ------------------------------
// Model
// ------------------------------
$('btnBuild').addEventListener('click', () => {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [featureNames.length] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  $('modelSummary').textContent = "Model built. Hidden:16 relu, Output:sigmoid";
  setDisabled('btnSummary', false);
  setDisabled('btnTrain', false);
});

$('btnSummary').addEventListener('click', () => {
  model.summary();
});

// ------------------------------
// Training
// ------------------------------
$('btnTrain').addEventListener('click', async () => {
  if (!model) return;
  stopTrainingFlag = false;
  $('trainLog').textContent = "";

  const history = await model.fit(trainXs, trainYs, {
    epochs: 50,
    batchSize: 32,
    validationData: [valXs, valYs],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        $('trainLog').textContent += `\nEpoch ${epoch}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc?.toFixed(4)}`;
        if (stopTrainingFlag) return false;
      }
    }
  });

  valPredProbs = model.predict(valXs).arraySync().map(p => p[0]);
  setDisabled('btnEval', false);
});

$('btnStop').addEventListener('click', () => stopTrainingFlag = true);

// ------------------------------
// Metrics
// ------------------------------
function confusionMatrix(yTrue, yProb, thresh = 0.5) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  yTrue.forEach((y, i) => {
    const pred = yProb[i] >= thresh ? 1 : 0;
    if (y === 1 && pred === 1) tp++;
    else if (y === 0 && pred === 0) tn++;
    else if (y === 0 && pred === 1) fp++;
    else if (y === 1 && pred === 0) fn++;
  });
  return { tp, tn, fp, fn };
}

$('thresholdSlider').addEventListener('input', e => {
  $('thresholdVal').textContent = e.target.value;
});

$('btnEval').addEventListener('click', () => {
  if (!valPredProbs) return;
  const yTrue = valYs.arraySync().map(r => r[0]);
  const thresh = parseFloat($('thresholdSlider').value);
  const cm = confusionMatrix(yTrue, valPredProbs, thresh);
  const precision = cm.tp / (cm.tp + cm.fp);
  const recall = cm.tp / (cm.tp + cm.fn);
  const f1 = 2 * precision * recall / (precision + recall);

  $('metricInfo').textContent =
    `Confusion Matrix @${thresh}:\nTP:${cm.tp}, TN:${cm.tn}, FP:${cm.fp}, FN:${cm.fn}\n` +
    `Precision:${precision.toFixed(3)}, Recall:${recall.toFixed(3)}, F1:${f1.toFixed(3)}`;

  setDisabled('btnPredict', false);
});

// ------------------------------
// Predict & Export
// ------------------------------
$('btnPredict').addEventListener('click', () => {
  if (!testXs) return alert("No test.csv loaded");
  const probs = model.predict(testXs).arraySync().map(p => p[0]);
  const thresh = parseFloat($('thresholdSlider').value);
  const preds = probs.map(p => (p >= thresh ? 1 : 0));
  rawTest.forEach((r, i) => r.Survived = preds[i]);

  $('predictionInfo').textContent = "Predictions ready. Example:\n" + JSON.stringify(rawTest.slice(0, 5), null, 2);
  setDisabled('btnExport', false);
});

$('btnExport').addEventListener('click', () => {
  const rows = [["PassengerId", "Survived"], ...rawTest.map(r => [r.PassengerId, r.Survived])];
  const csv = rows.map(r => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: 'text/csv' });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "submission.csv";
  link.click();
  setDisabled('btnSaveModel', false);
});

$('btnSaveModel').addEventListener('click', async () => {
  await model.save('downloads://titanic-tfjs');
  alert("Model saved to downloads");
});
