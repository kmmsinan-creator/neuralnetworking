// =============================
// TITANIC SURVIVAL CLASSIFIER
// Enhanced Visualization Version
// =============================

// Global state
let trainData = [];
let testData = [];
let preprocessedTrain = null;
let preprocessedTest = null;
let model = null;
let validationData = null;
let validationLabels = null;
let predictions = null;

// CONSTANTS
const TARGET_FEATURE = "Survived";
const ID_FEATURE = "PassengerId";
const NUMERICAL = ["Age", "Fare", "SibSp", "Parch"];
const CATEGORICAL = ["Pclass", "Sex", "Embarked"];

// Utility - File read
const readCSV = (file) =>
  new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => resolve(results.data),
      error: reject,
    });
  });

// =============================
// STEP 1: LOAD DATA
// =============================
document.getElementById("load-data-btn").onclick = async () => {
  const trainFile = document.getElementById("trainFile").files[0];
  const testFile = document.getElementById("testFile").files[0];
  const status = document.getElementById("data-status");

  if (!trainFile || !testFile) {
    alert("Please upload both train.csv and test.csv!");
    return;
  }

  status.textContent = "📥 Loading data...";
  trainData = await readCSV(trainFile);
  testData = await readCSV(testFile);

  status.textContent = `✅ Loaded: ${trainData.length} training rows, ${testData.length} test rows`;
  document.getElementById("inspect-btn").disabled = false;
};

// =============================
// STEP 2: INSPECT DATA
// =============================
document.getElementById("inspect-btn").onclick = () => {
  const previewDiv = document.getElementById("data-preview");
  const statsDiv = document.getElementById("data-stats");
  previewDiv.innerHTML = JSON.stringify(trainData.slice(0, 5), null, 2);

  const missing = {};
  Object.keys(trainData[0]).forEach((key) => {
    missing[key] =
      trainData.filter((r) => r[key] === null || r[key] === "").length;
  });

  statsDiv.innerHTML =
    "🔍 Missing values per column:\n" + JSON.stringify(missing, null, 2);

  document.getElementById("preprocess-btn").disabled = false;

  createVisualizations(trainData);
};

// =============================
// VISUALIZATIONS
// =============================
function createVisualizations(data) {
  // Survival by Sex
  const groupedSex = {};
  data.forEach((row) => {
    if (!row.Sex) return;
    if (!groupedSex[row.Sex]) groupedSex[row.Sex] = { survived: 0, total: 0 };
    groupedSex[row.Sex].total++;
    if (row.Survived === 1) groupedSex[row.Sex].survived++;
  });

  const sexValues = Object.entries(groupedSex).map(([sex, stats]) => ({
    x: sex,
    y: (stats.survived / stats.total) * 100,
  }));

  tfvis.render.barchart(
    { name: "Survival Rate by Sex", tab: "Data Insights" },
    sexValues,
    {
      height: 300,
      xLabel: "Sex",
      yLabel: "Survival Rate (%)",
      fontSize: 14,
      color: ["#06b6d4"],
      backgroundColor: "#0f172a",
    }
  );

  // Survival by Passenger Class
  const groupedClass = {};
  data.forEach((row) => {
    if (!row.Pclass) return;
    if (!groupedClass[row.Pclass])
      groupedClass[row.Pclass] = { survived: 0, total: 0 };
    groupedClass[row.Pclass].total++;
    if (row.Survived === 1) groupedClass[row.Pclass].survived++;
  });

  const pclassValues = Object.entries(groupedClass).map(([cls, stats]) => ({
    x: "Class " + cls,
    y: (stats.survived / stats.total) * 100,
  }));

  tfvis.render.barchart(
    { name: "Survival Rate by Class", tab: "Data Insights" },
    pclassValues,
    {
      height: 300,
      xLabel: "Class",
      yLabel: "Survival Rate (%)",
      color: ["#3b82f6"],
      backgroundColor: "#0f172a",
    }
  );
}

// =============================
// STEP 3: PREPROCESS DATA
// =============================
document.getElementById("preprocess-btn").onclick = () => {
  const addFamily = document.getElementById("toggle-family").checked;
  const output = document.getElementById("preprocessing-output");
  output.textContent = "⚙️ Preprocessing data...";

  const ageMedian = median(trainData.map((r) => r.Age).filter((v) => v));
  const fareMedian = median(trainData.map((r) => r.Fare).filter((v) => v));
  const embarkedMode = mode(trainData.map((r) => r.Embarked).filter((v) => v));

  const extractFeatures = (row) => {
    const age = row.Age ?? ageMedian;
    const fare = row.Fare ?? fareMedian;
    const embarked = row.Embarked ?? embarkedMode;
    let features = [age, fare, row.SibSp ?? 0, row.Parch ?? 0];

    // One-hot encode
    features = features.concat(oneHot(row.Pclass, [1, 2, 3]));
    features = features.concat(oneHot(row.Sex, ["male", "female"]));
    features = features.concat(oneHot(embarked, ["C", "Q", "S"]));

    if (addFamily) {
      const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
      features.push(familySize, familySize === 1 ? 1 : 0);
    }

    return features;
  };

  preprocessedTrain = {
    features: tf.tensor2d(trainData.map(extractFeatures)),
    labels: tf.tensor1d(trainData.map((r) => r[TARGET_FEATURE])),
  };
  preprocessedTest = {
    features: testData.map(extractFeatures),
    ids: testData.map((r) => r[ID_FEATURE]),
  };

  output.textContent = `✅ Preprocessed! Features: ${preprocessedTrain.features.shape[1]} columns`;
  document.getElementById("create-model-btn").disabled = false;
};

// =============================
// STEP 4: MODEL CREATION
// =============================
document.getElementById("create-model-btn").onclick = () => {
  const summaryDiv = document.getElementById("model-summary");

  const inputShape = preprocessedTrain.features.shape[1];
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [inputShape], units: 16, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 8, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

  summaryDiv.textContent = `🧠 Model created with ${model.countParams()} parameters.`;
  document.getElementById("train-btn").disabled = false;
};

// =============================
// STEP 5: TRAIN MODEL
// =============================
document.getElementById("train-btn").onclick = async () => {
  const output = document.getElementById("train-output");
  output.textContent = "🏋️ Training...";

  const split = Math.floor(preprocessedTrain.features.shape[0] * 0.8);
  const trainX = preprocessedTrain.features.slice(0, split);
  const valX = preprocessedTrain.features.slice(split);
  const trainY = preprocessedTrain.labels.slice(0, split);
  const valY = preprocessedTrain.labels.slice(split);
  validationData = valX;
  validationLabels = valY;

  await model.fit(trainX, trainY, {
    epochs: 40,
    batchSize: 32,
    validationData: [valX, valY],
    callbacks: tfvis.show.fitCallbacks({ name: "Training Progress", tab: "Training" },
      ["loss", "val_loss", "acc", "val_acc"],
      { height: 300, callbacks: ["onEpochEnd"] })
  });

  output.textContent = "✅ Training complete!";
  predictions = model.predict(valX);
  document.getElementById("threshold-slider").disabled = false;
  document.getElementById("predict-btn").disabled = false;
  document.getElementById("save-model-btn").disabled = false;

  updateMetrics();
};

// =============================
// STEP 6: METRICS
// =============================
document.getElementById("threshold-slider").oninput = updateMetrics;

async function updateMetrics() {
  if (!predictions) return;

  const threshold = parseFloat(document.getElementById("threshold-slider").value);
  document.getElementById("threshold-value").textContent = threshold.toFixed(2);

  const yTrue = validationLabels.arraySync();
  const yPred = predictions.arraySync().map((p) => (p[0] >= threshold ? 1 : 0));

  let tp = 0, fp = 0, tn = 0, fn = 0;
  yTrue.forEach((val, i) => {
    if (val === 1 && yPred[i] === 1) tp++;
    else if (val === 0 && yPred[i] === 1) fp++;
    else if (val === 0 && yPred[i] === 0) tn++;
    else fn++;
  });

  const accuracy = (tp + tn) / yTrue.length;
  const precision = tp / (tp + fp);
  const recall = tp / (tp + fn);
  const f1 = (2 * precision * recall) / (precision + recall);

  const metricsDiv = document.getElementById("metrics-output");
  metricsDiv.innerHTML = `
    <div class="metric-card">Accuracy: ${(accuracy * 100).toFixed(2)}%</div>
    <div class="metric-card">Precision: ${precision.toFixed(3)}</div>
    <div class="metric-card">Recall: ${recall.toFixed(3)}</div>
    <div class="metric-card">F1: ${f1.toFixed(3)}</div>
  `;

  // Confusion Matrix
  const cmDiv = document.getElementById("confusion-matrix");
  cmDiv.innerHTML = `
    <table>
      <tr><th></th><th>Pred 1</th><th>Pred 0</th></tr>
      <tr><th>Actual 1</th><td>${tp}</td><td>${fn}</td></tr>
      <tr><th>Actual 0</th><td>${fp}</td><td>${tn}</td></tr>
    </table>
  `;

  // ROC Curve
  const roc = [];
  for (let t = 0; t <= 1; t += 0.02) {
    let TP = 0, FP = 0, TN = 0, FN = 0;
    yTrue.forEach((v, i) => {
      const pred = predictions.arraySync()[i][0] >= t ? 1 : 0;
      if (v === 1 && pred === 1) TP++;
      else if (v === 1 && pred === 0) FN++;
      else if (v === 0 && pred === 1) FP++;
      else TN++;
    });
    const TPR = TP / (TP + FN);
    const FPR = FP / (FP + TN);
    roc.push({ x: FPR, y: TPR });
  }

  tfvis.render.scatterplot(
    { name: "ROC Curve", tab: "Metrics" },
    { values: roc },
    {
      height: 300,
      width: 350,
      xLabel: "False Positive Rate",
      yLabel: "True Positive Rate",
      backgroundColor: "#111827",
      pointColor: "#38bdf8",
    }
  );
}

// =============================
// STEP 7: PREDICT + EXPORT
// =============================
document.getElementById("predict-btn").onclick = async () => {
  const out = document.getElementById("prediction-output");
  out.textContent = "🚀 Predicting...";
  const Xtest = tf.tensor2d(preprocessedTest.features);
  const pred = model.predict(Xtest).arraySync();
  let csv = "PassengerId,Survived\n";
  preprocessedTest.ids.forEach((id, i) => {
    csv += `${id},${pred[i][0] >= 0.5 ? 1 : 0}\n`;
  });
  out.textContent = "✅ Prediction complete. Downloading submission.csv...";
  downloadFile(csv, "submission.csv");
  document.getElementById("export-btn").disabled = false;
};

// =============================
// STEP 8: SAVE MODEL
// =============================
document.getElementById("save-model-btn").onclick = async () => {
  await model.save("downloads://titanic_tfjs_model");
};

// =============================
// HELPER FUNCTIONS
// =============================
const median = (arr) => {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};
const mode = (arr) =>
  arr
    .sort(
      (a, b) =>
        arr.filter((v) => v === a).length -
        arr.filter((v) => v === b).length
    )
    .pop();
const oneHot = (val, cats) => cats.map((c) => (val === c ? 1 : 0));
const downloadFile = (text, name) => {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  link.download = name;
  link.click();
};
