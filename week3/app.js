// =============================
// TITANIC SURVIVAL CLASSIFIER
// Enhanced Visualization + Inspection
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

  createInspectionVisualizations(trainData);
};

// =============================
// VISUALIZATIONS (INSPECTION)
// =============================
function createInspectionVisualizations(data) {
  tfvis.visor().open();
  tfvis.visor().setActiveTab("Data Insights");

  // --- Survival by Sex ---
  const sexGroups = {};
  data.forEach((row) => {
    if (!row.Sex) return;
    if (!sexGroups[row.Sex]) sexGroups[row.Sex] = { survived: 0, total: 0 };
    sexGroups[row.Sex].total++;
    if (row.Survived === 1) sexGroups[row.Sex].survived++;
  });
  const sexValues = Object.entries(sexGroups).map(([sex, s]) => ({
    x: sex,
    y: (s.survived / s.total) * 100,
  }));
  tfvis.render.barchart(
    { name: "Survival Rate by Sex", tab: "Data Insights" },
    sexValues,
    {
      height: 300,
      xLabel: "Sex",
      yLabel: "Survival Rate (%)",
      color: ["#06b6d4"],
      backgroundColor: "#0f172a",
      fontSize: 14,
    }
  );

  // --- Survival by Pclass ---
  const classGroups = {};
  data.forEach((row) => {
    if (!row.Pclass) return;
    if (!classGroups[row.Pclass])
      classGroups[row.Pclass] = { survived: 0, total: 0 };
    classGroups[row.Pclass].total++;
    if (row.Survived === 1) classGroups[row.Pclass].survived++;
  });
  const pclassValues = Object.entries(classGroups).map(([cls, s]) => ({
    x: "Class " + cls,
    y: (s.survived / s.total) * 100,
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
      fontSize: 14,
    }
  );

  // --- Age distribution (histogram by Survived) ---
  const survivedAges = data.filter(d => d.Survived === 1 && d.Age).map(d => d.Age);
  const notSurvivedAges = data.filter(d => d.Survived === 0 && d.Age).map(d => d.Age);
  tfvis.render.histogram(
    { name: "Age Distribution (Survived vs Not)", tab: "Data Insights" },
    [survivedAges, notSurvivedAges],
    {
      height: 300,
      width: 400,
      xLabel: "Age",
      yLabel: "Count",
      fontSize: 14,
      seriesColors: ["#22c55e", "#ef4444"],
      backgroundColor: "#111827",
    }
  );

  // --- Fare distribution ---
  const survivedFare = data.filter(d => d.Survived === 1 && d.Fare).map(d => d.Fare);
  const notSurvivedFare = data.filter(d => d.Survived === 0 && d.Fare).map(d => d.Fare);
  tfvis.render.histogram(
    { name: "Fare Distribution (Survived vs Not)", tab: "Data Insights" },
    [survivedFare, notSurvivedFare],
    {
      height: 300,
      width: 400,
      xLabel: "Fare",
      yLabel: "Count",
      fontSize: 14,
      seriesColors: ["#22c55e", "#ef4444"],
      backgroundColor: "#111827",
    }
  );

  // --- Passengers by Embarked ---
  const embarkedCounts = {};
  data.forEach((row) => {
    if (!row.Embarked) return;
    if (!embarkedCounts[row.Embarked]) embarkedCounts[row.Embarked] = 0;
    embarkedCounts[row.Embarked]++;
  });
  const embarkedValues = Object.entries(embarkedCounts).map(([port, count]) => ({
    x: port,
    y: count,
  }));
  tfvis.render.barchart(
    { name: "Passengers by Embarked Port", tab: "Data Insights" },
    embarkedValues,
    {
      height: 300,
      xLabel: "Port",
      yLabel: "Passenger Count",
      color: ["#f59e0b"],
      backgroundColor: "#0f172a",
      fontSize: 14,
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
// MODEL, TRAINING, METRICS, EXPORT
// =============================
// (Same as previous message — unchanged modern version)
