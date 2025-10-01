// ==================== GLOBALS ====================
let trainData = null, testData = null;
let preprocessedTrainData = null, preprocessedTestData = null;
let model = null, trainingHistory = null;
let validationData = null, validationLabels = null, validationPredictions = null, testPredictions = null;
let visorOpened = false;

const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// ==================== FILE LOADING ====================
async function loadData() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];
  if (!trainFile || !testFile) {
    alert('Please upload both train.csv and test.csv');
    return;
  }
  const statusDiv = document.getElementById('data-status');
  statusDiv.innerHTML = 'Loading data...';
  try {
    const trainText = await readFile(trainFile);
    trainData = parseCSV(trainText);
    const testText = await readFile(testFile);
    testData = parseCSV(testText);
    statusDiv.innerHTML = `✅ Data loaded: train=${trainData.length}, test=${testData.length}`;
    document.getElementById('inspect-btn').disabled = false;
  } catch (e) {
    statusDiv.innerHTML = '❌ Error: ' + e.message;
  }
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

function parseCSV(text) {
  const results = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
  return results.data;
}

// ==================== INSPECTION ====================
function inspectData() {
  if (!trainData) return alert('Load data first');

  const previewDiv = document.getElementById('data-preview');
  previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
  previewDiv.appendChild(createTable(trainData.slice(0, 10)));

  createVisualizations();
  document.getElementById('preprocess-btn').disabled = false;
}

function createTable(data) {
  const table = document.createElement('table');
  const header = document.createElement('tr');
  Object.keys(data[0]).forEach(k => {
    const th = document.createElement('th'); th.textContent = k; header.appendChild(th);
  });
  table.appendChild(header);
  data.forEach(row => {
    const tr = document.createElement('tr');
    Object.values(row).forEach(val => {
      const td = document.createElement('td');
      td.textContent = val ?? 'NULL';
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  return table;
}

// ==================== VISUALIZATION ====================
function createVisualizations() {
  if (!trainData) return;

  // Group survival by sex
  const bySex = {};
  trainData.forEach(r => {
    if (r.Sex && r.Survived !== undefined) {
      if (!bySex[r.Sex]) bySex[r.Sex] = { survived: 0, total: 0 };
      bySex[r.Sex].total++;
      if (r.Survived === 1) bySex[r.Sex].survived++;
    }
  });
  const sexData = Object.entries(bySex).map(([sex, d]) => ({ x: sex, y: (d.survived / d.total) * 100 }));

  tfvis.render.barchart(
    { name: 'Survival Rate by Sex', tab: 'Charts' },
    sexData,
    { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
  );

  // Group survival by class
  const byClass = {};
  trainData.forEach(r => {
    if (r.Pclass && r.Survived !== undefined) {
      if (!byClass[r.Pclass]) byClass[r.Pclass] = { survived: 0, total: 0 };
      byClass[r.Pclass].total++;
      if (r.Survived === 1) byClass[r.Pclass].survived++;
    }
  });
  const classData = Object.entries(byClass).map(([pc, d]) => ({ x: `Class ${pc}`, y: (d.survived / d.total) * 100 }));

  tfvis.render.barchart(
    { name: 'Survival Rate by Class', tab: 'Charts' },
    classData,
    { xLabel: 'Class', yLabel: 'Survival Rate (%)' }
  );

  if (!visorOpened) { tfvis.visor().open(); visorOpened = true; }
}

// ==================== PREPROCESS ====================
function preprocessData() {
  const out = document.getElementById('preprocessing-output');
  try {
    const ageMedian = calculateMedian(trainData.map(r => r.Age).filter(v => v != null));
    const fareMedian = calculateMedian(trainData.map(r => r.Fare).filter(v => v != null));
    const embarkedMode = calculateMode(trainData.map(r => r.Embarked).filter(v => v != null));

    preprocessedTrainData = { features: [], labels: [] };
    trainData.forEach(r => {
      preprocessedTrainData.features.push(extractFeatures(r, ageMedian, fareMedian, embarkedMode));
      preprocessedTrainData.labels.push(Number(r.Survived));
    });
    preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
    preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);

    preprocessedTestData = { features: [], ids: [] };
    testData.forEach(r => {
      preprocessedTestData.features.push(extractFeatures(r, ageMedian, fareMedian, embarkedMode));
      preprocessedTestData.ids.push(r.PassengerId);
    });

    out.innerHTML = `✅ Preprocessing done<br/>
      Train: ${preprocessedTrainData.features.shape}<br/>
      Labels: ${preprocessedTrainData.labels.shape}<br/>
      Test: ${preprocessedTestData.features.length}`;
    document.getElementById('create-model-btn').disabled = false;
  } catch (e) {
    out.innerHTML = `❌ Error during preprocessing: ${e.message}`;
  }
}

// ==================== HELPERS ====================
function calculateMedian(arr) {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a,b)=>a-b);
  const mid = Math.floor(sorted.length/2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2;
}
function calculateMode(arr) {
  if (!arr.length) return null;
  const freq = {};
  arr.forEach(v => freq[v] = (freq[v]||0)+1);
  return Object.entries(freq).sort((a,b)=>b[1]-a[1])[0][0];
}
function calculateStdDev(arr) {
  if (!arr.length) return 1;
  const mean = arr.reduce((s,v)=>s+v,0)/arr.length;
  const variance = arr.reduce((s,v)=>s+Math.pow(v-mean,2),0)/arr.length;
  return Math.sqrt(variance);
}
function oneHotEncode(val, categories) {
  return categories.map(c => c===val?1:0);
}

function extractFeatures(r, ageMedian, fareMedian, embarkedMode) {
  const age = r.Age ?? ageMedian;
  const fare = r.Fare ?? fareMedian;
  const embarked = r.Embarked ?? embarkedMode;

  const stdAge = (age - ageMedian) / (calculateStdDev(trainData.map(x=>x.Age).filter(v=>v!=null))||1);
  const stdFare = (fare - fareMedian) / (calculateStdDev(trainData.map(x=>x.Fare).filter(v=>v!=null))||1);

  let feats = [stdAge, stdFare, r.SibSp||0, r.Parch||0];
  feats = feats.concat(oneHotEncode(r.Pclass, [1,2,3]));
  feats = feats.concat(oneHotEncode(r.Sex, ['male','female']));
  feats = feats.concat(oneHotEncode(embarked, ['C','Q','S']));

  if (document.getElementById('add-family-features')?.checked) {
    const family = (r.SibSp||0) + (r.Parch||0) + 1;
    feats.push(family, family===1?1:0);
  }
  return feats;
}
