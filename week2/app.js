/**
 * app.js
 * In-browser Titanic classifier using TensorFlow.js and tfjs-vis
 *
 * Workflow:
 * - Load train.csv / test.csv (file inputs)
 * - Preview and inspect (missing %, survival by Sex/Pclass)
 * - Preprocess (median Age, mode Embarked, standardize Age/Fare, one-hot encode)
 * - Build model: Dense(16, relu) -> Dense(1, sigmoid) (adam, binaryCrossentropy)
 * - Train with 80/20 stratified split, 50 epochs, batch 32, early stopping (patience 5)
 * - Evaluate: ROC/AUC, slider for threshold -> confusion matrix, precision/recall/F1
 * - Predict test set and export submission.csv & probabilities.csv; download model
 *
 * NOTE: Schema points are at the top; swap them if reusing for another dataset.
 */

// -------------------- Configuration / Schema -------------------- //
// Target & feature schema (swap when reusing for another dataset)
const TARGET_COL = 'Survived';
const ID_COL = 'PassengerId';
const FEATURE_COLS = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];
// ---------------------------------------------------------------- //

/* Global state */
let rawTrain = null;
let rawTest = null;
let processed = null; // {trainRows, valRows, testRows, featureNames, scalers, categories}
let model = null;
let stopRequested = false;
let valPredProbs = null;
let valLabels = null;
let testPredProbs = null;

/* DOM */
const $ = id => document.getElementById(id);
const log = (...s) => { console.log(...s); };

/* Helper: Enable/disable controls safely */
function setDisabled(id, v) { const el = $(id); if (el) el.disabled = v; }

/* Parse CSV from file input using PapaParse */
function parseFileInput(fileInput) {
  return new Promise((resolve, reject) => {
    const f = fileInput.files[0];
    if (!f) return reject(new Error('No file selected'));
    Papa.parse(f, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data),
      error: (err) => reject(err)
    });
  });
}

/* UI hookup: file inputs */
$('trainFile').addEventListener('change', async () => {
  try {
    rawTrain = await parseFileInput($('trainFile'));
    $('previewInfo').textContent = `Loaded train.csv — ${rawTrain.length} rows.`;
    setDisabled('btnPreview', false);
    setDisabled('btnInspect', false);
    setDisabled('btnPreprocess', true);
    setDisabled('btnBuild', true);
    setDisabled('btnTrain', true);
    setDisabled('btnEval', true);
    setDisabled('btnPredict', true);
    setDisabled('btnExport', true);
    setDisabled('btnSaveModel', true);
    // clear panels
    $('previewTable').innerHTML = '';
    $('inspectStats').innerHTML = '';
    $('predPreview').innerHTML = '';
  } catch (err) {
    alert('Error loading train.csv: ' + err.message);
  }
});

$('testFile').addEventListener('change', async () => {
  try {
    rawTest = await parseFileInput($('testFile'));
    $('previewInfo').textContent = (rawTrain ? `Train loaded (${rawTrain.length})` : '') + ` Test loaded — ${rawTest.length} rows.`;
  } catch (err) {
    alert('Error loading test.csv: ' + err.message);
  }
});

/* Preview & Inspect actions */
$('btnPreview').addEventListener('click', () => {
  if (!rawTrain) { alert('Upload train.csv first'); return; }
  renderPreview(rawTrain, $('previewTable'), 8);
  setDisabled('btnPreprocess', false);
  // charts for survival by Sex / Pclass
  renderSurvivalCharts(rawTrain);
});

/* Detailed inspect: shape, missing% */
$('btnInspect').addEventListener('click', () => {
  if (!rawTrain) { alert('Upload train.csv first'); return; }
  const stats = computeMissingAndShape(rawTrain);
  $('inspectStats').innerHTML = `<pre>${stats}</pre>`;
});

/* Preprocess */
$('btnPreprocess').addEventListener('click', () => {
  if (!rawTrain) { alert('No train.csv'); return; }
  try {
    processed = preprocess(rawTrain, rawTest, $('optFamily').checked);
    $('previewInfo').textContent = `Preprocessed. Features: ${processed.featureNames.join(', ')}. Train rows: ${processed.trainRows.length}, Val rows: ${processed.valRows.length}, Test rows: ${processed.testRows ? processed.testRows.length : 0}`;
    setDisabled('btnShowFeatures', false);
    setDisabled('btnBuild', false);
    $('predPreview').innerHTML = '';
  } catch (err) {
    alert('Preprocess error: ' + err.message);
    console.error(err);
  }
});

/* Show feature names and shapes */
$('btnShowFeatures').addEventListener('click', () => {
  if (!processed) { alert('Preprocess first'); return; }
  alert('Features: ' + processed.featureNames.join(', ') + '\nTrain tensor shape: (' + processed.trainRows.length + ',' + processed.featureNames.length + ')');
});

/* Build model */
$('btnBuild').addEventListener('click', () => {
  if (!processed) { alert('Preprocess first'); return; }
  model = buildModel(processed.featureNames.length);
  setDisabled('btnSummary', false);
  setDisabled('btnTrain', false);
  log('Model built');
});

/* Model summary */
$('btnSummary').addEventListener('click', () => {
  if (!model) { alert('Build model first'); return; }
  // model.summary prints to console; show in a small window
  const lines = [];
  model.summary(80, 2000, (line) => lines.push(line));
  alert(lines.join('\n'));
});

/* Train model with early stopping */
$('btnTrain').addEventListener('click', async () => {
  if (!model) { alert('Build model first'); return; }
  if (!processed) { alert('Preprocess first'); return; }
  setDisabled('btnTrain', true);
  setDisabled('btnStop', false);
  stopRequested = false;

  const epochs = parseInt($('epochs').value) || 50;
  const batchSize = parseInt($('batchSize').value) || 32;

  // Prepare tensors
  const {trainXs, trainYs, valXs, valYs} = toTensors(processed.trainRows, processed.valRows, processed.featureNames);

  // Early stopping implementation (patience on val_loss)
  let bestValLoss = Number.POSITIVE_INFINITY;
  let wait = 0;
  const patience = 5;

  const fitCallbacks = tfvis.show.fitCallbacks(
    { name: 'Training: loss & acc' },
    ['loss', 'acc', 'val_loss', 'val_acc'],
    { callbacks: ['onEpochEnd'] }
  );

  await model.fit(trainXs, trainYs, {
    epochs,
    batchSize,
    validationData: [valXs, valYs],
    shuffle: true,
    callbacks: {
      onEpochBegin: async (epoch) => {
        // noop
      },
      onEpochEnd: async (epoch, logs) => {
        await fitCallbacks.onEpochEnd(epoch, logs);
        // early stopping logic
        const valLoss = logs.val_loss;
        if (valLoss < bestValLoss - 1e-6) {
          bestValLoss = valLoss;
          wait = 0;
        } else {
          wait++;
          if (wait >= patience) {
            log('Early stopping at epoch', epoch + 1);
            model.stopTraining = true;
          }
        }
        if (stopRequested) {
          model.stopTraining = true;
          stopRequested = false;
        }
      },
      onTrainEnd: async () => {
        // compute val predictions for ROC/audit
        const valProbsTensor = model.predict(valXs);
        valPredProbs = Array.from(await valProbsTensor.data());
        valLabels = Array.from(await valYs.data());
        valProbsTensor.dispose();

        trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();

        setDisabled('btnEval', false);
        setDisabled('btnStop', true);
        setDisabled('btnPredict', false);
        setDisabled('btnSaveModel', false);
        setDisabled('btnExport', false);
        alert('Training finished.');
      }
    }
  });
});

/* Stop requested */
$('btnStop').addEventListener('click', () => {
  stopRequested = true;
  $('btnStop').disabled = true;
});

/* Evaluate ROC/AUC and show confusion matrix with slider */
$('btnEval').addEventListener('click', () => {
  if (!valPredProbs || !valLabels) { alert('Train the model first (to get validation predictions)'); return; }
  const roc = computeROC(valLabels, valPredProbs);
  tfvis.render.scatterplot(
    { name: 'ROC Curve (Val)', tab: 'Evaluation' },
    { values: roc.points.map(p => ({ x: p.fpr, y: p.tpr })) },
    { xLabel: 'FPR', yLabel: 'TPR', height: 300 }
  );
  $('auc').textContent = `AUC: ${roc.auc.toFixed(4)}`;
  // enable slider
  $('threshold').disabled = false;
  $('threshold').value = 0.5;
  $('thVal').textContent = '0.50';
  updateConfusionMatrix(valLabels, valPredProbs, 0.5);
});

/* Threshold slider change */
$('threshold').addEventListener('input', (e) => {
  const t = parseFloat(e.target.value);
  $('thVal').textContent = t.toFixed(2);
  if (valPredProbs && valLabels) {
    updateConfusionMatrix(valLabels, valPredProbs, t);
  }
});

/* Predict on test and show preview; also create final arrays for export */
$('btnPredict').addEventListener('click', async () => {
  if (!processed || !model) { alert('Preprocess & build model first'); return; }
  if (!processed.testRows || processed.testRows.length === 0) { alert('No test.csv uploaded'); return; }

  // Convert test feature matrix to tensor
  const testXs = tf.tensor2d(processed.testRows.map(r => processed.featureNames.map(fn => r[fn])));
  const probsTensor = model.predict(testXs);
  testPredProbs = Array.from(await probsTensor.data());
  probsTensor.dispose();
  testXs.dispose();

  // Preview first 8
  const rows = processed.testRows.slice(0, 8).map((r, i) => {
    return { PassengerId: r[ID_COL], Prob: testPredProbs[i].toFixed(4), Pred: testPredProbs[i] >= parseFloat($('threshold').value) ? 1 : 0 };
  });
  $('predPreview').innerHTML = `<pre>${JSON.stringify(rows, null, 2)}</pre>`;
  setDisabled('btnExport', false);
});

/* Export submission.csv, probabilities.csv and allow model download */
$('btnExport').addEventListener('click', () => {
  if (!testPredProbs || !processed || !processed.testRows) { alert('No predictions available. Run Predict Test first.'); return; }
  const threshold = parseFloat($('threshold').value);

  // submission.csv (PassengerId,Survived)
  let csv = 'PassengerId,Survived\n';
  processed.testRows.forEach((r, idx) => {
    const pid = r[ID_COL];
    const pred = testPredProbs[idx] >= threshold ? 1 : 0;
    csv += `${pid},${pred}\n`;
  });
  downloadString(csv, 'text/csv', 'submission.csv');

  // probabilities.csv (PassengerId,Probability)
  let pcsv = 'PassengerId,Prob\n';
  processed.testRows.forEach((r, idx) => {
    pcsv += `${r[ID_COL]},${testPredProbs[idx].toFixed(6)}\n`;
  });
  downloadString(pcsv, 'text/csv', 'probabilities.csv');

  // save model for download
  model.save('downloads://titanic-tfjs-model').then(() => {
    alert('Exported submission.csv, probabilities.csv and triggered model download.');
  }).catch(err => {
    console.warn('Model save/download error:', err);
    alert('CSV files downloaded. Model download failed: ' + err.message);
  });
});

/* Save model button (separate) */
$('btnSaveModel').addEventListener('click', async () => {
  if (!model) { alert('No model to save'); return; }
  try {
    await model.save('downloads://titanic-tfjs-model');
  } catch (err) {
    alert('Model save error: ' + err.message);
  }
});

/* ---------- Implementation helpers ---------- */

/* Render preview table (first N rows) */
function renderPreview(data, container, n=8) {
  const rows = data.slice(0,n);
  if (rows.length === 0) { container.innerHTML = '<div class="small muted">No rows</div>'; return; }
  const cols = Object.keys(rows[0]);
  let html = '<table><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
  rows.forEach(r => {
    html += '<tr>' + cols.map(c => `<td>${escapeHtml(String(r[c] === undefined ? '' : r[c]))}</td>`).join('') + '</tr>';
  });
  html += '</table>';
  container.innerHTML = html;
}

/* Escape HTML helper */
function escapeHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

/* Compute missing % and shape string */
function computeMissingAndShape(data) {
  if (!data || data.length === 0) return 'No data';
  const n = data.length;
  const cols = Object.keys(data[0]);
  const missing = cols.map(c => {
    const m = data.reduce((acc,r) => acc + ((r[c] === null || r[c] === undefined || r[c] === '') ? 1 : 0), 0);
    return `${c}: ${m} missing (${(m/n*100).toFixed(2)}%)`;
  });
  return `Rows: ${n}, Columns: ${cols.length}\n` + missing.join('\n');
}

/* Render survival charts by Sex and Pclass */
function renderSurvivalCharts(data) {
  // by Sex
  const bySex = {};
  data.forEach(r => {
    const s = (r['Sex'] || '').toString() || 'NA';
    if (!bySex[s]) bySex[s] = {total:0, surv:0};
    bySex[s].total++;
    if (Number(r[TARGET_COL]) === 1) bySex[s].surv++;
  });
  const sexSeries = Object.keys(bySex).map(k => ({label: k, value: bySex[k].surv / bySex[k].total}));
  tfvis.render.barchart({name: 'Survival rate by Sex', tab: 'Data'}, { values: sexSeries }, { height: 180 });

  // by Pclass
  const byP = {};
  data.forEach(r => {
    const p = String(r['Pclass'] || 'NA');
    if (!byP[p]) byP[p] = {total:0, surv:0};
    byP[p].total++;
    if (Number(r[TARGET_COL]) === 1) byP[p].surv++;
  });
  const pSeries = Object.keys(byP).sort().map(k => ({label: 'P' + k, value: byP[k].surv / byP[k].total}));
  tfvis.render.barchart({name: 'Survival rate by Pclass', tab: 'Data'}, { values: pSeries }, { height: 180 });
}

/* Preprocess function: returns processed rows + feature names + scalers */
function preprocess(trainRowsRaw, testRowsRaw = null, addFamily = true) {
  // Convert inputs to deep copy
  const trainRows = trainRowsRaw.map(r => ({...r}));
  const testRows = testRowsRaw ? testRowsRaw.map(r => ({...r})) : null;

  // 1) compute medians/modes on train
  function toNum(x) { const v = Number(x); return isNaN(v) ? null : v; }
  const ages = trainRows.map(r => toNum(r['Age'])).filter(v => v !== null);
  const fares = trainRows.map(r => toNum(r['Fare'])).filter(v => v !== null);
  function median(arr) {
    if (!arr.length) return 0;
    const a = arr.slice().sort((x,y)=>x-y);
    const m = Math.floor(a.length/2);
    return a.length%2 ? a[m] : (a[m-1]+a[m])/2;
  }
  function mode(arr) {
    const counts = {};
    arr.forEach(x => { if (x!==null && x!==undefined && x!=='') counts[x] = (counts[x]||0)+1; });
    let best=null, bestC=-1;
    Object.keys(counts).forEach(k=>{ if (counts[k]>bestC){ bestC=counts[k]; best=k; }});
    return best;
  }
  const ageMed = median(ages);
  const fareMed = median(fares);
  const embMode = mode(trainRows.map(r => r['Embarked'] || '') ) || 'S';

  // 2) Fill missing and compute FamilySize/IsAlone
  const fillRow = (r) => {
    if (r['Age'] === null || r['Age'] === undefined || r['Age'] === '') r['Age'] = ageMed;
    if (r['Fare'] === null || r['Fare'] === undefined || r['Fare'] === '') r['Fare'] = fareMed;
    if (!r['Embarked'] || r['Embarked'] === '') r['Embarked'] = embMode;
    r['SibSp'] = r['SibSp'] !== undefined ? Number(r['SibSp']) : 0;
    r['Parch'] = r['Parch'] !== undefined ? Number(r['Parch']) : 0;
    r['FamilySize'] = (Number(r['SibSp']) || 0) + (Number(r['Parch']) || 0) + 1;
    r['IsAlone'] = r['FamilySize'] === 1 ? 1 : 0;
  };
  trainRows.forEach(fillRow);
  if (testRows) testRows.forEach(fillRow);

  // 3) categories values determined from train (for one-hot)
  const sexCats = Array.from(new Set(trainRows.map(r => (r['Sex']||'').toString()))).filter(Boolean).sort();
  const pclassCats = Array.from(new Set(trainRows.map(r => r['Pclass']))).sort();
  const embarkedCats = Array.from(new Set(trainRows.map(r => (r['Embarked']||'').toString()))).filter(Boolean).sort();

  // 4) Build feature list (Age_std, Fare_std, SibSp, Parch, optional FamilySize, IsAlone, one-hots)
  const featureNames = [];
  featureNames.push('Age','Fare','SibSp','Parch');
  if (addFamily) { featureNames.push('FamilySize','IsAlone'); }
  sexCats.forEach(s => featureNames.push('Sex_' + s));
  pclassCats.forEach(p => featureNames.push('Pclass_' + p));
  embarkedCats.forEach(e => featureNames.push('Embarked_' + e));

  // 5) compute mean/std for Age & Fare on train for standardization
  const ageVals = trainRows.map(r => Number(r['Age']));
  const fareVals = trainRows.map(r => Number(r['Fare']));
  const stats = (arr) => {
    const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
    const std = Math.sqrt(arr.reduce((s,v)=>s+(v-mean)*(v-mean),0)/arr.length) || 1;
    return {mean,std};
  };
  const ageStats = stats(ageVals);
  const fareStats = stats(fareVals);

  // 6) function to produce flat feature object per row
  function makeFeatureObject(r) {
    const obj = {};
    obj['Age'] = (Number(r['Age']) - ageStats.mean) / ageStats.std;
    obj['Fare'] = (Number(r['Fare']) - fareStats.mean) / fareStats.std;
    obj['SibSp'] = Number(r['SibSp']) || 0;
    obj['Parch'] = Number(r['Parch']) || 0;
    if (addFamily) { obj['FamilySize'] = r['FamilySize']; obj['IsAlone'] = r['IsAlone']; }
    sexCats.forEach(s => obj['Sex_' + s] = (r['Sex'] === s) ? 1 : 0);
    pclassCats.forEach(p => obj['Pclass_' + p] = (Number(r['Pclass']) === Number(p)) ? 1 : 0);
    embarkedCats.forEach(e => obj['Embarked_' + e] = (r['Embarked'] === e) ? 1 : 0);
    // keep ID & target for convenience
    obj[ID_COL] = r[ID_COL];
    if (r[TARGET_COL] !== undefined) obj[TARGET_COL] = Number(r[TARGET_COL]);
    return obj;
  }

  const trainFeatures = trainRows.map(makeFeatureObject);
  const testFeatures = testRows ? testRows.map(makeFeatureObject) : null;

  // 7) stratified 80/20 split on trainFeatures by target
  const grouped = {};
  trainFeatures.forEach(r => {
    const label = String(r[TARGET_COL] || 0);
    if (!grouped[label]) grouped[label] = [];
    grouped[label].push(r);
  });
  const trainSplit = [], valSplit = [];
  Object.keys(grouped).forEach(lbl => {
    const arr = shuffleArray(grouped[lbl].slice());
    const cutoff = Math.floor(arr.length * 0.8);
    trainSplit.push(...arr.slice(0,cutoff));
    valSplit.push(...arr.slice(cutoff));
  });

  // 8) Return processed object
  return {
    trainRows: shuffleArray(trainSplit),
    valRows: shuffleArray(valSplit),
    testRows: testFeatures,
    featureNames,
    scalers: { ageStats, fareStats },
    categories: { sexCats, pclassCats, embarkedCats }
  };
}

/* Build tf.sequential model as specified */
function buildModel(inputDim) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputDim] }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  m.compile({ optimizer: tf.train.adam(), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return m;
}

/* Convert processed rows arrays to tensors for training */
function toTensors(trainRows, valRows, featureNames) {
  const trainXs = tf.tensor2d(trainRows.map(r => featureNames.map(fn => r[fn])));
  const trainYs = tf.tensor2d(trainRows.map(r => [Number(r[TARGET_COL])]));
  const valXs = tf.tensor2d(valRows.map(r => featureNames.map(fn => r[fn])));
  const valYs = tf.tensor2d(valRows.map(r => [Number(r[TARGET_COL])]));
  return { trainXs, trainYs, valXs, valYs };
}

/* Utility: shuffle array in-place copy */
function shuffleArray(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/* Compute ROC curve and AUC (simple trapezoidal) */
function computeROC(labels, scores) {
  // labels: array 0/1, scores: probabilities
  const paired = labels.map((lab, i) => ({ lab: Number(lab), score: scores[i] }));
  paired.sort((a,b) => b.score - a.score);
  const P = labels.reduce((s,v) => s + (v===1?1:0), 0);
  const N = labels.length - P;
  let tp=0, fp=0;
  const points = [];
  let prevScore = null;
  for (let i=0;i<paired.length;i++) {
    const p = paired[i];
    if (prevScore === null || p.score !== prevScore) {
      points.push({ fpr: N===0?0:fp/N, tpr: P===0?0:tp/P });
      prevScore = p.score;
    }
    if (p.lab === 1) tp++; else fp++;
  }
  points.push({ fpr: N===0?0:fp/N, tpr: P===0?0:tp/P });
  // trapezoidal AUC
  let auc = 0;
  for (let i=1;i<points.length;i++) {
    const x1 = points[i-1].fpr, x2 = points[i].fpr;
    const y1 = points[i-1].tpr, y2 = points[i].tpr;
    auc += (x2-x1) * (y1 + y2) / 2;
  }
  return { points, auc };
}

/* Update confusion matrix and precision/recall/F1 */
function updateConfusionMatrix(trueLabels, probs, threshold=0.5) {
  let tp=0, tn=0, fp=0, fn=0;
  for (let i=0;i<trueLabels.length;i++) {
    const t = Number(trueLabels[i]);
    const p = probs[i] >= threshold ? 1 : 0;
    if (t===1 && p===1) tp++;
    if (t===0 && p===0) tn++;
    if (t===0 && p===1) fp++;
    if (t===1 && p===0) fn++;
  }
  const acc = (tp+tn) / (tp+tn+fp+fn);
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = (precision + recall) === 0 ? 0 : 2 * precision * recall / (precision + recall);

  $('confusion').innerHTML = `<table>
    <tr><th></th><th>Pred 1</th><th>Pred 0</th></tr>
    <tr><th>Actual 1</th><td>${tp}</td><td>${fn}</td></tr>
    <tr><th>Actual 0</th><td>${fp}</td><td>${tn}</td></tr>
  </table>`;
  $('prf').textContent = `Accuracy: ${(acc*100).toFixed(2)}%  •  Precision: ${precision.toFixed(4)}  •  Recall: ${recall.toFixed(4)}  •  F1: ${f1.toFixed(4)}`;
}

/* Helper: download string as file */
function downloadString(text, mime, filename) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
}

/* ---------- Error handling for missing or wrong data ---------- */
/* The code checks for missing train/test when actions require them and alerts the user. */

/* ---------- Notes for re-use / schema swap ---------- */
/*
 * If you want to adapt this app to another binary dataset:
 * - change TARGET_COL, ID_COL and FEATURE_COLS at the top
 * - verify preprocess() transforms (imputation/encoding/standardization) match new schema
 * - ensure categorical value lists are derived from train rows (the code uses train values)
 */

/* End of app.js */
