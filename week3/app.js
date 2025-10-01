// Titanic Binary Classifier Web App - TF.js
// Schema swap note: To reuse, change the features below accordingly.

// --- Global state ---
let trainData = null, testData = null;
let preprocessedTrainData = null, preprocessedTestData = null;
let model = null, trainingHistory = null;
let validationData = null, validationLabels = null;
let validationPredictions = null, testPredictions = null;

// --- Dataset schema (change for other datasets) ---
const TARGETFEATURE = 'Survived';    // Binary classification target
const IDFEATURE = 'PassengerId';    // Identifier, exclude from features
const NUMERICALFEATURES = ['Age','Fare','SibSp','Parch'];
const CATEGORICALFEATURES = ['Pclass','Sex','Embarked'];

// ========== UTILITIES ==========
// Robust CSV parser supporting quoted, escaped commas
function parseCSV(csvText) {
    const rows = [];
    const lines = csvText.replace(/\r/g,'').split('\n').filter(l => l.trim());
    if (lines.length === 0) return [];
    let headerParts = parseCSVRow(lines[0]);
    const headers = headerParts.map(h => h.trim());
    for (let i = 1; i < lines.length; ++i) {
        let rowParts = parseCSVRow(lines[i]);
        if (rowParts.length !== headers.length) continue;
        const obj = {};
        headers.forEach((h, idx) => {
            let v = rowParts[idx];
            if (v === "") obj[h] = null;
            else if (!isNaN(v) && NUMERICALFEATURES.includes(h))
                obj[h] = v === null ? null : parseFloat(v);
            else obj[h] = v;
        });
        rows.push(obj);
    }
    return rows;
}
// Helper for quoted, escaped commas:
function parseCSVRow(line) {
    const res = [];
    let i = 0, val = '', inQuotes = false, quoteChar = '';
    while (i < line.length) {
        const ch = line[i];
        if (ch === '"' || ch === "'") {
            if (!inQuotes) {
                inQuotes = true; quoteChar = ch;
            } else if (ch === quoteChar) {
                inQuotes = false;
            } else val += ch;
        } else if (ch === ',' && !inQuotes) {
            res.push(val); val = '';
        } else val += ch;
        i++;
    }
    res.push(val);
    return res.map(v => v.trim());
}

// File input: read as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// ========== DATA LOAD & INSPECTION ==========
async function loadData() {
    const trainFileInput = document.getElementById('train-file');
    const testFileInput = document.getElementById('test-file');
    const statusDiv = document.getElementById('data-status');
    const inspectBtn = document.getElementById('inspect-btn');
    inspectBtn.disabled = true; // Only enabled after load success

    const trainFile = trainFileInput.files[0];
    const testFile = testFileInput.files[0];
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    statusDiv.innerHTML = 'Loading data...';
    try {
        const trainText = await readFile(trainFile);
        const testText = await readFile(testFile);

        trainData = parseCSV(trainText);
        testData = parseCSV(testText);

        if (!trainData.length || !testData.length) throw new Error("One or both CSV files are empty or corrupted.");

        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples.`;
        inspectBtn.disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        alert(`Error loading data: ${error.message}`);
        console.error(error);
    }
}

// Preview, stats, missing %
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    // Preview table
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0,10)));

    // Data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGETFEATURE]==="1" || row[TARGETFEATURE]===1).length;
    const survivalRate = trainData.length ? (survivalCount / trainData.length * 100).toFixed(2) : "0";
    let missingInfo = "<h4>Missing Values %</h4><ul>";
    Object.keys(trainData[0]).forEach(f => {
        const missingCount = trainData.filter(row => row[f] == null || row[f] === '').length;
        const missingPercent = trainData.length ? (missingCount / trainData.length * 100).toFixed(2) : "0";
        missingInfo += `<li>${f}: ${missingPercent}%</li>`;
    });
    missingInfo += "</ul>";
    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>Survival Rate: ${survivalCount}/${trainData.length} (${survivalRate}%)</p>${missingInfo}`;
    createVisualizations();
    document.getElementById('preprocess-btn').disabled = false;
}

// --- Table rendering ---
function createPreviewTable(data) {
    const table = document.createElement('table');
    // Header
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    // Data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(val => {
            const td = document.createElement('td');
            td.textContent = val != null ? val : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    return table;
}

// Data visualizations: Survival by Sex, Class
function createVisualizations() {
    if (!window.trainData || !trainData.length) return;
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';

    // Survival by Sex
    let bySex = {};
    trainData.forEach(row => {
        const sex = row['Sex'];
        const survived = (row[TARGETFEATURE] === "1" || row[TARGETFEATURE] === 1) ? 1 : 0;
        if (!bySex[sex]) bySex[sex] = { survived: 0, total: 0 };
        bySex[sex].total++;
        if (survived) bySex[sex].survived++;
    });
    const sexData = Object.keys(bySex).map(sex => ({
        sex,
        survivalRate: bySex[sex].total ? bySex[sex].survived / bySex[sex].total * 100 : 0
    }));
    tfvis.render.barchart(
        {name:'Survival Rate by Sex', tab:'Charts'},
        sexData.map(d => ({x: d.sex, y: d.survivalRate})),
        {xLabel:'Sex', yLabel:'Survival Rate (%)'}
    );

    // Survival by Pclass
    let byClass = {};
    trainData.forEach(row => {
        const cls = row['Pclass'];
        const survived = (row[TARGETFEATURE] === "1" || row[TARGETFEATURE] === 1) ? 1 : 0;
        if (!byClass[cls]) byClass[cls] = { survived: 0, total: 0 };
        byClass[cls].total++;
        if (survived) byClass[cls].survived++;
    });
    const classData = Object.keys(byClass).map(cls => ({
        cls,
        survivalRate: byClass[cls].total ? byClass[cls].survived / byClass[cls].total * 100 : 0
    }));
    tfvis.render.barchart(
        {name:'Survival Rate by Class', tab:'Charts'},
        classData.map(d => ({x: d.cls, y: d.survivalRate})),
        {xLabel:'Class', yLabel:'Survival Rate (%)'}
    );

    chartsDiv.innerHTML += `<p>Charts shown via tfjs-vis visor (see icon at bottom right).</p>`;
}

// ========== PREPROCESSING ==========
function preprocessData() {
    if (!trainData || !testData) {
        alert("Please load data first.");
        return;
    }
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    try {
        // Impute Age/Fare (median), Embarked (mode)
        const ageMedian = calculateMedian(trainData.map(r => r['Age']).filter(a => a !== null && a !== ''));
        const fareMedian = calculateMedian(trainData.map(r => r['Fare']).filter(f => f !== null && f !== ''));
        const embarkedMode = calculateMode(trainData.map(r => r['Embarked']).filter(e => e != null && e !== ''));

        // Preprocess train
        preprocessedTrainData = {features:[],labels:[]};
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push((row[TARGETFEATURE]==="1"||row[TARGETFEATURE]===1)?1:0);
        });

        // Preprocess test
        preprocessedTestData = {features:[],passengerIds:[]};
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[IDFEATURE]);
        });

        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);

        outputDiv.innerHTML = `<p>Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: ${preprocessedTestData.features.length} x ${(preprocessedTestData.features[0] ? preprocessedTestData.features[0].length : 0)}</p>`;
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}

// Impute/standardize/one-hot toggle family features
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute
    const age = row['Age'] !== null ? row['Age'] : ageMedian;
    const fare = row['Fare'] !== null ? row['Fare'] : fareMedian;
    const embarked = row['Embarked'] !== null && row['Embarked'] !== '' ? row['Embarked'] : embarkedMode;

    // Standardize Age/Fare
    const ageStd = calculateStdDev(trainData.map(r => r['Age']).filter(a => a !== null && a !== '')) || 1;
    const fareStd = calculateStdDev(trainData.map(r => r['Fare']).filter(f => f !== null && f !== '')) || 1;
    const standardizedAge = ageStd ? (age - ageMedian) / ageStd : 0;
    const standardizedFare = fareStd ? (fare - fareMedian) / fareStd : 0;

    // One-hot encode categorical
    const pclassOneHot = oneHotEncode(row['Pclass'], [1,2,3]);
    const sexOneHot = oneHotEncode(row['Sex'], ['male','female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C','Q','S']);

    // Numerical features
    let features = [standardizedAge, standardizedFare, row['SibSp']||0, row['Parch']||0];
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);

    // Optionally include family features
    if (document.getElementById('add-family-features').checked) {
        const familySize = (row['SibSp']||0) + (row['Parch']||0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }
    return features;
}

function calculateMedian(values) {
    if (!values.length) return 0;
    values = values.slice().sort((a,b)=>a-b);
    const half = Math.floor(values.length/2);
    if (values.length % 2)
        return values[half];
    return (values[half-1] + values[half])/2.0;
}
function calculateMode(values) {
    if (!values.length) return null;
    const count = {};
    let mode = values[0], max = 0;
    for (const v of values) {
        count[v] = (count[v]||0)+1;
        if(count[v]>max) { max = count[v]; mode = v; }
    }
    return mode;
}
function calculateStdDev(values) {
    if (!values.length) return 0;
    const mean = values.reduce((a,b)=>a+b,0)/values.length;
    const variance = values.map(v => Math.pow(v-mean,2)).reduce((a,b)=>a+b,0)/values.length;
    return Math.sqrt(variance);
}
function oneHotEncode(value, categories) {
    const arr = new Array(categories.length).fill(0);
    const idx = categories.indexOf(value);
    if (idx !== -1) arr[idx] = 1;
    return arr;
}

// ========== MODEL ==========
function createModel() {
    if (!preprocessedTrainData) {
        alert("Please preprocess data first.");
        return;
    }
    const inputShape = preprocessedTrainData.features.shape[1];
    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [inputShape], units: 16, activation: 'relu'}));
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    // Print summary (browser can't use .summary(), so basic info)
    const summaryDiv = document.getElementById('model-summary');
    let summaryText = '<ul>';
    model.layers.forEach((layer,i) => {
        summaryText += `<li>Layer ${i+1}: ${layer.getClassName()} - Output Shape ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML = summaryText;
    document.getElementById('train-btn').disabled = false;
}

// ========== TRAINING ==========
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert("Please create model first.");
        return;
    }
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';
    try {
        // 80/20 stratified split
        const splitIdx = Math.floor(preprocessedTrainData.features.shape[0]*0.8);
        const trainFeatures = preprocessedTrainData.features.slice([0,0],[splitIdx,-1]);
        const trainLabels = preprocessedTrainData.labels.slice([0],[splitIdx]);
        const valFeatures = preprocessedTrainData.features.slice([splitIdx,0],[-1,-1]);
        const valLabels = preprocessedTrainData.labels.slice([splitIdx],[-1]);
        validationData = valFeatures;
        validationLabels = valLabels;

        trainingHistory = await model.fit(
            trainFeatures, trainLabels,
            {
                epochs: 50,
                batchSize: 32,
                validationData:[valFeatures,valLabels],
                callbacks: tfvis.show.fitCallbacks(
                    {name:'Training Performance',tab:'Training'},
                    ['loss','acc','val_loss','val_acc'],
                    {callbacks:['onEpochEnd'],metrics:['loss','accuracy']}
                )
            }
        );
        statusDiv.innerHTML = '<p>Training completed!</p>';
        // Predict on validation set
        validationPredictions = model.predict(validationData);
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('predict-btn').disabled = false;
        document.getElementById('predict-btn').disabled = false;
        // Setup slider and evaluation
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
        updateMetrics();
    } catch(error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        console.error(error);
    }
}

// ========== METRICS ==========
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    const thresholdInput = document.getElementById('threshold-slider');
    const threshold = parseFloat(thresholdInput.value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    const predVals = Array.from(await validationPredictions.dataSync());
    const trueVals = Array.from(await validationLabels.dataSync());

    let tp=0, tn=0, fp=0, fn=0;
    for (let i=0; i<predVals.length; ++i) {
        const pred = predVals[i] >= threshold ? 1 : 0;
        const actual = trueVals[i];
        if (pred===1 && actual===1) tp++;
        else if (pred===0 && actual===0) tn++;
        else if (pred===1 && actual===0) fp++;
        else if (pred===0 && actual===1) fn++;
    }
    // Confusion matrix
    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `<table>
        <tr><th></th><th>Predicted: 1</th><th>Predicted: 0</th></tr>
        <tr><th>Actual: 1</th><td>${tp}</td><td>${fn}</td></tr>
        <tr><th>Actual: 0</th><td>${fp}</td><td>${tn}</td></tr>
    </table>`;
    // Metrics
    const precision = tp+fp ? tp/(tp+fp) : 0;
    const recall = tp+fn ? tp/(tp+fn) : 0;
    const f1 = precision+recall ? 2*precision*recall/(precision+recall):0;
    const accuracy = tp+tn+fp+fn ? (tp+tn)/(tp+tn+fp+fn) : 0;

    let metricsHtml = `<p>Accuracy: ${(accuracy*100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>`;
    // ROC, AUC
    const rocOut = await plotROC(trueVals,predVals);
    metricsHtml += `<p>AUC: ${rocOut.auc.toFixed(4)}</p>`;
    document.getElementById('performance-metrics').innerHTML = metricsHtml;
}

async function plotROC(trueLabels, predictions) {
    const thresholds = Array.from({length:100}, (_,i)=>i/99);
    const rocData = [];
    for (let t of thresholds) {
        let tp=0, tn=0, fp=0, fn=0;
        for(let i=0;i<trueLabels.length;++i){
            const pred = predictions[i]>=t?1:0;
            const actual = trueLabels[i];
            if(pred===1 && actual===1)tp++;
            else if(pred===0 && actual===0)tn++;
            else if(pred===1 && actual===0)fp++;
            else if(pred===0 && actual===1)fn++;
        }
        const tpr = tp+fn ? tp/(tp+fn) : 0;
        const fpr = fp+tn ? fp/(fp+tn) : 0;
        rocData.push({threshold:t,fpr, tpr});
    }
    // AUC trapezoidal rule
    let auc = 0;
    for(let i=1;i<rocData.length;++i) {
        auc += (rocData[i].fpr-rocData[i-1].fpr)*(rocData[i].tpr+rocData[i-1].tpr)/2;
    }
    tfvis.render.linechart(
        {name:'ROC Curve',tab:'Evaluation'},
        rocData.map(d => ({x:d.fpr,y:d.tpr})),
        {xLabel:'False Positive Rate', yLabel:'True Positive Rate', series:'ROC Curve', width:400, height:400}
    );
    return {auc};
}

// ========== PREDICTION & EXPORT ==========
async function predict() {
    if (!model || !preprocessedTestData) {
        alert("Please train model first.");
        return;
    }
    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';
    try {
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        testPredictions = model.predict(testFeatures);
        const predVals = Array.from(await testPredictions.dataSync());
        // Prepare results table
        const results = preprocessedTestData.passengerIds.map((id,i) => ({
            PassengerId: id,
            Survived: predVals[i]>=0.5?1:0,
            Probability: predVals[i]
        }));
        outputDiv.innerHTML = "<h3>Prediction Results (First 10 Rows)</h3>";
        outputDiv.appendChild(createPredictionTable(results.slice(0,10)));
        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples.</p>`;
        document.getElementById('export-btn').disabled = false;
    } catch(error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        console.error(error);
    }
}

function createPredictionTable(rows) {
    const table = document.createElement('table');
    const fields = ['PassengerId','Survived','Probability'];
    // Header
    const headerRow = document.createElement('tr');
    fields.forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    // Data rows
    rows.forEach(row => {
        const tr = document.createElement('tr');
        fields.forEach(f => {
            const td = document.createElement('td');
            td.textContent = f === 'Probability' ? Number(row[f]).toFixed(4) : row[f];
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    return table;
}

async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert("Please make predictions first.");
        return;
    }
    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';
    try {
        const predVals = Array.from(await testPredictions.dataSync());
        // Submission CSV
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id,i) => {
            submissionCSV += `${id},${predVals[i]>=0.5?1:0}\n`;
        });
        // Probabilities CSV
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id,i) => {
            probabilitiesCSV += `${id},${predVals[i].toFixed(6)}\n`;
        });
        // Download files
        downloadTextFile(submissionCSV, 'submission.csv');
        downloadTextFile(probabilitiesCSV, 'probabilities.csv');
        // Save model
        await model.save('downloads://titanic-tfjs-model');
        statusDiv.innerHTML = `<p>Export completed!</p>
            <p>Downloaded <b>submission.csv</b> (Kaggle format), <b>probabilities.csv</b> (probabilities), and model (browser download).</p>`;
    } catch(error) {
        statusDiv.innerHTML = 'Error during export: '+error.message;
        console.error(error);
    }
}

// Download helper
function downloadTextFile(content, filename) {
    const blob = new Blob([content], {type:'text/csv'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    setTimeout(()=>document.body.removeChild(link), 100);
}

// ========== BUTTONS & EVENTS ==========
document.getElementById('load-btn').addEventListener('click', loadData);
document.getElementById('inspect-btn').addEventListener('click', inspectData);
document.getElementById('preprocess-btn').addEventListener('click', preprocessData);
document.getElementById('create-model-btn').addEventListener('click', createModel);
document.getElementById('train-btn').addEventListener('click', trainModel);
document.getElementById('predict-btn').addEventListener('click', predict);
document.getElementById('export-btn').addEventListener('click', exportResults);

// ========== END ==========
