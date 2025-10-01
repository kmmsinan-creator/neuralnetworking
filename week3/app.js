// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;
let visorOpened = false; // ensure we open visor only once

// Schema configuration
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// -------------------- Data loading --------------------
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }

    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';

    try {
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);

        const testText = await readFile(testFile);
        testData = parseCSV(testText);

        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
        document.getElementById('inspect-btn').disabled = false;

        console.log("✅ Training sample:", trainData[0]);
        console.log("✅ Test sample:", testData[0]);
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// -------------------- CSV parsing (PapaParse) --------------------
function parseCSV(csvText) {
    // PapaParse handles quoted fields with commas, dynamic typing, and skipping empty lines
    const results = Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });
    if (results.errors && results.errors.length) {
        console.warn("PapaParse warnings/errors:", results.errors);
    }
    return results.data;
}

// -------------------- Inspect --------------------
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }

    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));

    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';

    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;

    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}%</li>`;
    });
    missingInfo += '</ul>';

    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;

    createVisualizations();
    document.getElementById('preprocess-btn').disabled = false;
}

function createPreviewTable(data) {
    const table = document.createElement('table');

    // header
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = (value === null || value === undefined) ? 'NULL' : String(value);
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    return table;
}

// -------------------- Visualizations (tfjs-vis) --------------------
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';

    // Survival by Sex
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex !== undefined && row.Survived !== undefined) {
            if (!survivalBySex[row.Sex]) survivalBySex[row.Sex] = { survived: 0, total: 0 };
            survivalBySex[row.Sex].total++;
            if (row.Survived === 1) survivalBySex[row.Sex].survived++;
        }
    });
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        sex,
        survivalRate: (stats.survived / stats.total) * 100
    }));

    tfvis.render.barchart(
        { name: 'Survival Rate by Sex', tab: 'Charts' },
        sexData.map(d => ({ x: d.sex, y: d.survivalRate })),
        { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
    );

    // Survival by Pclass
    const survivalByPclass = {};
    trainData.forEach(row => {
        if (row.Pclass !== undefined && row.Survived !== undefined) {
            if (!survivalByPclass[row.Pclass]) survivalByPclass[row.Pclass] = { survived: 0, total: 0 };
            survivalByPclass[row.Pclass].total++;
            if (row.Survived === 1) survivalByPclass[row.Pclass].survived++;
        }
    });
    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
        pclass: `Class ${pclass}`,
        survivalRate: (stats.survived / stats.total) * 100
    }));

    tfvis.render.barchart(
        { name: 'Survival Rate by Passenger Class', tab: 'Charts' },
        pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })),
        { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' }
    );

    chartsDiv.innerHTML += '<p>Charts open in the tfjs-vis visor (floating panel). Press <b>Shift+V</b> if it is hidden.</p>';

    // Open the visor once so the floating button is visible
    try {
        if (!visorOpened && tfvis && tfvis.visor) {
            tfvis.visor().toggle();
            visorOpened = true;
        }
    } catch (err) {
        console.warn("tfvis visor toggle failed:", err);
    }
}

// -------------------- Preprocessing --------------------
function preprocessData() {
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';

    try {
        if (!trainData || !testData) {
            outputDiv.innerHTML = `<p style="color:red">❌ Data not loaded. Please load CSV files first.</p>`;
            return;
        }
        if (trainData.length === 0 || testData.length === 0) {
            outputDiv.innerHTML = `<p style="color:red">❌ CSV files are empty or parsed incorrectly.</p>`;
            return;
        }

        console.log("Preprocessing - sample train row:", trainData[0]);
        console.log("Preprocessing - sample test row:", testData[0]);

        // Imputation values
        const ageMedian = calculateMedian(trainData.map(r => r.Age).filter(v => v !== null && v !== undefined));
        const fareMedian = calculateMedian(trainData.map(r => r.Fare).filter(v => v !== null && v !== undefined));
        const embarkedMode = calculateMode(trainData.map(r => r.Embarked).filter(v => v !== null && v !== undefined));

        // Build arrays
        preprocessedTrainData = { features: [], labels: [] };
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            // ensure label is numeric (for safety)
            preprocessedTrainData.labels.push(Number(row[TARGET_FEATURE] === undefined || row[TARGET_FEATURE] === null ? 0 : row[TARGET_FEATURE]));
        });

        preprocessedTestData = { features: [], passengerIds: [] };
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });

        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels, 'float32');

        outputDiv.innerHTML = `
            <p>✅ Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: [${preprocessedTestData.features.length}, ${preprocessedTestData.features[0]?.length || 0}]</p>
        `;
        document.getElementById('create-model-btn').disabled = false;

    } catch (error) {
        outputDiv.innerHTML = `<p style="color:red">❌ Error during preprocessing: ${error.message}</p>`;
        console.error("Preprocessing error:", error);
    }
}

// Extract features from a row
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    const age = (row.Age !== null && row.Age !== undefined && !isNaN(row.Age)) ? row.Age : ageMedian;
    const fare = (row.Fare !== null && row.Fare !== undefined && !isNaN(row.Fare)) ? row.Fare : fareMedian;
    const embarked = (row.Embarked !== null && row.Embarked !== undefined) ? row.Embarked : embarkedMode;

    const stdAge = (age - ageMedian) / (calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null && a !== undefined)) || 1);
    const stdFare = (fare - fareMedian) / (calculateStdDev(trainData.map(r => r.Fare).filter(f => f !== null && f !== undefined)) || 1);

    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]);
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);

    let features = [
        stdAge,
        stdFare,
        (row.SibSp !== undefined && row.SibSp !== null) ? row.SibSp : 0,
        (row.Parch !== undefined && row.Parch !== null) ? row.Parch : 0
    ];

    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);

    if (document.getElementById('add-family-features')?.checked) {
        const familySize = ((row.SibSp || 0) + (row.Parch || 0) + 1);
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }

    return features;
}

// -------------------- Helper stats functions --------------------
function calculateMedian(values) {
    const arr = values.filter(v => v !== null && v !== undefined && !isNaN(v)).slice();
    if (arr.length === 0) return 0;
    arr.sort((a, b) => a - b);
    const half = Math.floor(arr.length / 2);
    if (arr.length % 2 === 0) return (arr[half - 1] + arr[half]) / 2;
    return arr[half];
}

function calculateMode(values) {
    const arr = values.filter(v => v !== null && v !== undefined);
    if (arr.length === 0) return null;
    const freq = {};
    let max = 0;
    let mode = null;
    arr.forEach(v => {
        freq[v] = (freq[v] || 0) + 1;
        if (freq[v] > max) {
            max = freq[v];
            mode = v;
        }
    });
    return mode;
}

function calculateStdDev(values) {
    const arr = values.filter(v => v !== null && v !== undefined && !isNaN(v));
    if (arr.length === 0) return 0;
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    const sq = arr.map(v => Math.pow(v - mean, 2));
    const variance = sq.reduce((s, v) => s + v, 0) / arr.length;
    return Math.sqrt(variance);
}

function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const idx = categories.indexOf(value);
    if (idx !== -1) encoding[idx] = 1;
    return encoding;
}

// -------------------- Model --------------------
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }

    const inputShape = preprocessedTrainData.features.shape[1];

    model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputShape] }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    // Use 'accuracy' metric (we will support both logs.acc and logs.accuracy in status)
    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => {
        summaryText += `<li>Layer ${i+1}: ${layer.getClassName()} - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += `</ul><p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML += summaryText;

    document.getElementById('train-btn').disabled = false;
}

// -------------------- Training --------------------
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }

    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';

    try {
        const total = preprocessedTrainData.features.shape[0];
        const nFeatures = preprocessedTrainData.features.shape[1];
        const splitIndex = Math.floor(total * 0.8);

        const trainFeatures = preprocessedTrainData.features.slice([0, 0], [splitIndex, nFeatures]);
        const trainLabels = preprocessedTrainData.labels.slice([0], [splitIndex]);

        const valFeatures = preprocessedTrainData.features.slice([splitIndex, 0], [total - splitIndex, nFeatures]);
        const valLabels = preprocessedTrainData.labels.slice([splitIndex], [total - splitIndex]);

        validationData = valFeatures;
        validationLabels = valLabels;

        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: [
                tfvis.show.fitCallbacks(
                    { name: 'Training Performance' },
                    ['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                    { callbacks: ['onEpochEnd'] }
                ),
                {
                    onEpochEnd: (epoch, logs) => {
                        const acc = (logs.acc !== undefined) ? logs.acc : logs.accuracy;
                        const valAcc = (logs.val_acc !== undefined) ? logs.val_acc : logs.val_accuracy;
                        statusDiv.innerHTML = `Epoch ${epoch + 1}/50 - loss: ${logs.loss.toFixed(4)}, acc: ${ (acc !== undefined && acc !== null) ? acc.toFixed(4) : 'N/A' }, val_loss: ${logs.val_loss?.toFixed(4) || 'N/A'}, val_acc: ${(valAcc !== undefined && valAcc !== null) ? valAcc.toFixed(4) : 'N/A'}`;
                    }
                }
            ]
        });

        statusDiv.innerHTML += '<p>Training completed!</p>';

        validationPredictions = model.predict(validationData);

        document.getElementById('threshold-slider').disabled = false;
        // avoid adding duplicate listeners
        const slider = document.getElementById('threshold-slider');
        slider.removeEventListener('input', updateMetrics);
        slider.addEventListener('input', updateMetrics);

        document.getElementById('predict-btn').disabled = false;

        await updateMetrics();
    } catch (error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        console.error(error);
    }
}

// -------------------- Metrics & ROC --------------------
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;

    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    const predValsRaw = validationPredictions.arraySync();
    const trueValsRaw = validationLabels.arraySync();

    // Normalize predictions to flat numbers
    const predVals = predValsRaw.map(p => Array.isArray(p) ? p[0] : p);
    const trueVals = trueValsRaw.map(v => Array.isArray(v) ? v[0] : v);

    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < predVals.length; i++) {
        const prediction = predVals[i] >= threshold ? 1 : 0;
        const actual = trueVals[i];
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 1) fn++;
    }

    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <table>
            <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;

    const precision = (tp + fp) ? tp / (tp + fp) : 0;
    const recall = (tp + fn) ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) ? 2 * (precision * recall) / (precision + recall) : 0;
    const accuracy = (tp + tn + fp + fn) ? (tp + tn) / (tp + tn + fp + fn) : 0;

    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <p>Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>
    `;

    await plotROC(trueVals, predVals);
}

async function plotROC(trueLabels, predictions) {
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const rocData = [];

    thresholds.forEach(th => {
        let tp = 0, fn = 0, fp = 0, tn = 0;
        for (let i = 0; i < predictions.length; i++) {
            const p = predictions[i] >= th ? 1 : 0;
            if (trueLabels[i] === 1) {
                if (p === 1) tp++; else fn++;
            } else {
                if (p === 1) fp++; else tn++;
            }
        }
        const tpr = (tp + fn) ? tp / (tp + fn) : 0;
        const fpr = (fp + tn) ? fp / (fp + tn) : 0;
        rocData.push({ threshold: th, fpr, tpr });
    });

    // AUC (trapezoid)
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        const x1 = rocData[i-1].fpr, x2 = rocData[i].fpr;
        const y1 = rocData[i-1].tpr, y2 = rocData[i].tpr;
        auc += (x2 - x1) * (y1 + y2) / 2;
    }

    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: rocData.map(d => ({ x: d.fpr, y: d.tpr })) },
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            width: 400,
            height: 400
        }
    );

    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML += `<p>AUC: ${auc.toFixed(4)}</p>`;
}

// -------------------- Prediction --------------------
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }

    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';

    try {
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        testPredictions = model.predict(testFeatures);
        const predValsRaw = testPredictions.arraySync();
        const predVals = predValsRaw.map(p => Array.isArray(p) ? p[0] : p);

        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predVals[i] >= 0.5 ? 1 : 0,
            Probability: predVals[i]
        }));

        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples</p>`;

        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        console.error(error);
    }
}

function createPredictionTable(data) {
    const table = document.createElement('table');
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th'); th.textContent = header; headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    data.forEach(row => {
        const tr = document.createElement('tr');
        ['PassengerId', 'Survived', 'Probability'].forEach(key => {
            const td = document.createElement('td');
            td.textContent = key === 'Probability' ? row[key].toFixed(4) : row[key];
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    return table;
}

// -------------------- Export --------------------
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }

    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';

    try {
        const predValsRaw = testPredictions.arraySync();
        const predVals = predValsRaw.map(p => Array.isArray(p) ? p[0] : p);

        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            submissionCSV += `${id},${predVals[i] >= 0.5 ? 1 : 0}\n`;
        });

        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            probabilitiesCSV += `${id},${predVals[i].toFixed(6)}\n`;
        });

        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'submission.csv';

        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
        probabilitiesLink.download = 'probabilities.csv';

        submissionLink.click();
        probabilitiesLink.click();

        await model.save('downloads://titanic-tfjs-model');

        statusDiv.innerHTML = `
            <p>Export completed!</p>
            <p>Downloaded: submission.csv</p>
            <p>Downloaded: probabilities.csv</p>
            <p>Model saved to browser downloads</p>
        `;
    } catch (error) {
        statusDiv.innerHTML = `Error during export: ${error.message}`;
        console.error(error);
    }
}
