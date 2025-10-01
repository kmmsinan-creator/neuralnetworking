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

// Schema configuration
const TARGET_FEATURE = 'Survived'; // Binary classification target
const ID_FEATURE = 'PassengerId'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// Load data from uploaded CSV files
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

        console.log("✅ Training data sample:", trainData[0]);
        console.log("✅ Test data sample:", testData[0]);
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// ✅ FIXED CSV parser using PapaParse
function parseCSV(csvText) {
    const results = Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });
    return results.data;
}

// Inspect the loaded data
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

// Create preview table
function createPreviewTable(data) {
    const table = document.createElement('table');
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null ? value : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    return table;
}

// Visualizations (tfjs-vis) + auto show visor
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined) {
            if (!survivalBySex[row.Sex]) survivalBySex[row.Sex] = { survived: 0, total: 0 };
            survivalBySex[row.Sex].total++;
            if (row.Survived === 1) survivalBySex[row.Sex].survived++;
        }
    });
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        sex,
        survivalRate: (stats.survived / stats.total) * 100
    }));
    tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Charts' },
        sexData.map(d => ({ x: d.sex, y: d.survivalRate })), { xLabel: 'Sex', yLabel: 'Survival Rate (%)' });
    
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
    tfvis.render.barchart({ name: 'Survival Rate by Passenger Class', tab: 'Charts' },
        pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })), { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' });
    
    chartsDiv.innerHTML += '<p>Charts open in the tfjs-vis visor (floating panel). Press <b>Shift+V</b> if it is hidden.</p>';
    tfvis.visor().toggle(); // ✅ force button to appear
}

// ✅ Improved preprocessing with clear errors
function preprocessData() {
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    
    try {
        if (!trainData || !testData) {
            outputDiv.innerHTML = `<p style="color:red">❌ Data not loaded. Please load CSVs first.</p>`;
            return;
        }
        if (trainData.length === 0 || testData.length === 0) {
            outputDiv.innerHTML = `<p style="color:red">❌ CSV files are empty or parsed incorrectly.</p>`;
            return;
        }

        // Median/mode imputation values
        const ageMedian = calculateMedian(trainData.map(r => r.Age).filter(v => v !== null));
        const fareMedian = calculateMedian(trainData.map(r => r.Fare).filter(v => v !== null));
        const embarkedMode = calculateMode(trainData.map(r => r.Embarked).filter(v => v !== null));

        preprocessedTrainData = { features: [], labels: [] };
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });

        preprocessedTestData = { features: [], passengerIds: [] };
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });

        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);

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

// --- keep rest of your existing code: extractFeatures, calculateMedian/Mode/StdDev, createModel, trainModel (with fixed callbacks), updateMetrics, predict, exportResults ---
