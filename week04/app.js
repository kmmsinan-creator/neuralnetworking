// app.js - main app logic for CNN stock predictor demo

import DataLoader from './data-loader.js';
import CNNModel from './gru.js';

let loader, model, tensors;

const fileInput = document.getElementById('csv-file');
const loadBtn = document.getElementById('btn-load');
const trainBtn = document.getElementById('btn-train');
const evalBtn = document.getElementById('btn-eval');
const logDiv = document.getElementById('log');
const chartCanvas = document.getElementById('chart').getContext('2d');
let chartInstance;

function log(msg) {
  logDiv.innerHTML = `[${new Date().toLocaleTimeString()}] ${msg}<br>` + logDiv.innerHTML;
}

loadBtn.onclick = async () => {
  const file = fileInput.files[0];
  if (!file) return alert('Please upload a CSV file first.');
  try {
    log('Loading CSV file...');
    loader = new DataLoader({});
    await loader.loadFile(file);
    tensors = loader.getTensors();
    log(`Loaded data for ${tensors.symbols.length} stocks (${tensors.X_train.shape[0]} training samples).`);
  } catch (err) {
    console.error(err);
    alert('Error loading CSV: ' + err.message);
  }
};

trainBtn.onclick = async () => {
  if (!tensors) return alert('Load data first.');
  const { seqLen, symbols, forecastHorizon, X_train, y_train } = tensors;
  const numFeatures = X_train.shape[2];
  model = new CNNModel({ seqLen, numFeatures, nSymbols: symbols.length, horizon: forecastHorizon });
  model.build();
  log('Training model...');
  await model.train(X_train, y_train, {
    epochs: 10,
    onEpoch: (epoch, logs) => log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${(logs.binaryAccuracy * 100).toFixed(2)}%`)
  });
  log('Training complete.');
};

evalBtn.onclick = async () => {
  if (!model) return alert('Train model first.');
  const { X_test, y_test, symbols, forecastHorizon } = tensors;
  log('Evaluating model...');
  const results = await model.evaluate(X_test, y_test, symbols, forecastHorizon);
  renderChart(results);
  log('Evaluation complete.');
};

function renderChart(results) {
  const sorted = results.sort((a, b) => b.accuracy - a.accuracy);
  if (chartInstance) chartInstance.destroy();
  chartInstance = new Chart(chartCanvas, {
    type: 'bar',
    data: {
      labels: sorted.map(r => r.symbol),
      datasets: [{
        label: 'Accuracy (%)',
        data: sorted.map(r => (r.accuracy * 100).toFixed(2)),
        backgroundColor: 'rgba(54, 162, 235, 0.7)'
      }]
    },
    options: {
      indexAxis: 'y',
      scales: { x: { min: 0, max: 100, title: { display: true, text: 'Accuracy %' } } },
      plugins: { legend: { display: false } }
    }
  });
}
