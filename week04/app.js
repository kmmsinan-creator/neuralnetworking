// app.js — orchestrates data loading, training, evaluation, and visualization

import DataLoader from './data-loader.js';
import CNNModel from './gru.js';

let loader, model, tensors;

const fileInput = document.getElementById('csv-file');
const loadBtn = document.getElementById('btn-load');
const trainBtn = document.getElementById('btn-train');
const evalBtn = document.getElementById('btn-eval');
const logDiv = document.getElementById('log');
const chartCanvas = document.getElementById('chart').getContext('2d');

function log(msg) {
  logDiv.innerHTML = `[${new Date().toLocaleTimeString()}] ${msg}<br>` + logDiv.innerHTML;
}

loadBtn.onclick = async () => {
  const file = fileInput.files[0];
  if (!file) return alert('Upload CSV first!');
  loader = new DataLoader({});
  await loader.loadFile(file);
  tensors = loader.getTensors();
  log(`Loaded ${tensors.symbols.length} symbols and ${tensors.X_train.shape[0]} samples`);
};

trainBtn.onclick = async () => {
  const seqLen = tensors.seqLen;
  const numFeatures = tensors.X_train.shape[2];
  const nSymbols = tensors.symbols.length;
  const horizon = tensors.forecastHorizon;
  model = new CNNModel({ seqLen, numFeatures, nSymbols, horizon });
  model.build();
  log('Training...');
  await model.train(tensors.X_train, tensors.y_train, {
    epochs: 10,
    onEpoch: (e, l) => log(`Epoch ${e + 1}: loss=${l.loss.toFixed(4)}`)
  });
  log('Training finished.');
};

evalBtn.onclick = async () => {
  log('Evaluating...');
  const results = await model.evaluate(tensors.X_test, tensors.y_test, tensors.symbols);
  renderChart(results);
  log('Evaluation complete.');
};

function renderChart(results) {
  const sorted = results.sort((a, b) => b.accuracy - a.accuracy);
  new Chart(chartCanvas, {
    type: 'bar',
    data: {
      labels: sorted.map(r => r.symbol),
      datasets: [{
        label: 'Accuracy',
        data: sorted.map(r => r.accuracy * 100)
      }]
    },
    options: { indexAxis: 'y', scales: { x: { min: 0, max: 100 } } }
  });
}
