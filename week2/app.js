// Global state
let trainData, testData, model;

// Utility: Enable button
function enable(id) { document.getElementById(id).disabled = false; }

// Data Load
document.getElementById('trainFile').addEventListener('change', e => {
  Papa.parse(e.target.files[0], {
    header: true,
    dynamicTyping: true,
    complete: results => {
      trainData = results.data;
      enable('previewBtn');
    }
  });
});

document.getElementById('testFile').addEventListener('change', e => {
  Papa.parse(e.target.files[0], {
    header: true,
    dynamicTyping: true,
    complete: results => {
      testData = results.data;
    }
  });
});

// Preview Data
document.getElementById('previewBtn').addEventListener('click', () => {
  const previewDiv = document.getElementById('dataPreview');
  previewDiv.innerHTML = "<pre>" + JSON.stringify(trainData.slice(0, 5), null, 2) + "</pre>";
  enable('preprocessBtn');
});

// Preprocess
document.getElementById('preprocessBtn').addEventListener('click', () => {
  // TODO: implement preprocessing (impute, encode, standardize)
  alert("Preprocessing placeholder done!");
  enable('buildModelBtn');
});

// Build Model
document.getElementById('buildModelBtn').addEventListener('click', () => {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [10] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  model.summary();
  enable('trainBtn');
});

// Train
document.getElementById('trainBtn').addEventListener('click', async () => {
  const epochs = parseInt(document.getElementById('epochs').value);
  const batchSize = parseInt(document.getElementById('batchSize').value);

  // TODO: use processed tensors
  const xs = tf.randomNormal([100, 10]);
  const ys = tf.randomUniform([100, 1]).round();

  const history = await model.fit(xs, ys, {
    epochs, batchSize,
    validationSplit: 0.2,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'acc'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });

  enable('evalBtn');
});

// Evaluate
document.getElementById('evalBtn').addEventListener('click', () => {
  document.getElementById('thresholdSlider').disabled = false;
  enable('predictBtn');
});

// Predict + Export
document.getElementById('predictBtn').addEventListener('click', () => {
  // TODO: run prediction
  document.getElementById('predictionPreview').innerHTML = "Predictions ready!";
  enable('exportBtn');
});

document.getElementById('exportBtn').addEventListener('click', () => {
  // TODO: create submission.csv and trigger download
  alert("Export CSV placeholder!");
});
