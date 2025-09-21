// Global state
let trainData, testData;
let xTrain, yTrain, xVal, yVal, testTensor, passengerIds;
let model;
let featureNames = [];

// Utils
const enable = id => document.getElementById(id).disabled = false;

// ---------- Data Load ----------
document.getElementById('trainFile').addEventListener('change', e => {
  Papa.parse(e.target.files[0], {
    header: true,
    dynamicTyping: true,
    complete: results => {
      trainData = results.data.filter(r => r.Survived !== undefined);
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

// ---------- Preview ----------
document.getElementById('previewBtn').addEventListener('click', () => {
  const previewDiv = document.getElementById('dataPreview');
  previewDiv.innerHTML =
    "<pre>" + JSON.stringify(trainData.slice(0, 5), null, 2) + "</pre>";
  enable('preprocessBtn');
});

// ---------- Preprocess ----------
function preprocessData(data, isTrain = true) {
  // Drop unused cols
  let df = data.map(r => ({
    PassengerId: r.PassengerId,
    Survived: r.Survived,
    Pclass: r.Pclass,
    Sex: r.Sex,
    Age: r.Age,
    SibSp: r.SibSp,
    Parch: r.Parch,
    Fare: r.Fare,
    Embarked: r.Embarked
  }));

  // Impute Age/Fare with median
  const median = (arr) => {
    const nums = arr.filter(x => x !== null && !isNaN(x)).sort((a, b) => a - b);
    const mid = Math.floor(nums.length / 2);
    return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
  };
  const mode = (arr) => {
    const counts = {};
    arr.forEach(x => { if (x) counts[x] = (counts[x] || 0) + 1; });
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
  };

  const ageMed = median(df.map(r => r.Age));
  const fareMed = median(df.map(r => r.Fare));
  const embMode = mode(df.map(r => r.Embarked));

  df.forEach(r => {
    if (!r.Age) r.Age = ageMed;
    if (!r.Fare) r.Fare = fareMed;
    if (!r.Embarked) r.Embarked = embMode;
  });

  // Add FamilySize & IsAlone
  df.forEach(r => {
    r.FamilySize = (r.SibSp || 0) + (r.Parch || 0) + 1;
    r.IsAlone = r.FamilySize === 1 ? 1 : 0;
  });

  // One-hot encode categorical
  const sexVals = ['male', 'female'];
  const pclassVals = [1, 2, 3];
  const embVals = ['C', 'Q', 'S'];

  let processed = df.map(r => {
    let feat = {
      Age: r.Age,
      Fare: r.Fare,
      FamilySize: r.FamilySize,
      IsAlone: r.IsAlone,
    };

    sexVals.forEach(v => feat['Sex_' + v] = r.Sex === v ? 1 : 0);
    pclassVals.forEach(v => feat['Pclass_' + v] = r.Pclass === v ? 1 : 0);
    embVals.forEach(v => feat['Embarked_' + v] = r.Embarked === v ? 1 : 0);

    return {
      PassengerId: r.PassengerId,
      Survived: r.Survived,
      features: feat
    };
  });

  // Standardize Age, Fare
  const meanStd = (arr) => {
    const mean = arr.reduce((a, b) => a + b) / arr.length;
    const std = Math.sqrt(arr.map(x => (x - mean) ** 2).reduce((a, b) => a + b) / arr.length);
    return { mean, std };
  };

  const ageStats = meanStd(processed.map(r => r.features.Age));
  const fareStats = meanStd(processed.map(r => r.features.Fare));

  processed.forEach(r => {
    r.features.Age = (r.features.Age - ageStats.mean) / ageStats.std;
    r.features.Fare = (r.features.Fare - fareStats.mean) / fareStats.std;
  });

  return processed;
}

document.getElementById('preprocessBtn').addEventListener('click', () => {
  let processedTrain = preprocessData(trainData, true);
  let processedTest = preprocessData(testData, false);

  featureNames = Object.keys(processedTrain[0].features);

  // Create tensors
  const xs = tf.tensor2d(processedTrain.map(r => featureNames.map(f => r.features[f])));
  const ys = tf.tensor2d(processedTrain.map(r => [r.Survived]));

  // Train/val split
  const split = Math.floor(xs.shape[0] * 0.8);
  xTrain = xs.slice([0, 0], [split, -1]);
  yTrain = ys.slice([0, 0], [split, -1]);
  xVal = xs.slice([split, 0], [-1, -1]);
  yVal = ys.slice([split, 0], [-1, -1]);

  passengerIds = processedTest.map(r => r.PassengerId);
  testTensor = tf.tensor2d(processedTest.map(r => featureNames.map(f => r.features[f])));

  alert("Preprocessing complete. Features: " + featureNames.join(", "));
  enable('buildModelBtn');
});

// ---------- Build Model ----------
document.getElementById('buildModelBtn').addEventListener('click', () => {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [featureNames.length] }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  model.summary();
  enable('trainBtn');
});

// ---------- Train ----------
document.getElementById('trainBtn').addEventListener('click', async () => {
  const epochs = parseInt(document.getElementById('epochs').value);
  const batchSize = parseInt(document.getElementById('batchSize').value);

  await model.fit(xTrain, yTrain, {
    epochs, batchSize,
    validationData: [xVal, yVal],
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'acc'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });

  enable('evalBtn');
});

// ---------- Evaluate ----------
document.getElementById('evalBtn').addEventListener('click', async () => {
  const preds = model.predict(xVal).arraySync().map(p => p[0]);
  const labels = yVal.arraySync().map(p => p[0]);

  // Default threshold
  updateConfusion(preds, labels, 0.5);

  document.getElementById('thresholdSlider').disabled = false;
  document.getElementById('thresholdSlider').addEventListener('input', e => {
    const t = parseFloat(e.target.value);
    document.getElementById('thresholdValue').innerText = t.toFixed(2);
    updateConfusion(preds, labels, t);
  });

  enable('predictBtn');
});

function updateConfusion(preds, labels, threshold) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  preds.forEach((p, i) => {
    const pred = p >= threshold ? 1 : 0;
    if (pred === 1 && labels[i] === 1) tp++;
    if (pred === 0 && labels[i] === 0) tn++;
    if (pred === 1 && labels[i] === 0) fp++;
    if (pred === 0 && labels[i] === 1) fn++;
  });
  const acc = (tp + tn) / preds.length;

  document.getElementById('confMatrix').innerHTML = `
    <p>TP: ${tp}, FP: ${fp}, TN: ${tn}, FN: ${fn}</p>
    <p>Accuracy: ${(acc*100).toFixed(2)}%</p>
  `;
}

// ---------- Predict & Export ----------
let finalPreds;

document.getElementById('predictBtn').addEventListener('click', () => {
  const preds = model.predict(testTensor).arraySync().map(p => p[0]);
  finalPreds = preds.map(p => p >= 0.5 ? 1 : 0);

  const preview = passengerIds.slice(0, 5).map((id, i) => ({
    PassengerId: id, Survived: finalPreds[i]
  }));
  document.getElementById('predictionPreview').innerHTML =
    "<pre>" + JSON.stringify(preview, null, 2) + "</pre>";

  enable('exportBtn');
});

document.getElementById('exportBtn').addEventListener('click', () => {
  let csv = "PassengerId,Survived\n";
  passengerIds.forEach((id, i) => {
    csv += `${id},${finalPreds[i]}\n`;
  });
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "submission.csv";
  a.click();
  window.URL.revokeObjectURL(url);
});
