// gru.js (actually CNN-based model using Conv1D in TensorFlow.js)

export default class CNNModel {
  constructor({ seqLen, numFeatures, nSymbols, horizon, learningRate = 0.001 } = {}) {
    this.seqLen = seqLen;
    this.numFeatures = numFeatures;
    this.nSymbols = nSymbols;
    this.horizon = horizon;
    this.outDim = nSymbols * horizon;
    this.learningRate = learningRate;
  }

  build() {
    const input = tf.input({ shape: [this.seqLen, this.numFeatures] });
    let x = tf.layers.conv1d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'causal' }).apply(input);
    x = tf.layers.conv1d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'causal' }).apply(x);
    x = tf.layers.globalAveragePooling1d().apply(x);
    x = tf.layers.dropout({ rate: 0.2 }).apply(x);
    x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);
    const output = tf.layers.dense({ units: this.outDim, activation: 'sigmoid' }).apply(x);

    this.model = tf.model({ inputs: input, outputs: output });
    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });
  }

  async train(X_train, y_train, { epochs = 15, batchSize = 32, onEpoch } = {}) {
    await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationSplit: 0.1,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (onEpoch) onEpoch(epoch, logs);
          await tf.nextFrame();
        }
      }
    });
  }

  async evaluate(X_test, y_test, symbols, horizon) {
    const preds = this.model.predict(X_test);
    const y_pred = await preds.array();
    const y_true = await y_test.array();
    preds.dispose();

    const results = symbols.map(s => ({ symbol: s, correct: 0, total: 0 }));
    y_true.forEach((trueVals, i) => {
      trueVals.forEach((val, j) => {
        const sIdx = Math.floor(j / horizon);
        const pred = y_pred[i][j] > 0.5 ? 1 : 0;
        if (pred === val) results[sIdx].correct++;
        results[sIdx].total++;
      });
    });
    results.forEach(r => (r.accuracy = r.correct / r.total));
    return results;
  }
}
