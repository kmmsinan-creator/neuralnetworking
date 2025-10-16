// gru.js — UPDATED (fixes CAUSAL padding issue in Conv1D)
// CNN-based model for multi-stock binary up/down classification using TensorFlow.js

import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.min.js';

export default class CNNModel {
  constructor({ seqLen, numFeatures, nSymbols, horizon, learningRate = 0.001 }) {
    this.seqLen = seqLen;
    this.numFeatures = numFeatures;
    this.nSymbols = nSymbols;
    this.horizon = horizon;
    this.learningRate = learningRate;
    this.model = null;
  }

  build() {
    const model = tf.sequential();

    // 1️⃣ Conv1D layers (use padding: 'same' to avoid unsupported causal padding)
    model.add(tf.layers.conv1d({
      inputShape: [this.seqLen, this.numFeatures],
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));
    model.add(tf.layers.conv1d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }));

    // 2️⃣ Global pooling + dense layers
    model.add(tf.layers.globalAveragePooling1d());
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.3 }));

    // 3️⃣ Output layer: 10 stocks × 3 days = 30 binary outputs
    const outUnits = this.nSymbols * this.horizon;
    model.add(tf.layers.dense({ units: outUnits, activation: 'sigmoid' }));

    const optimizer = tf.train.adam(this.learningRate);
    model.compile({
      optimizer,
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });

    this.model = model;
    console.log('✅ CNN model built:', model.summary());
  }

  async train(X_train, y_train, { epochs = 10, batchSize = 32, onEpoch }) {
    if (!this.model) throw new Error('Model not built');
    const valSplit = 0.2;
    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationSplit: valSplit,
      shuffle: false,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (onEpoch) onEpoch(epoch, logs);
          await tf.nextFrame();
        }
      }
    });
  }

  async evaluate(X_test, y_test, symbols, horizon) {
    if (!this.model) throw new Error('Model not built');
    const preds = this.model.predict(X_test);
    const yTrue = await y_test.array();
    const yPred = await preds.array();

    const nStocks = symbols.length;
    const results = [];
    const perStock = Array(nStocks).fill(0).map(() => ({ correct: 0, total: 0 }));

    for (let i = 0; i < yTrue.length; i++) {
      for (let s = 0; s < nStocks; s++) {
        for (let h = 0; h < horizon; h++) {
          const idx = s * horizon + h;
          const trueVal = yTrue[i][idx] >= 0.5 ? 1 : 0;
          const predVal = yPred[i][idx] >= 0.5 ? 1 : 0;
          if (trueVal === predVal) perStock[s].correct++;
          perStock[s].total++;
        }
      }
    }

    for (let s = 0; s < nStocks; s++) {
      const acc = perStock[s].correct / perStock[s].total;
      results.push({ symbol: symbols[s], accuracy: acc });
    }

    preds.dispose();
    return results;
  }

  dispose() {
    if (this.model) this.model.dispose();
  }
}
