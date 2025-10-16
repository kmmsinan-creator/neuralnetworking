// gru.js — FINAL FIX (No tf.sequential, No Conv1D, Works 100% in browser)
// CNN removed. Uses GRU layers only. Compatible with tf.js CDN + ES modules.

import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.esm.min.js';

export default class GRUModel {
  constructor({ seqLen, numFeatures, nSymbols, horizon, learningRate = 0.001 }) {
    this.seqLen = seqLen;
    this.numFeatures = numFeatures;
    this.nSymbols = nSymbols;
    this.horizon = horizon;
    this.learningRate = learningRate;
    this.model = null;
  }

  build() {
    // ✅ Functional model (works even if tf.sequential() is unavailable)
    const inputs = tf.input({ shape: [this.seqLen, this.numFeatures] });

    const gru1 = tf.layers.gru({
      units: 64,
      returnSequences: true,
      activation: 'tanh',
      recurrentActivation: 'sigmoid'
    }).apply(inputs);

    const gru2 = tf.layers.gru({
      units: 32,
      returnSequences: false,
      activation: 'tanh',
      recurrentActivation: 'sigmoid'
    }).apply(gru1);

    const dense1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(gru2);
    const drop1 = tf.layers.dropout({ rate: 0.3 }).apply(dense1);

    const outputUnits = this.nSymbols * this.horizon;
    const outputs = tf.layers.dense({ units: outputUnits, activation: 'sigmoid' }).apply(drop1);

    this.model = tf.model({ inputs, outputs });
    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['binaryAccuracy']
    });

    console.log('✅ GRU model built successfully.');
    try { this.model.summary(); } catch (e) { /* ignore console summary */ }
  }

  async train(X_train, y_train, { epochs = 10, batchSize = 32, onEpoch } = {}) {
    if (!this.model) throw new Error('Model not built.');

    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      shuffle: true,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (onEpoch) onEpoch(epoch, logs);
          await tf.nextFrame();
        }
      }
    });
  }

  async evaluate(X_test, y_test, symbols, horizon) {
    if (!this.model) throw new Error('Model not built.');

    const preds = this.model.predict(X_test);
    const [yTrue, yPred] = await Promise.all([y_test.array(), preds.array()]);
    preds.dispose();

    const nStocks = symbols.length;
    const results = [];
    const perStock = Array.from({ length: nStocks }, () => ({ correct: 0, total: 0 }));

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

    return results;
  }

  dispose() {
    if (this.model) this.model.dispose();
  }
}
