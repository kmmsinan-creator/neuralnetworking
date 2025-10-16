// gru.js
// ES Module: defines GRUModel class (build, fit, predict, computePerStockAccuracy).
// Uses TF.js ESM import to ensure tf.* functions exist.

import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0/dist/tf.esm.js";

export class GRUModel {
  constructor({ inputShape = [12, 20], gruUnits = [64, 32], denseUnits = 30, learningRate = 0.001 } = {}) {
    this.inputShape = inputShape;
    this.gruUnits = gruUnits;
    this.denseUnits = denseUnits;
    this.learningRate = learningRate;
    this.model = null;
  }

  build() {
    const inp = tf.input({ shape: this.inputShape });
    let x = inp;
    for (let i = 0; i < this.gruUnits.length; i++) {
      const returnSeq = i < this.gruUnits.length - 1;
      x = tf.layers.gru({ units: this.gruUnits[i], returnSequences: returnSeq, recurrentInitializer: "glorotUniform" }).apply(x);
    }
    x = tf.layers.dense({ units: Math.max(32, Math.floor(this.gruUnits[this.gruUnits.length - 1] / 2)), activation: "relu" }).apply(x);
    const out = tf.layers.dense({ units: this.denseUnits, activation: "sigmoid" }).apply(x);

    this.model = tf.model({ inputs: inp, outputs: out });
    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "binaryCrossentropy",
      metrics: ["binaryAccuracy"],
    });
    return this.model;
  }

  async fit(X_train, y_train, { epochs = 20, batchSize = 32, validationSplit = 0.1, shuffle = false, onEpoch = null } = {}) {
    if (!this.model) this.build();
    const history = await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationSplit,
      shuffle,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (onEpoch) {
            try { onEpoch(epoch, logs); } catch (e) { console.warn("onEpoch callback error", e); }
          }
          await tf.nextFrame();
        },
      },
    });
    return history;
  }

  predict(X) {
    if (!this.model) throw new Error("Model not built");
    return tf.tidy(() => {
      const p = this.model.predict(X);
      return p.clone(); // caller must dispose
    });
  }

  // predsTensor: [samples, S*H], yTensor: [samples, S*H], symbols: [S]
  async computePerStockAccuracy(predsTensor, yTensor, symbols) {
    const preds = await predsTensor.array();
    const ys = await yTensor.array();
    const samples = preds.length;
    if (samples === 0) return { accuracies: symbols.map(s => ({ symbol: s, accuracy: 0 })), horizon: 0 };
    const totalOut = preds[0].length;
    const S = symbols.length;
    if (totalOut % S !== 0) throw new Error("Output size not divisible by number of symbols");
    const H = totalOut / S;

    const perStock = symbols.map(() => ({ correct: 0, total: 0, timeline: [] }));
    for (let i = 0; i < samples; i++) {
      for (let s = 0; s < S; s++) {
        for (let h = 0; h < H; h++) {
          const idx = s + h * S; // day1 all stocks, day2 all stocks...
          const p = preds[i][idx] >= 0.5 ? 1 : 0;
          const t = ys[i][idx] >= 0.5 ? 1 : 0;
          if (p === t) perStock[s].correct += 1;
          perStock[s].total += 1;
          perStock[s].timeline.push(p === t ? 1 : 0);
        }
      }
    }
    const accuracies = symbols.map((sym, i) => ({
      symbol: sym,
      accuracy: perStock[i].total > 0 ? perStock[i].correct / perStock[i].total : 0,
      timeline: perStock[i].timeline,
    }));
    return { accuracies, horizon: H };
  }

  async saveToLocalStorage(name = "gru-stock-demo") {
    if (!this.model) throw new Error("No model to save");
    return await this.model.save(`localstorage://${name}`);
  }

  async loadFromLocalStorage(name = "gru-stock-demo") {
    this.model = await tf.loadLayersModel(`localstorage://${name}`);
    return this.model;
  }

  dispose() {
    if (this.model) {
      try { this.model.dispose(); } catch (e) { /* ignore */ }
      this.model = null;
    }
  }
}
