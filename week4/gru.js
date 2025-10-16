// gru.js
// ES6 module: defines GRUModel class using TensorFlow.js. Builds, trains, predicts, evaluates.
// Exports GRUModel class.

import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0/dist/tf.min.js";

export class GRUModel {
  constructor({
    inputShape = [12, 20],
    gruUnits = [64, 32],
    denseUnits = 30, // 10 stocks * 3 days = 30
    learningRate = 0.001,
  } = {}) {
    this.inputShape = inputShape;
    this.gruUnits = gruUnits;
    this.denseUnits = denseUnits;
    this.learningRate = learningRate;
    this.model = null;
  }

  build() {
    // Functional API to allow named input
    const input = tf.input({ shape: this.inputShape });
    // stack GRU layers
    let x = input;
    for (let i = 0; i < this.gruUnits.length; i++) {
      const returnSeq = i < this.gruUnits.length - 1 ? true : false;
      const gru = tf.layers.gru({
        units: this.gruUnits[i],
        returnSequences: returnSeq,
        recurrentInitializer: "glorotUniform",
      });
      x = gru.apply(x);
    }
    // Optional dropout/dense
    x = tf.layers.dense({ units: Math.max(32, this.gruUnits[this.gruUnits.length - 1] / 2), activation: "relu" }).apply(x);
    // final output
    const output = tf.layers.dense({ units: this.denseUnits, activation: "sigmoid" }).apply(x);
    this.model = tf.model({ inputs: input, outputs: output });
    const opt = tf.train.adam(this.learningRate);
    this.model.compile({
      optimizer: opt,
      loss: "binaryCrossentropy",
      metrics: ["binaryAccuracy"],
    });
    return this.model;
  }

  // fit with callbacks for progress updates (on epoch end)
  async fit(X_train, y_train, {
    epochs = 20,
    batchSize = 32,
    validationSplit = 0.1,
    shuffle = false,
    onEpoch = null, // function(epoch, logs) for UI updates
  } = {}) {
    if (!this.model) this.build();
    // tfjs fit
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

  // Predict (returns Tensor)
  predict(X) {
    if (!this.model) throw new Error("Model not built");
    return tf.tidy(() => {
      const preds = this.model.predict(X);
      return preds.clone(); // return cloned tensor (caller should dispose)
    });
  }

  // Evaluate using tfjs evaluate (gives loss & binaryAccuracy across all outputs)
  async evaluate(X, y) {
    if (!this.model) throw new Error("Model not built");
    const evals = await this.model.evaluate(X, y, { batchSize: 32 });
    // model.evaluate may return scalar or array of scalars depending on metrics
    const results = Array.isArray(evals) ? evals.map(r => r.dataSync()[0]) : [evals.dataSync()[0]];
    return results;
  }

  // Compute per-stock accuracy from predicted probs and true labels (both arrays or tensors)
  // predsTensor: [samples, 30]; yTensor: [samples, 30]; symbols: array length S; horizon H inferred
  async computePerStockAccuracy(predsTensor, yTensor, symbols) {
    const preds = await predsTensor.array();
    const ys = await yTensor.array();
    const samples = preds.length;
    const totalOutputs = preds[0].length;
    const S = symbols.length;
    const H = totalOutputs / S;
    if (!Number.isInteger(H)) throw new Error("Output size not divisible by number of symbols");
    // threshold at 0.5
    const perStock = symbols.map(() => ({ correct: 0, total: 0, confusion: { tp: 0, tn: 0, fp: 0, fn: 0 }, timeline: [] }));
    for (let i = 0; i < samples; i++) {
      const pRow = preds[i];
      const yRow = ys[i];
      for (let s = 0; s < S; s++) {
        let correctCount = 0;
        let totalCount = 0;
        for (let h = 0; h < H; h++) {
          const idx = s + h * S; // stored as day1 all stocks, day2 all stocks...
          const p = pRow[idx] >= 0.5 ? 1 : 0;
          const t = yRow[idx] >= 0.5 ? 1 : 0;
          totalCount += 1;
          if (p === t) perStock[s].correct += 1;
          // confusion
          if (p === 1 && t === 1) perStock[s].confusion.tp += 1;
          else if (p === 0 && t === 0) perStock[s].confusion.tn += 1;
          else if (p === 1 && t === 0) perStock[s].confusion.fp += 1;
          else if (p === 0 && t === 1) perStock[s].confusion.fn += 1;
          // for timeline store boolean correctness for day h (we'll compress across days if desired)
          perStock[s].timeline.push(p === t ? 1 : 0); // timeline length = samples * H, we will slice per sample if needed
          perStock[s].total += 1;
        }
      }
    }
    // compute accuracies averaged across all sample-days per stock
    const accuracies = symbols.map((s, i) => {
      const obj = perStock[i];
      return { symbol: s, accuracy: obj.total > 0 ? obj.correct / obj.total : 0, confusion: obj.confusion, timeline: obj.timeline };
    });
    return { accuracies, horizon: H, perStockRaw: perStock };
  }

  // Save and load weights via localstorage (or IndexedDB)
  async saveToLocalStorage(name = "gru-model") {
    if (!this.model) throw new Error("No model to save");
    return await this.model.save(`localstorage://${name}`);
  }

  async loadFromLocalStorage(name = "gru-model") {
    this.model = await tf.loadLayersModel(`localstorage://${name}`);
    return this.model;
  }

  // Dispose model to release memory
  dispose() {
    if (this.model) {
      try { this.model.dispose(); } catch (e) { console.warn("model.dispose failed", e); }
      this.model = null;
    }
  }
}
