// gru.js
// Improved GRU architecture: batch normalization, dropout, lr decay, yields to UI.
// Uses global tf. Exposes GRUModel globally.

class GRUModel {
  constructor({ inputShape = [12, 50], outputSize = 30, learningRate = 0.001 } = {}) {
    this.inputShape = inputShape;
    this.outputSize = outputSize;
    this.learningRate = learningRate;
    this.model = null;
  }

  build() {
    const inp = tf.input({ shape: this.inputShape });

    // GRU stack with batch normalization and dropout
    let x = tf.layers.gru({ units: 128, returnSequences: true }).apply(inp);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.dropout({ rate: 0.3 }).apply(x);

    x = tf.layers.gru({ units: 64, returnSequences: false }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.dropout({ rate: 0.3 }).apply(x);

    x = tf.layers.dense({ units: 128, activation: "relu" }).apply(x);
    x = tf.layers.dropout({ rate: 0.2 }).apply(x);

    const out = tf.layers.dense({ units: this.outputSize, activation: "sigmoid" }).apply(x);

    this.model = tf.model({ inputs: inp, outputs: out });
    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "binaryCrossentropy",
      metrics: ["binaryAccuracy"],
    });
  }

  // Train with single-epoch loop to allow tf.nextFrame yielding and manual LR schedule
  async fit(X_train, y_train, { epochs = 80, batchSize = 16, validationSplit = 0.1, onEpoch = null } = {}) {
    if (!this.model) this.build();
    const optimizer = this.model.optimizer;

    // track best val loss for a simple early stop (patience)
    let bestValLoss = Infinity;
    let patience = 10;
    let wait = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const h = await this.model.fit(X_train, y_train, {
        epochs: 1,
        batchSize,
        shuffle: true,
        validationSplit,
        verbose: 0,
      });

      // extract metrics
      const loss = h.history.loss[0];
      const acc = h.history.binaryAccuracy ? h.history.binaryAccuracy[0] : (h.history.acc ? h.history.acc[0] : 0);
      const valLoss = h.history.val_loss ? h.history.val_loss[0] : null;
      const valAcc = h.history.val_binaryAccuracy ? h.history.val_binaryAccuracy[0] : (h.history.val_acc ? h.history.val_acc[0] : null);

      // call UI callback
      if (onEpoch) {
        try { onEpoch(epoch, { loss, binaryAccuracy: acc, val_loss: valLoss, val_binaryAccuracy: valAcc }); } catch (e) { console.warn("onEpoch callback", e); }
      }

      // simple LR decay if validation loss plateaus or accuracy low
      if (valLoss !== null) {
        if (valLoss < bestValLoss - 1e-4) {
          bestValLoss = valLoss;
          wait = 0;
        } else {
          wait++;
        }
        if (wait >= 5) {
          // reduce lr by 0.7
          const oldLR = optimizer.learningRate;
          try {
            // tf.train.Optimizer may store learningRate as scalar or number
            if (typeof oldLR === "object" && oldLR.mul) {
              optimizer.learningRate = oldLR.mul(tf.scalar(0.7));
            } else {
              optimizer.learningRate = oldLR * 0.7;
            }
          } catch (e) {
            try { optimizer.learningRate = optimizer.learningRate * 0.7; } catch (ee) { /* ignore */ }
          }
          wait = 0;
        }
      } else {
        // if no valLoss available, decay every 15 epochs as fallback
        if ((epoch + 1) % 15 === 0) {
          try {
            const oldLR = optimizer.learningRate;
            if (typeof oldLR === "object" && oldLR.mul) {
              optimizer.learningRate = oldLR.mul(tf.scalar(0.8));
            } else {
              optimizer.learningRate = oldLR * 0.8;
            }
          } catch (e) {
            // ignore
          }
        }
      }

      // early stopping based on valLoss improvement
      if (valLoss !== null) {
        if (bestValLoss < Infinity && wait >= patience) {
          console.log(`Early stopping at epoch ${epoch + 1}`);
          break;
        }
      }

      // yield control to UI
      await tf.nextFrame();
    }
  }

  // Predict and optionally smooth preds across adjacent samples (simple moving average)
  predict(X, { smoothing = true, window = 3 } = {}) {
    const raw = this.model.predict(X);
    if (!smoothing) return raw;

    // smoothing implemented in async-friendly way: convert to array, smooth, return tensor
    // Note: caller is responsible for disposing returned tensor
    const arrPromise = raw.array();
    raw.dispose();
    return (async () => {
      const arr = await arrPromise; // [samples, outputs]
      const S = arr.length;
      const O = arr[0].length;
      const smoothed = [];
      for (let j = 0; j < O; j++) {
        // get series for output j across samples
        const series = new Array(S);
        for (let i = 0; i < S; i++) series[i] = arr[i][j];
        // moving average
        const outSeries = new Array(S);
        for (let i = 0; i < S; i++) {
          let sum = 0, cnt = 0;
          const half = Math.floor(window / 2);
          for (let k = i - half; k <= i + half; k++) {
            if (k >= 0 && k < S) { sum += series[k]; cnt++; }
          }
          outSeries[i] = sum / Math.max(1, cnt);
        }
        // write back to smoothed as column j
        for (let i = 0; i < S; i++) {
          if (!smoothed[i]) smoothed[i] = new Array(O);
          smoothed[i][j] = outSeries[i];
        }
      }
      return tf.tensor2d(smoothed);
    })();
  }

  // Compute per-stock accuracy given predictions tensor (or Promise resolving to tensor) and true labels
  // predsTensor: tf.Tensor2D [samples, S*H] OR Promise resolving to same (if smoothing was async)
  async computePerStockAccuracy(predsTensorOrPromise, yTensor, symbols) {
    let predsTensor = predsTensorOrPromise;
    if (predsTensorOrPromise && typeof predsTensorOrPromise.then === "function") {
      predsTensor = await predsTensorOrPromise;
    }
    const preds = await predsTensor.array();
    const ys = await yTensor.array();
    const samples = preds.length;
    if (samples === 0) {
      if (predsTensor && predsTensor.dispose) predsTensor.dispose();
      return { accuracies: symbols.map(s => ({ symbol: s, accuracy: 0 })), horizon: 0 };
    }
    const totalOut = preds[0].length;
    const S = symbols.length;
    if (totalOut % S !== 0) throw new Error("Output size not divisible by number of symbols");
    const H = totalOut / S;

    const perStock = symbols.map(() => ({ correct: 0, total: 0, timeline: [] }));
    for (let i = 0; i < samples; i++) {
      for (let s = 0; s < S; s++) {
        for (let h = 0; h < H; h++) {
          const idx = s + h * S;
          const p = preds[i][idx] >= 0.5 ? 1 : 0;
          const t = ys[i][idx] >= 0.5 ? 1 : 0;
          if (p === t) perStock[s].correct += 1;
          perStock[s].total += 1;
          perStock[s].timeline.push(p === t ? 1 : 0);
        }
      }
    }

    const accuracies = symbols.map((s, i) => ({
      symbol: s,
      accuracy: perStock[i].total > 0 ? perStock[i].correct / perStock[i].total : 0,
      timeline: perStock[i].timeline,
    }));

    // dispose predsTensor if we created it here
    if (predsTensor && predsTensor.dispose) predsTensor.dispose();
    return { accuracies, horizon: H };
  }

  dispose() {
    if (this.model) {
      try { this.model.dispose(); } catch (e) { /* ignore */ }
      this.model = null;
    }
  }
}

window.GRUModel = GRUModel;
