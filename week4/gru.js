// gru.js

class GRUModel {
  constructor({ inputShape = [12, 20], denseUnits = 30, learningRate = 0.001 } = {}) {
    this.inputShape = inputShape;
    this.denseUnits = denseUnits;
    this.learningRate = learningRate;
  }

  build() {
    const input = tf.input({ shape: this.inputShape });
    let x = tf.layers.gru({ units: 64, returnSequences: true }).apply(input);
    x = tf.layers.gru({ units: 32 }).apply(x);
    x = tf.layers.dense({ units: 32, activation: "relu" }).apply(x);
    const output = tf.layers.dense({ units: this.denseUnits, activation: "sigmoid" }).apply(x);

    this.model = tf.model({ inputs: input, outputs: output });
    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "binaryCrossentropy",
      metrics: ["binaryAccuracy"],
    });
  }

  async fit(X_train, y_train, { epochs = 20, batchSize = 32, onEpoch } = {}) {
    if (!this.model) this.build();
    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      shuffle: false,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (onEpoch) onEpoch(epoch, logs);
          await tf.nextFrame();
        },
      },
    });
  }

  predict(X) {
    return this.model.predict(X);
  }

  async computeAccuracy(preds, yTrue, symbols) {
    const p = await preds.array();
    const y = await yTrue.array();
    const S = symbols.length;
    const H = p[0].length / S;
    const acc = symbols.map(() => ({ correct: 0, total: 0 }));

    for (let i = 0; i < p.length; i++) {
      for (let s = 0; s < S; s++) {
        for (let h = 0; h < H; h++) {
          const idx = s + h * S;
          const pred = p[i][idx] >= 0.5 ? 1 : 0;
          const trueVal = y[i][idx];
          if (pred === trueVal) acc[s].correct++;
          acc[s].total++;
        }
      }
    }

    return symbols.map((sym, i) => ({
      symbol: sym,
      accuracy: acc[i].correct / acc[i].total,
    }));
  }
}
