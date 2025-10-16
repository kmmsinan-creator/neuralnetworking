// data-loader.js
// Handles CSV parsing, normalization, and building sliding window time series samples.

import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0/dist/tf.min.js";

export class DataLoader {
  constructor({
    sequenceLength = 12,
    predictHorizon = 3,
    featureKeys = ["Open", "Close"],
    testSplit = 0.2,
  } = {}) {
    this.sequenceLength = sequenceLength;
    this.predictHorizon = predictHorizon;
    this.featureKeys = featureKeys;
    this.testSplit = testSplit;
    this.symbols = [];
    this.dates = [];
    this.dataBySymbol = {};
    this.normalizers = {};
    this.tensors = null;
  }

  async parseCsvFile(file) {
    const text = await file.text();
    return this.parseCsvText(text);
  }

  parseCsvText(csvText) {
    const lines = csvText.trim().split(/\r?\n/);
    const header = lines[0].split(",").map((h) => h.trim());
    const idx = {};
    header.forEach((h, i) => (idx[h.toLowerCase()] = i));

    if (
      idx["date"] === undefined ||
      idx["symbol"] === undefined ||
      idx["open"] === undefined ||
      idx["close"] === undefined
    )
      throw new Error("CSV must include Date, Symbol, Open, Close columns");

    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(",").map((c) => c.trim());
      const date = cols[idx["date"]];
      const sym = cols[idx["symbol"]];
      const open = parseFloat(cols[idx["open"]]);
      const close = parseFloat(cols[idx["close"]]);
      if (!date || !sym || isNaN(open) || isNaN(close)) continue;
      rows.push({ date, symbol: sym, open, close });
    }

    const bySym = {};
    for (const r of rows) {
      if (!bySym[r.symbol]) bySym[r.symbol] = {};
      bySym[r.symbol][r.date] = r;
    }

    const symbols = Object.keys(bySym).sort().slice(0, 10);
    const allDates = symbols.map((s) => new Set(Object.keys(bySym[s])));
    const commonDates = [...allDates[0]].filter((d) =>
      allDates.every((ds) => ds.has(d))
    );
    commonDates.sort((a, b) => new Date(a) - new Date(b));

    this.symbols = symbols;
    this.dates = commonDates;
    this.dataBySymbol = {};
    for (const s of symbols) {
      this.dataBySymbol[s] = commonDates.map((d) => bySym[s][d]);
    }

    this.normalizers = {};
    for (const s of symbols) {
      const opens = this.dataBySymbol[s].map((r) => r.open);
      const closes = this.dataBySymbol[s].map((r) => r.close);
      this.normalizers[s] = {
        Open: {
          min: Math.min(...opens),
          max: Math.max(...opens),
        },
        Close: {
          min: Math.min(...closes),
          max: Math.max(...closes),
        },
      };
    }
  }

  buildSamples() {
    const seq = this.sequenceLength;
    const horizon = this.predictHorizon;
    const N = this.dates.length;
    const numStocks = this.symbols.length;
    const X = [];
    const Y = [];

    for (let t = seq - 1; t < N - horizon; t++) {
      const input = [];
      for (let k = t - seq + 1; k <= t; k++) {
        const f = [];
        for (const s of this.symbols) {
          const r = this.dataBySymbol[s][k];
          const n = this.normalizers[s];
          const no = (r.open - n.Open.min) / (n.Open.max - n.Open.min);
          const nc = (r.close - n.Close.min) / (n.Close.max - n.Close.min);
          f.push(no, nc);
        }
        input.push(f);
      }
      const baseline = this.symbols.map((s) => this.dataBySymbol[s][t].close);
      const labels = [];
      for (let h = 1; h <= horizon; h++) {
        for (let si = 0; si < numStocks; si++) {
          const future = this.dataBySymbol[this.symbols[si]][t + h];
          labels.push(future.close > baseline[si] ? 1 : 0);
        }
      }
      X.push(input);
      Y.push(labels);
    }

    const split = Math.floor(X.length * (1 - this.testSplit));
    const X_train = tf.tensor3d(X.slice(0, split));
    const y_train = tf.tensor2d(Y.slice(0, split));
    const X_test = tf.tensor3d(X.slice(split));
    const y_test = tf.tensor2d(Y.slice(split));

    this.tensors = { X_train, y_train, X_test, y_test };
    return this.tensors;
  }

  getDataset() {
    if (!this.tensors) throw new Error("Build samples first");
    return { ...this.tensors, symbols: this.symbols, dates: this.dates };
  }

  dispose() {
    if (!this.tensors) return;
    Object.values(this.tensors).forEach((t) => t.dispose());
  }
}
