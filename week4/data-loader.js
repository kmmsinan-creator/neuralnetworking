// data-loader.js
// ES Module: parse CSV uploaded via file input, pivot, normalize, build sliding-window samples,
// and export tensor datasets. Uses TF.js ESM import to guarantee tf.tensor3d exists.

import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0/dist/tf.esm.js";

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

  // Parse a File object (CSV) uploaded by the user
  async parseCsvFile(file) {
    if (!file) throw new Error("No file provided");
    const text = await file.text();
    return this.parseCsvText(text);
  }

  // Parse CSV text. Expect headers Date, Symbol, Open, Close (case-insensitive)
  parseCsvText(csvText) {
    if (!csvText || typeof csvText !== "string") throw new Error("CSV content is empty or invalid");
    const lines = csvText.trim().split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length < 2) throw new Error("CSV must contain header and at least one data row");

    const header = lines[0].split(",").map(h => h.trim());
    const hmap = {};
    header.forEach((h, i) => { hmap[h.toLowerCase()] = i; });

    if (hmap["date"] === undefined || hmap["symbol"] === undefined || hmap["open"] === undefined || hmap["close"] === undefined) {
      throw new Error("CSV header must include Date, Symbol, Open, Close columns");
    }

    // collect rows grouped by symbol and date
    const rowsBySymbol = {};
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(",").map(c => c.trim());
      if (cols.length < header.length) continue;
      const date = cols[hmap["date"]];
      const symbol = cols[hmap["symbol"]];
      const open = parseFloat(cols[hmap["open"]]);
      const close = parseFloat(cols[hmap["close"]]);
      if (!date || !symbol || Number.isNaN(open) || Number.isNaN(close)) continue;
      if (!rowsBySymbol[symbol]) rowsBySymbol[symbol] = {};
      rowsBySymbol[symbol][date] = { Date: date, Open: open, Close: close };
    }

    const symbols = Object.keys(rowsBySymbol).sort();
    if (symbols.length === 0) throw new Error("No symbols detected in CSV");

    // If more than 10 symbols present, choose first 10 (user said 10 S&P stocks)
    this.symbols = symbols.slice(0, 10);

    // Find intersection of dates present in all chosen symbols
    const dateLists = this.symbols.map(s => Object.keys(rowsBySymbol[s]));
    // compute intersection
    const dateSet = new Set(dateLists[0]);
    for (let i = 1; i < dateLists.length; i++) {
      const s = new Set(dateLists[i]);
      for (const d of Array.from(dateSet)) {
        if (!s.has(d)) dateSet.delete(d);
      }
    }
    if (dateSet.size === 0) throw new Error("No common dates across selected symbols; CSV must align dates for all symbols");

    // sort dates chronologically
    this.dates = Array.from(dateSet).sort((a, b) => new Date(a) - new Date(b));

    // Build aligned arrays for each symbol
    this.dataBySymbol = {};
    for (const s of this.symbols) {
      const arr = [];
      for (const d of this.dates) {
        const entry = rowsBySymbol[s][d];
        if (!entry) throw new Error(`Missing date ${d} for symbol ${s}`);
        arr.push({ Date: entry.Date, Open: entry.Open, Close: entry.Close });
      }
      this.dataBySymbol[s] = arr;
    }

    // Compute per-symbol feature min/max for normalization (Open, Close)
    this.normalizers = {};
    for (const s of this.symbols) {
      const opens = this.dataBySymbol[s].map(r => r.Open);
      const closes = this.dataBySymbol[s].map(r => r.Close);
      const oMin = Math.min(...opens), oMax = Math.max(...opens);
      const cMin = Math.min(...closes), cMax = Math.max(...closes);
      this.normalizers[s] = {
        Open: { min: oMin, max: oMax === oMin ? oMin + 1e-6 : oMax },
        Close: { min: cMin, max: cMax === cMin ? cMin + 1e-6 : cMax },
      };
    }

    return true;
  }

  // Build sliding-window samples and produce tensors
  buildSamples() {
    if (!this.symbols || this.symbols.length === 0) throw new Error("No data loaded. Call parseCsvFile first.");
    const seq = this.sequenceLength;
    const horizon = this.predictHorizon;
    const S = this.symbols.length;
    const featurePerStock = this.featureKeys.length; // 2 (Open, Close)
    const inputDim = S * featurePerStock;
    const N = this.dates.length;

    const X = [];
    const Y = [];

    // iterate over last index of input window t (D)
    for (let t = seq - 1; t <= N - 1 - horizon; t++) {
      // build input window [seq, inputDim]
      const window = [];
      let skip = false;
      for (let k = t - seq + 1; k <= t; k++) {
        const dayFeat = [];
        for (const s of this.symbols) {
          const row = this.dataBySymbol[s][k];
          if (!row) { skip = true; break; }
          const normOpen = (row.Open - this.normalizers[s].Open.min) / (this.normalizers[s].Open.max - this.normalizers[s].Open.min);
          const normClose = (row.Close - this.normalizers[s].Close.min) / (this.normalizers[s].Close.max - this.normalizers[s].Close.min);
          dayFeat.push(normOpen, normClose);
        }
        if (skip) break;
        window.push(dayFeat);
      }
      if (skip || window.length !== seq) continue;

      // baseline close on day D
      const baseline = this.symbols.map((s) => this.dataBySymbol[s][t].Close);
      const labels = [];
      for (let h = 1; h <= horizon; h++) {
        const idx = t + h;
        for (let si = 0; si < this.symbols.length; si++) {
          const future = this.dataBySymbol[this.symbols[si]][idx];
          if (!future) { skip = true; break; }
          labels.push(future.Close > baseline[si] ? 1 : 0);
        }
        if (skip) break;
      }
      if (skip || labels.length !== S * horizon) continue;

      X.push(window);
      Y.push(labels);
    }

    if (X.length === 0) throw new Error("No samples could be built - not enough data for sequence + horizon.");

    // Chronological split
    const splitIndex = Math.floor(X.length * (1 - this.testSplit));
    const X_train_arr = X.slice(0, splitIndex);
    const y_train_arr = Y.slice(0, splitIndex);
    const X_test_arr = X.slice(splitIndex);
    const y_test_arr = Y.slice(splitIndex);

    // Ensure tf functions exist
    if (typeof tf.tensor3d !== "function" || typeof tf.tensor2d !== "function") {
      throw new Error("TensorFlow.js does not appear to be loaded correctly (tf.tensor3d is missing). Ensure you're importing tf.esm.js in each module and not relying on a global tf.");
    }

    // Create tensors
    const X_train = tf.tensor3d(X_train_arr);
    const y_train = tf.tensor2d(y_train_arr);
    const X_test = tf.tensor3d(X_test_arr);
    const y_test = tf.tensor2d(y_test_arr);

    // store
    this.tensors = { X_train, y_train, X_test, y_test };
    return { X_train, y_train, X_test, y_test, symbols: this.symbols };
  }

  // Dispose tensors to free memory
  dispose() {
    if (this.tensors) {
      for (const k of Object.keys(this.tensors)) {
        try { this.tensors[k].dispose(); } catch (e) { /* ignore */ }
      }
      this.tensors = null;
    }
  }
}
