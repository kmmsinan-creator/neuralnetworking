// data-loader.js
// ES6 module: parses CSV uploaded via <input type="file">, pivots, normalizes, builds sliding-window samples.
// Exports DataLoader class with methods to parse file and return tensors: X_train, y_train, X_test, y_test and symbol list.

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
    this.symbols = []; // ordered symbol list
    this.dates = []; // ordered dates
    this.dataBySymbol = {}; // symbol -> {dates: [], rows: [{Date,Open,Close}]}
    this.normalizers = {}; // symbol -> {min, max} per feature
    this._built = false;
  }

  // Parse uploaded CSV File object. Returns Promise resolving to true/throws error.
  async parseCsvFile(file) {
    if (!file) throw new Error("No file provided");
    const text = await file.text();
    return this.parseCsvText(text);
  }

  // Parse CSV string content
  parseCsvText(csvText) {
    // Basic CSV parse - tolerant of CRLF and extra whitespace
    const lines = csvText.trim().split(/\r?\n/).map(l => l.trim()).filter(l => l);
    if (lines.length < 2) throw new Error("CSV empty or only header");
    const header = lines[0].split(",").map(h => h.trim());
    // expected fields: Date, Symbol, Open, Close (case-insensitive)
    const idx = {};
    header.forEach((h, i) => { idx[h.toLowerCase()] = i; });
    if (idx["date"] === undefined || idx["symbol"] === undefined ||
        idx["open"] === undefined || idx["close"] === undefined) {
      throw new Error("CSV must include Date, Symbol, Open, Close columns");
    }

    // collect rows
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(",").map(c => c.trim());
      if (cols.length < header.length) continue;
      const date = cols[idx["date"]];
      const sym = cols[idx["symbol"]];
      const open = parseFloat(cols[idx["open"]]);
      const close = parseFloat(cols[idx["close"]]);
      if (!date || !sym || Number.isNaN(open) || Number.isNaN(close)) continue;
      rows.push({ date, symbol: sym, open, close });
    }
    if (rows.length === 0) throw new Error("No valid rows found in CSV");

    // Pivot: group by symbol
    const bySym = {};
    for (const r of rows) {
      if (!bySym[r.symbol]) bySym[r.symbol] = {};
      bySym[r.symbol][r.date] = { Date: r.date, Open: r.open, Close: r.close };
    }

    // Find intersection of dates across symbols (only keep dates present for all symbols)
    const symbolNames = Object.keys(bySym).sort();
    if (symbolNames.length === 0) throw new Error("No symbols found");
    const dateSets = symbolNames.map(s => new Set(Object.keys(bySym[s])));
    // intersect
    const commonDates = [...dateSets[0]].filter(d => dateSets.every(ds => ds.has(d)));
    if (commonDates.length === 0) throw new Error("No common dates across symbols; ensure each symbol has matching dates");
    // sort dates chronologically (ISO-like expected)
    commonDates.sort((a, b) => new Date(a) - new Date(b));

    // Keep only top-10 if more (problem says 10 stocks). If exactly 10, keep as is. If >10 keep first 10 sorted.
    const chosenSymbols = symbolNames.slice(0, 10);

    // Build dataBySymbol with arrays aligned to commonDates
    this.symbols = chosenSymbols;
    this.dates = commonDates;
    this.dataBySymbol = {};
    for (const s of this.symbols) {
      const arr = [];
      for (const d of this.dates) {
        const v = bySym[s][d];
        if (!v) {
          // shouldn't happen due to intersection check
          throw new Error(`Missing date ${d} for symbol ${s}`);
        }
        arr.push({ Date: d, Open: v.Open, Close: v.Close });
      }
      this.dataBySymbol[s] = arr;
    }

    // compute per-symbol min/max for normalization (feature-wise)
    this.normalizers = {};
    for (const s of this.symbols) {
      const opens = this.dataBySymbol[s].map(r => r.Open);
      const closes = this.dataBySymbol[s].map(r => r.Close);
      const minOpen = Math.min(...opens);
      const maxOpen = Math.max(...opens);
      const minClose = Math.min(...closes);
      const maxClose = Math.max(...closes);
      // guard against constant series
      this.normalizers[s] = {
        Open: { min: minOpen, max: maxOpen === minOpen ? minOpen + 1e-6 : maxOpen },
        Close: { min: minClose, max: maxClose === minClose ? minClose + 1e-6 : maxClose },
      };
    }
    this._built = false;
    return true;
  }

  // Build sliding-window samples and convert to tensors
  buildSamples() {
    if (!this.symbols || this.symbols.length === 0) throw new Error("No data loaded");
    const N = this.dates.length;
    const seq = this.sequenceLength;
    const horizon = this.predictHorizon;
    const numStocks = this.symbols.length;
    const featuresPerStock = this.featureKeys.length;
    const inputFeatureDim = numStocks * featuresPerStock;

    const X = []; // [samples, seq, inputFeatureDim]
    const Y = []; // [samples, numStocks * horizon] flattened as stock1_day1, stock1_day2,... stockN_dayH

    // for each possible sample: we need seq days up to day t (inclusive), and horizon days after t
    // iterate t index as the index of the last day of input window (D)
    for (let t = seq - 1; t <= N - 1 - horizon; t++) {
      // ensure t+1..t+horizon exist
      const sampleDates = [];
      const inputWindow = [];
      let skip = false;
      // build input: last seq days (t - seq + 1 .. t)
      for (let k = t - seq + 1; k <= t; k++) {
        const dayFeatures = [];
        for (const s of this.symbols) {
          const row = this.dataBySymbol[s][k];
          if (!row) { skip = true; break; }
          // normalize per symbol
          const normOpen = (row.Open - this.normalizers[s].Open.min) / (this.normalizers[s].Open.max - this.normalizers[s].Open.min);
          const normClose = (row.Close - this.normalizers[s].Close.min) / (this.normalizers[s].Close.max - this.normalizers[s].Close.min);
          dayFeatures.push(normOpen, normClose);
        }
        if (skip) break;
        inputWindow.push(dayFeatures); // length inputFeatureDim
      }
      if (skip || inputWindow.length !== seq) continue;

      // baseline closes at day D for each stock
      const baselineCloses = this.symbols.map((s) => {
        return this.dataBySymbol[s][t].Close;
      });

      // build labels flattened
      const labels = [];
      for (let offset = 1; offset <= horizon; offset++) {
        const dayIndex = t + offset;
        for (let si = 0; si < this.symbols.length; si++) {
          const s = this.symbols[si];
          const futureRow = this.dataBySymbol[s][dayIndex];
          if (!futureRow) { skip = true; break; }
          const lab = futureRow.Close > baselineCloses[si] ? 1 : 0;
          labels.push(lab);
        }
        if (skip) break;
      }
      if (skip || labels.length !== numStocks * horizon) continue;

      X.push(inputWindow);
      Y.push(labels);
    }

    if (X.length === 0) throw new Error("No samples built - not enough data for sequence + horizon.");

    // Split chronologically: first (1 - testSplit) fraction for train, last for test
    const splitIndex = Math.floor(X.length * (1 - this.testSplit));
    const X_train = tf.tensor3d(X.slice(0, splitIndex)); // [samples_train, seq, inputFeatureDim]
    const y_train = tf.tensor2d(Y.slice(0, splitIndex)); // [samples_train, numStocks*horizon]
    const X_test = tf.tensor3d(X.slice(splitIndex));
    const y_test = tf.tensor2d(Y.slice(splitIndex));

    // expose some metadata
    this._built = true;
    this.tensors = { X_train, y_train, X_test, y_test };
    return this.tensors;
  }

  // Utility to dispose created tensors
  dispose() {
    if (this.tensors) {
      for (const k of Object.keys(this.tensors)) {
        try { this.tensors[k].dispose(); } catch (e) {}
      }
    }
  }

  // Convenience getter
  getDataset() {
    if (!this._built) throw new Error("Call buildSamples() first");
    return {
      X_train: this.tensors.X_train,
      y_train: this.tensors.y_train,
      X_test: this.tensors.X_test,
      y_test: this.tensors.y_test,
      symbols: this.symbols.slice(),
      dates: this.dates.slice(),
    };
  }
}
