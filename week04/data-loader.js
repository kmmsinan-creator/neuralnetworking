// data-loader.js
// Parse CSV, enrich features (Open, Close, scaled return, 2-day momentum, short volatility),
// normalize per-stock per-feature, build sliding-window samples, and export tensors.
// Uses global tf (tf.min.js loaded in index.html). No ES module imports to match existing design.

class DataLoader {
  constructor({ sequenceLength = 12, predictHorizon = 3, testSplit = 0.2 } = {}) {
    this.sequenceLength = sequenceLength;
    this.predictHorizon = predictHorizon;
    this.testSplit = testSplit;
    this.symbols = [];
    this.dates = [];
    this.dataBySymbol = {};
    this.normalizers = {};
    this.tensors = null;
  }

  // Accepts a File object from an <input type="file">
  async parseCsvFile(file) {
    if (!file) throw new Error("No file provided");
    const text = await file.text();
    this.parseCsvText(text);
  }

  // Parse CSV text and pivot into symbol-aligned arrays
  parseCsvText(csvText) {
    if (!csvText || typeof csvText !== "string") throw new Error("CSV text is empty or invalid");
    const lines = csvText.trim().split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length < 2) throw new Error("CSV must include header and at least one data row");

    const header = lines[0].split(",").map(h => h.trim());
    const hmap = {};
    header.forEach((h, i) => { hmap[h.toLowerCase()] = i; });

    if (hmap["date"] === undefined || hmap["symbol"] === undefined || hmap["open"] === undefined || hmap["close"] === undefined) {
      throw new Error("CSV header must include Date, Symbol, Open, Close columns (case-insensitive)");
    }

    // Group rows by symbol and date -> { symbol: {date: {Open,Close}} }
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
    if (symbols.length === 0) throw new Error("No symbols found in CSV");

    // limit to first 10 symbols if more exist
    this.symbols = symbols.slice(0, 10);

    // compute intersection of dates present across chosen symbols
    const dateSets = this.symbols.map(s => new Set(Object.keys(rowsBySymbol[s])));
    const commonDates = [...dateSets[0]].filter(d => dateSets.every(ds => ds.has(d)));
    if (commonDates.length === 0) throw new Error("No common dates across selected symbols; ensure CSV has matching dates for each symbol");
    // sort dates chronologically
    commonDates.sort((a, b) => new Date(a) - new Date(b));
    this.dates = commonDates;

    // build aligned arrays for each symbol
    this.dataBySymbol = {};
    for (const s of this.symbols) {
      const arr = [];
      for (const d of this.dates) {
        const v = rowsBySymbol[s][d];
        if (!v) throw new Error(`Missing date ${d} for symbol ${s}`);
        arr.push({ Date: d, Open: v.Open, Close: v.Close });
      }
      this.dataBySymbol[s] = arr;
    }

    // compute per-symbol per-feature min/max for normalization (features: Open, Close, return, mom2, vol)
    // We'll compute feature-wise stats after computing derived features in buildSamples(), but
    // precompute basic min/max for Open/Close for numeric stability as well.
    this.normalizers = {};
    for (const s of this.symbols) {
      const opens = this.dataBySymbol[s].map(r => r.Open);
      const closes = this.dataBySymbol[s].map(r => r.Close);
      const oMin = Math.min(...opens), oMax = Math.max(...opens);
      const cMin = Math.min(...closes), cMax = Math.max(...closes);
      this.normalizers[s] = { oMin, oMax: oMax === oMin ? oMin + 1e-6 : oMax, cMin, cMax: cMax === cMin ? cMin + 1e-6 : cMax };
    }
  }

  // Build sliding-window samples
  // Features per stock: [normOpen, normClose, scaledReturn, 2dayMomentum, shortVolatility] => 5 features
  // Input shape: [samples, sequenceLength, symbols.length * 5]
  // Output: flattened binary labels of length symbols.length * predictHorizon
  buildSamples() {
    if (!this.symbols || this.symbols.length === 0) throw new Error("No data loaded; call parseCsvFile first");
    const seq = this.sequenceLength;
    const H = this.predictHorizon;
    const S = this.symbols.length;
    const featuresPerStock = 5;
    const inputDim = S * featuresPerStock;
    const N = this.dates.length;

    const X = [];
    const Y = [];

    for (let t = seq - 1; t <= N - 1 - H; t++) {
      // build input window for days [t-seq+1 .. t]
      const window = [];
      let bad = false;
      for (let k = t - seq + 1; k <= t; k++) {
        const dayFeat = [];
        for (const s of this.symbols) {
          const row = this.dataBySymbol[s][k];
          if (!row) { bad = true; break; }
          const n = this.normalizers[s];
          const normOpen = (row.Open - n.oMin) / (n.oMax - n.oMin + 1e-9);
          const normClose = (row.Close - n.cMin) / (n.cMax - n.cMin + 1e-9);
          const prevClose = k > 0 ? this.dataBySymbol[s][k - 1].Close : row.Close;
          const prev2Close = k > 1 ? this.dataBySymbol[s][k - 2].Close : prevClose;
          const ret = (row.Close - prevClose) / (prevClose + 1e-9); // 1-day return
          const mom2 = (row.Close - prev2Close) / (prev2Close + 1e-9); // 2-day momentum
          const vol = Math.abs(ret - mom2); // simple short volatility signal
          // scale returns to reasonable range (-1..1) by clamping after scaling
          const rScaled = Math.max(-1, Math.min(1, ret * 5));
          const momScaled = Math.max(-1, Math.min(1, mom2 * 5));
          const volScaled = Math.max(0, Math.min(2, vol * 5)); // vol >=0
          dayFeat.push(normOpen, normClose, rScaled, momScaled, volScaled);
        }
        if (bad) break;
        window.push(dayFeat);
      }
      if (bad || window.length !== seq) continue;

      // baseline close at day D = t
      const baselineCloses = this.symbols.map(s => this.dataBySymbol[s][t].Close);

      // labels: for offsets 1..H, for each stock
      const labels = [];
      for (let offset = 1; offset <= H; offset++) {
        const di = t + offset;
        for (let si = 0; si < S; si++) {
          const futureRow = this.dataBySymbol[this.symbols[si]][di];
          if (!futureRow) { bad = true; break; }
          labels.push(futureRow.Close > baselineCloses[si] ? 1 : 0);
        }
        if (bad) break;
      }
      if (bad || labels.length !== S * H) continue;

      X.push(window);
      Y.push(labels);
    }

    if (X.length === 0) throw new Error("No samples built. Not enough data for sequence + horizon.");

    // Chronological split: first portion train, last portion test
    const splitIndex = Math.floor(X.length * (1 - this.testSplit));
    const X_train_arr = X.slice(0, splitIndex);
    const y_train_arr = Y.slice(0, splitIndex);
    const X_test_arr = X.slice(splitIndex);
    const y_test_arr = Y.slice(splitIndex);

    // Create tensors using global tf
    if (typeof tf === "undefined" || typeof tf.tensor3d !== "function") {
      throw new Error("TensorFlow.js is not loaded or not available in global scope (tf.tensor3d missing). Ensure tf.min.js is included in index.html before scripts.");
    }

    // Convert arrays to tensors
    const X_train = tf.tensor3d(X_train_arr);
    const y_train = tf.tensor2d(y_train_arr);
    const X_test = tf.tensor3d(X_test_arr);
    const y_test = tf.tensor2d(y_test_arr);

    // store for disposal later
    this.tensors = { X_train, y_train, X_test, y_test };
    return { X_train, y_train, X_test, y_test, symbols: this.symbols };
  }

  dispose() {
    if (this.tensors) {
      for (const k of Object.keys(this.tensors)) {
        try { this.tensors[k].dispose(); } catch (e) { /* ignore */ }
      }
      this.tensors = null;
    }
  }
}

// Expose globally so non-module app.js can construct it
window.DataLoader = DataLoader;
