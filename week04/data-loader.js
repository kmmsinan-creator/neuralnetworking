// data-loader.js
// Handles CSV parsing and sequence dataset creation
class DataLoader {
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
  }

  async parseCsvFile(file) {
    const text = await file.text();
    this.parseCsvText(text);
  }

  parseCsvText(csvText) {
    const lines = csvText.trim().split(/\r?\n/);
    const headers = lines[0].split(",").map(h => h.trim().toLowerCase());
    const idx = {
      date: headers.indexOf("date"),
      symbol: headers.indexOf("symbol"),
      open: headers.indexOf("open"),
      close: headers.indexOf("close"),
    };
    if (Object.values(idx).some(v => v < 0))
      throw new Error("CSV must have Date, Symbol, Open, Close columns.");

    const rows = lines.slice(1).map(l => l.split(","));
    const data = {};
    for (const r of rows) {
      const date = r[idx.date];
      const sym = r[idx.symbol];
      const open = parseFloat(r[idx.open]);
      const close = parseFloat(r[idx.close]);
      if (!data[sym]) data[sym] = [];
      data[sym].push({ date, open, close });
    }

    Object.keys(data).forEach(s => {
      data[s].sort((a, b) => new Date(a.date) - new Date(b.date));
    });

    this.symbols = Object.keys(data).slice(0, 10);
    this.dates = data[this.symbols[0]].map(d => d.date);
    this.dataBySymbol = data;

    this.normalizers = {};
    for (const s of this.symbols) {
      const opens = data[s].map(r => r.open);
      const closes = data[s].map(r => r.close);
      this.normalizers[s] = {
        openMin: Math.min(...opens),
        openMax: Math.max(...opens),
        closeMin: Math.min(...closes),
        closeMax: Math.max(...closes),
      };
    }
  }

  buildSamples() {
    const seq = this.sequenceLength;
    const horizon = this.predictHorizon;
    const stocks = this.symbols;
    const N = this.dates.length;
    const X = [];
    const Y = [];

    for (let t = seq - 1; t < N - horizon; t++) {
      const input = [];
      for (let k = t - seq + 1; k <= t; k++) {
        const feats = [];
        for (const s of stocks) {
          const { open, close } = this.dataBySymbol[s][k];
          const n = this.normalizers[s];
          const oNorm = (open - n.openMin) / (n.openMax - n.openMin + 1e-6);
          const cNorm = (close - n.closeMin) / (n.closeMax - n.closeMin + 1e-6);
          feats.push(oNorm, cNorm);
        }
        input.push(feats);
      }

      const baseline = stocks.map(s => this.dataBySymbol[s][t].close);
      const labels = [];
      for (let h = 1; h <= horizon; h++) {
        for (let s = 0; s < stocks.length; s++) {
          const future = this.dataBySymbol[stocks[s]][t + h].close;
          labels.push(future > baseline[s] ? 1 : 0);
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
    return { X_train, y_train, X_test, y_test, symbols: this.symbols };
  }
}
