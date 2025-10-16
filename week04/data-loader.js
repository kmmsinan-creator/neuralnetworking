// data-loader.js
// Handles CSV parsing, normalization, and dataset creation for CNN-based stock prediction.

export default class DataLoader {
  constructor({ seqLen = 12, forecastHorizon = 3, testSplitPct = 0.2 } = {}) {
    this.seqLen = seqLen;
    this.forecastHorizon = forecastHorizon;
    this.testSplitPct = testSplitPct;
  }

  async loadFile(file) {
    const text = await file.text();
    const rows = this.parseCSV(text);
    if (!rows || rows.length === 0) throw new Error("CSV file is empty or invalid.");
    this.prepareData(rows);
  }

  parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const headers = lines[0].split(',').map(h => h.trim());
    const requiredHeaders = ["Date", "Symbol", "Open", "Close"];
    for (const h of requiredHeaders) {
      if (!headers.includes(h)) throw new Error(`Missing column: ${h}`);
    }

    const data = [];
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',');
      if (cols.length < headers.length) continue;
      const obj = {};
      headers.forEach((h, idx) => obj[h] = cols[idx]);
      data.push(obj);
    }
    return data;
  }

  prepareData(rows) {
    const symbols = [...new Set(rows.map(r => r.Symbol))].sort();
    const dates = [...new Set(rows.map(r => r.Date))].sort((a, b) => new Date(a) - new Date(b));

    const symbolData = {};
    for (const s of symbols) {
      symbolData[s] = {};
      rows.filter(r => r.Symbol === s).forEach(r => {
        symbolData[s][r.Date] = { open: +r.Open, close: +r.Close };
      });
    }

    // Normalize each stock individually
    const norm = {};
    for (const s of symbols) {
      const vals = Object.values(symbolData[s]);
      const all = vals.flatMap(v => [v.open, v.close]);
      const min = Math.min(...all);
      const max = Math.max(...all);
      norm[s] = { min, max };
      for (const d in symbolData[s]) {
        const v = symbolData[s][d];
        symbolData[s][d] = {
          open: (v.open - min) / (max - min + 1e-9),
          close: (v.close - min) / (max - min + 1e-9)
        };
      }
    }

    const X = [];
    const Y = [];
    const seqLen = this.seqLen;
    const horizon = this.forecastHorizon;

    for (let i = seqLen - 1; i < dates.length - horizon; i++) {
      const seqDates = dates.slice(i - seqLen + 1, i + 1);
      const xSeq = seqDates.map(d => {
        const features = [];
        for (const s of symbols) {
          const v = symbolData[s][d];
          if (!v) return null;
          features.push(v.open, v.close);
        }
        return features;
      });
      if (xSeq.includes(null)) continue;

      const ySeq = [];
      for (const s of symbols) {
        const base = symbolData[s][dates[i]].close;
        for (let h = 1; h <= horizon; h++) {
          const next = symbolData[s][dates[i + h]]?.close;
          ySeq.push(next > base ? 1 : 0);
        }
      }

      X.push(xSeq);
      Y.push(ySeq);
    }

    const split = Math.floor(X.length * (1 - this.testSplitPct));
    this.symbols = symbols;
    this.X_train = tf.tensor(X.slice(0, split));
    this.y_train = tf.tensor(Y.slice(0, split));
    this.X_test = tf.tensor(X.slice(split));
    this.y_test = tf.tensor(Y.slice(split));
  }

  getTensors() {
    return {
      X_train: this.X_train,
      y_train: this.y_train,
      X_test: this.X_test,
      y_test: this.y_test,
      symbols: this.symbols,
      seqLen: this.seqLen,
      forecastHorizon: this.forecastHorizon
    };
  }
}
