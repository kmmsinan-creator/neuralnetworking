// data-loader.js
// Handles CSV parsing, normalization, and dataset creation for stock prediction.

export default class DataLoader {
  constructor({ seqLen = 12, forecastHorizon = 3, testSplitPct = 0.2 } = {}) {
    this.seqLen = seqLen;
    this.forecastHorizon = forecastHorizon;
    this.testSplitPct = testSplitPct;
  }

  async loadFile(file) {
    const text = await file.text();
    const rows = this.parseCSV(text);
    this.prepareData(rows);
  }

  parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const headers = lines[0].split(',').map(h => h.trim());
    return lines.slice(1).map(line => {
      const cols = line.split(',');
      const obj = {};
      headers.forEach((h, i) => obj[h] = cols[i]);
      return obj;
    });
  }

  prepareData(rows) {
    const symbols = [...new Set(rows.map(r => r.Symbol))].sort();
    const dates = [...new Set(rows.map(r => r.Date))].sort((a, b) => new Date(a) - new Date(b));

    // Pivot data: per symbol {date -> {open, close}}
    const symbolData = {};
    for (const s of symbols) {
      symbolData[s] = {};
      const subset = rows.filter(r => r.Symbol === s);
      subset.forEach(r => {
        symbolData[s][r.Date] = { open: +r.Open, close: +r.Close };
      });
    }

    // Normalize each symbol individually (min-max)
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
          open: (v.open - min) / (max - min),
          close: (v.close - min) / (max - min),
        };
      }
    }

    const seqLen = this.seqLen;
    const horizon = this.forecastHorizon;
    const X = [];
    const Y = [];

    for (let i = seqLen - 1; i < dates.length - horizon; i++) {
      const seqDates = dates.slice(i - seqLen + 1, i + 1);
      const xRow = [];
      for (const d of seqDates) {
        const f = [];
        for (const s of symbols) {
          const data = symbolData[s][d];
          if (!data) return;
          f.push(data.open, data.close);
        }
        xRow.push(f);
      }

      const yRow = [];
      for (const s of symbols) {
        const base = symbolData[s][dates[i]].close;
        for (let h = 1; h <= horizon; h++) {
          const next = symbolData[s][dates[i + h]]?.close;
          yRow.push(next > base ? 1 : 0);
        }
      }
      X.push(xRow);
      Y.push(yRow);
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
