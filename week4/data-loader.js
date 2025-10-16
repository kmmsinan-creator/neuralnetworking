// data-loader.js
// Prepares time-series tensors from CSV.

class DataLoader {
  constructor({ sequenceLength = 12, predictHorizon = 3, testSplit = 0.2 } = {}) {
    this.sequenceLength = sequenceLength;
    this.predictHorizon = predictHorizon;
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
    if (Object.values(idx).some(v => v < 0)) throw new Error("Missing required columns");

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

    // sort dates
    Object.keys(data).forEach(s => data[s].sort((a,b)=>new Date(a.date)-new Date(b.date)));
    this.symbols = Object.keys(data).slice(0,10);
    this.dates = data[this.symbols[0]].map(d=>d.date);
    this.dataBySymbol = data;

    // normalize
    this.normalizers = {};
    for (const s of this.symbols) {
      const opens = data[s].map(r=>r.open);
      const closes = data[s].map(r=>r.close);
      this.normalizers[s] = {
        oMin: Math.min(...opens),
        oMax: Math.max(...opens),
        cMin: Math.min(...closes),
        cMax: Math.max(...closes)
      };
    }
  }

  buildSamples() {
    const seq = this.sequenceLength, H = this.predictHorizon, S = this.symbols.length;
    const N = this.dates.length;
    const X=[], Y=[];

    for(let t=seq-1; t<N-H; t++){
      const input=[];
      for(let k=t-seq+1;k<=t;k++){
        const feats=[];
        for(const s of this.symbols){
          const {open,close}=this.dataBySymbol[s][k];
          const n=this.normalizers[s];
          const o=(open-n.oMin)/(n.oMax-n.oMin+1e-6);
          const c=(close-n.cMin)/(n.cMax-n.cMin+1e-6);
          feats.push(o,c);
        }
        input.push(feats);
      }

      const base=this.symbols.map(s=>this.dataBySymbol[s][t].close);
      const labels=[];
      for(let h=1;h<=H;h++){
        for(let s=0;s<S;s++){
          const fut=this.dataBySymbol[this.symbols[s]][t+h].close;
          labels.push(fut>base[s]?1:0);
        }
      }
      X.push(input);
      Y.push(labels);
    }

    const split=Math.floor(X.length*(1-this.testSplit));
    const X_train=tf.tensor3d(X.slice(0,split));
    const y_train=tf.tensor2d(Y.slice(0,split));
    const X_test=tf.tensor3d(X.slice(split));
    const y_test=tf.tensor2d(Y.slice(split));
    return {X_train,y_train,X_test,y_test,symbols:this.symbols};
  }
}
