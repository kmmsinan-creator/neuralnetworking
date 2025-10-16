// app.js
// ES6 module: orchestrates UI, ties DataLoader and GRUModel, renders charts using Chart.js (client-side).
// Index.html should include elements with IDs used below. Use tf.js from CDN and Chart.js.

import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0/dist/tf.min.js";
import { DataLoader } from "./data-loader.js";
import { GRUModel } from "./gru.js";

// Chart.js CDN is expected in index.html. If not present, we create a simple fallback.

class StockApp {
  constructor() {
    // UI elements
    this.fileInput = document.getElementById("csvFile");
    this.loadBtn = document.getElementById("btnLoad");
    this.trainBtn = document.getElementById("btnTrain");
    this.saveBtn = document.getElementById("btnSave");
    this.loadModelBtn = document.getElementById("btnLoadModel");
    this.progressText = document.getElementById("progress");
    this.accuracyContainer = document.getElementById("accuracyChart");
    this.timelineContainer = document.getElementById("timelineArea");
    this.logArea = document.getElementById("logArea");

    // Instances
    this.loader = new DataLoader();
    this.model = new GRUModel();
    this.dataset = null; // will hold tensors and symbols

    // Chart references
    this.accuracyChart = null;
    this.timelineCharts = [];

    // Bind handlers
    this.loadBtn.addEventListener("click", () => this.onLoadCsv());
    this.trainBtn.addEventListener("click", () => this.onTrain());
    this.saveBtn.addEventListener("click", () => this.onSaveModel());
    this.loadModelBtn.addEventListener("click", () => this.onLoadModel());

    // Safety: prevent accidental navigation during operations
    window.addEventListener("beforeunload", (e) => {
      if (this.training) {
        e.preventDefault();
        e.returnValue = "";
      }
    });

    this.appendLog("App initialized.");
  }

  appendLog(msg) {
    const time = new Date().toLocaleTimeString();
    if (this.logArea) {
      this.logArea.value = `[${time}] ${msg}\n` + this.logArea.value;
    } else {
      console.log(msg);
    }
  }

  async onLoadCsv() {
    try {
      const file = this.fileInput.files[0];
      if (!file) { alert("Choose a CSV file first"); return; }
      this.appendLog(`Parsing CSV ${file.name}...`);
      this.progressText.innerText = "Parsing CSV...";
      await this.loader.parseCsvFile(file);
      this.appendLog(`CSV parsed. Symbols: ${this.loader.symbols.join(", ")}`);
      this.progressText.innerText = "Building samples...";
      const tensors = this.loader.buildSamples();
      this.dataset = {
        X_train: tensors.X_train,
        y_train: tensors.y_train,
        X_test: tensors.X_test,
        y_test: tensors.y_test,
        symbols: this.loader.symbols,
      };
      this.appendLog(`Samples built. Train samples: ${this.dataset.X_train.shape[0]}, Test samples: ${this.dataset.X_test.shape[0]}`);
      this.progressText.innerText = "Ready.";
      // Enable train button
      this.trainBtn.disabled = false;
    } catch (e) {
      console.error(e);
      alert("Error loading CSV: " + e.message);
      this.appendLog("Error loading CSV: " + e.message);
      this.progressText.innerText = "Error.";
    }
  }

  async onTrain() {
    if (!this.dataset) { alert("Load data first"); return; }
    this.training = true;
    try {
      this.appendLog("Building model...");
      // derive input shape from dataset
      const [samples, seq, feat] = this.dataset.X_train.shape;
      this.model = new GRUModel({ inputShape: [seq, feat], denseUnits: this.dataset.y_train.shape[1] });
      this.model.build();
      this.appendLog("Model built. Starting training...");
      this.progressText.innerText = "Training...";
      // Fit with epoch callback to update UI
      const history = await this.model.fit(this.dataset.X_train, this.dataset.y_train, {
        epochs: 30,
        batchSize: Math.min(64, Math.max(8, Math.floor(samples / 10))),
        validationSplit: 0.1,
        shuffle: false,
        onEpoch: (epoch, logs) => {
          this.progressText.innerText = `Epoch ${epoch + 1}: loss=${(logs.loss).toFixed(4)} val_loss=${(logs.val_loss||0).toFixed(4)} acc=${(logs.binaryAccuracy||0).toFixed(4)}`;
          this.appendLog(`Epoch ${epoch + 1} - loss ${logs.loss.toFixed(4)} - val_loss ${(logs.val_loss||0).toFixed(4)} - acc ${(logs.binaryAccuracy||0).toFixed(4)}`);
        },
      });
      this.appendLog("Training complete.");
      this.progressText.innerText = "Training complete. Evaluating...";
      // Evaluate & Predict
      const preds = await this.model.predict(this.dataset.X_test);
      const res = await this.model.computePerStockAccuracy(preds, this.dataset.y_test, this.dataset.symbols);
      // show visualizations
      this.renderAccuracyChart(res.accuracies);
      this.renderTimelines(res.accuracies, res.horizon, preds, this.dataset.y_test);
      this.appendLog("Evaluation & visualization done.");
      this.progressText.innerText = "Done.";
      preds.dispose();
    } catch (e) {
      console.error(e);
      alert("Training error: " + e.message);
      this.appendLog("Training error: " + e.message);
      this.progressText.innerText = "Error.";
    } finally {
      this.training = false;
      await tf.nextFrame();
    }
  }

  async onSaveModel() {
    if (!this.model || !this.model.model) { alert("No model to save"); return; }
    try {
      await this.model.saveToLocalStorage("gru-stock-demo");
      this.appendLog("Model saved to localStorage as 'gru-stock-demo'.");
      alert("Model saved to localStorage.");
    } catch (e) {
      console.error(e);
      alert("Save failed: " + e.message);
    }
  }

  async onLoadModel() {
    try {
      const m = new GRUModel();
      await m.loadFromLocalStorage("gru-stock-demo");
      this.model = m;
      this.appendLog("Model loaded from localStorage 'gru-stock-demo'.");
      alert("Model loaded. You can now evaluate on loaded data.");
    } catch (e) {
      console.error(e);
      alert("Load failed: " + e.message);
    }
  }

  // Render horizontal bar chart of accuracies sorted best->worst
  renderAccuracyChart(accuracies) {
    // accuracies: [{symbol, accuracy, confusion, timeline}, ...]
    const sorted = accuracies.slice().sort((a, b) => b.accuracy - a.accuracy);
    const labels = sorted.map(s => s.symbol + ` (${(s.accuracy * 100).toFixed(1)}%)`);
    const data = sorted.map(s => (s.accuracy * 100));

    // clear container
    this.accuracyContainer.innerHTML = '<canvas id="accuracyCanvas"></canvas>';
    const ctx = document.getElementById("accuracyCanvas").getContext("2d");
    if (window.Chart) {
      if (this.accuracyChart) { this.accuracyChart.destroy(); }
      this.accuracyChart = new Chart(ctx, {
        type: "horizontalBar",
        data: {
          labels,
          datasets: [{ label: "Accuracy %", data }],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          scales: {
            x: { beginAtZero: true, max: 100 },
          },
        },
      });
    } else {
      // fallback simple list
      this.accuracyContainer.innerHTML = "<pre>" + sorted.map(s => `${s.symbol}: ${(s.accuracy * 100).toFixed(2)}%`).join("\n") + "</pre>";
    }
  }

  // Render per-stock timelines; correctness (green=correct, red=wrong). Timeline is across test samples*H.
  async renderTimelines(accuracies, horizon, predsTensor, yTensor) {
    // predsTensor and yTensor are tensors; convert to arrays
    const preds = await predsTensor.array();
    const ys = await yTensor.array();
    const samples = preds.length;
    const S = accuracies.length;
    const H = horizon;
    // build per-stock boolean arrays of correctness by sample-day (we'll represent each test sample as a block of H days)
    const perStockTimeline = accuracies.map((a, si) => {
      const arr = [];
      for (let i = 0; i < samples; i++) {
        for (let h = 0; h < H; h++) {
          const idx = si + h * S;
          const p = preds[i][idx] >= 0.5 ? 1 : 0;
          const t = ys[i][idx] >= 0.5 ? 1 : 0;
          arr.push(p === t ? 1 : 0);
        }
      }
      return arr;
    });

    // create timeline area
    this.timelineContainer.innerHTML = "";
    // For each stock, create a small canvas visualizing timeline (green/red strips)
    perStockTimeline.forEach((arr, si) => {
      const wrapper = document.createElement("div");
      wrapper.className = "stock-timeline";
      const title = document.createElement("div");
      title.innerText = `${accuracies[si].symbol} — Acc ${(accuracies[si].accuracy * 100).toFixed(1)}%`;
      title.style.fontWeight = "600";
      title.style.marginBottom = "4px";
      wrapper.appendChild(title);
      // create canvas
      const canvas = document.createElement("canvas");
      canvas.width = Math.min(1200, Math.max(200, arr.length));
      canvas.height = 20;
      canvas.style.border = "1px solid #ddd";
      wrapper.appendChild(canvas);
      this.timelineContainer.appendChild(wrapper);

      // draw
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      const cellW = w / arr.length;
      for (let i = 0; i < arr.length; i++) {
        ctx.fillStyle = arr[i] ? "#2e7d32" : "#c62828"; // green / red
        ctx.fillRect(i * cellW, 0, Math.ceil(cellW), h);
      }
    });

    this.appendLog("Timelines rendered.");
  }
}

window.addEventListener("DOMContentLoaded", () => {
  // ensure Chart.js is available; if not, user will still get fallback
  const app = new StockApp();
});
