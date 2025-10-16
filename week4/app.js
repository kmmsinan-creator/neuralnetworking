// app.js
// ES Module: UI glue, tie DataLoader + GRUModel, training loop, compute per-stock accuracies and visualizations.
// Important: import TF ESM in module scope to ensure tf.* functions exist.

import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0/dist/tf.esm.js";
import { DataLoader } from "./data-loader.js";
import { GRUModel } from "./gru.js";

class StockApp {
  constructor() {
    // UI elements (index.html must contain these IDs)
    this.fileInput = document.getElementById("csvFile");
    this.loadBtn = document.getElementById("btnLoad");
    this.trainBtn = document.getElementById("btnTrain");
    this.saveBtn = document.getElementById("btnSave");
    this.loadModelBtn = document.getElementById("btnLoadModel");
    this.progressText = document.getElementById("progress");
    this.accuracyContainer = document.getElementById("accuracyChart");
    this.timelineContainer = document.getElementById("timelineArea");
    this.logArea = document.getElementById("logArea");

    // instances
    this.loader = new DataLoader();
    this.modelWrapper = null;
    this.dataset = null;
    this.accuracyChart = null;

    // bind
    this.loadBtn.addEventListener("click", () => this.onLoadCsv());
    this.trainBtn.addEventListener("click", () => this.onTrain());
    this.saveBtn.addEventListener("click", () => this.onSaveModel());
    this.loadModelBtn.addEventListener("click", () => this.onLoadModel());

    this.appendLog("App ready.");
  }

  appendLog(msg) {
    const ts = new Date().toLocaleTimeString();
    if (this.logArea) this.logArea.value = `[${ts}] ${msg}\n` + this.logArea.value;
    else console.log(msg);
  }

  async onLoadCsv() {
    try {
      const file = this.fileInput.files[0];
      if (!file) { alert("Please select a CSV file"); return; }
      this.progressText.innerText = "Parsing CSV...";
      await this.loader.parseCsvFile(file);
      this.appendLog(`CSV parsed. Symbols: ${this.loader.symbols.join(", ")}`);
      this.progressText.innerText = "Building samples...";
      const ds = this.loader.buildSamples(); // may throw if tf not loaded or shapes invalid
      this.dataset = ds;
      this.appendLog(`Samples built. Train samples: ${ds.X_train.shape[0]}, Test: ${ds.X_test.shape[0]}`);
      this.progressText.innerText = "Ready to train";
      this.trainBtn.disabled = false;
    } catch (err) {
      console.error(err);
      this.appendLog("Error loading CSV: " + (err && err.message ? err.message : err));
      alert("Error loading CSV: " + (err && err.message ? err.message : err));
      this.progressText.innerText = "Error loading CSV";
    }
  }

  async onTrain() {
    try {
      if (!this.dataset) { alert("Load data first"); return; }
      this.trainBtn.disabled = true;
      const { X_train, y_train, X_test, y_test, symbols } = this.dataset;
      const [samples, seq, feat] = X_train.shape;
      this.modelWrapper = new GRUModel({ inputShape: [seq, feat], denseUnits: y_train.shape[1] });
      this.modelWrapper.build();
      this.appendLog("Model built. Starting training...");
      this.progressText.innerText = "Training...";
      await this.modelWrapper.fit(X_train, y_train, {
        epochs: 30,
        batchSize: Math.min(64, Math.max(8, Math.floor(samples / 10))),
        validationSplit: 0.1,
        shuffle: false,
        onEpoch: (epoch, logs) => {
          this.progressText.innerText = `Epoch ${epoch + 1}: loss=${(logs.loss||0).toFixed(4)} val_loss=${(logs.val_loss||0).toFixed(4)} acc=${(logs.binaryAccuracy||0).toFixed(4)}`;
        },
      });
      this.appendLog("Training finished. Predicting on test set...");
      this.progressText.innerText = "Predicting...";
      const preds = this.modelWrapper.predict(X_test);
      const results = await this.modelWrapper.computePerStockAccuracy(preds, y_test, symbols);
      this.renderAccuracy(results.accuracies);
      this.renderTimelines(results.accuracies, results.horizon);
      this.appendLog("Done evaluation.");
      preds.dispose();
      this.progressText.innerText = "Done";
      this.trainBtn.disabled = false;
    } catch (err) {
      console.error(err);
      this.appendLog("Training error: " + (err && err.message ? err.message : err));
      alert("Training error: " + (err && err.message ? err.message : err));
      this.progressText.innerText = "Error during training";
      this.trainBtn.disabled = false;
    }
  }

  async onSaveModel() {
    try {
      if (!this.modelWrapper || !this.modelWrapper.model) { alert("No model to save"); return; }
      await this.modelWrapper.saveToLocalStorage("gru-stock-demo");
      this.appendLog("Model saved to localStorage: gru-stock-demo");
      alert("Model saved to browser localStorage.");
    } catch (err) {
      console.error(err);
      alert("Save failed: " + err.message);
    }
  }

  async onLoadModel() {
    try {
      const wrapper = new GRUModel();
      await wrapper.loadFromLocalStorage("gru-stock-demo");
      this.modelWrapper = wrapper;
      this.appendLog("Model loaded from localStorage");
      alert("Model loaded. You can now predict on loaded dataset (load CSV first).");
    } catch (err) {
      console.error(err);
      alert("Load failed: " + (err && err.message ? err.message : err));
    }
  }

  renderAccuracy(accuracies) {
    // sort descending
    const sorted = accuracies.slice().sort((a, b) => b.accuracy - a.accuracy);
    const labels = sorted.map(s => s.symbol + ` (${(s.accuracy*100).toFixed(1)}%)`);
    const data = sorted.map(s => +(s.accuracy*100).toFixed(2));

    this.accuracyContainer.innerHTML = `<canvas id="accuracyCanvas"></canvas>`;
    const ctx = document.getElementById("accuracyCanvas").getContext("2d");
    if (window.Chart) {
      if (this.accuracyChart) this.accuracyChart.destroy();
      this.accuracyChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels,
          datasets: [{ label: "Accuracy %", data }],
        },
        options: {
          indexAxis: "y",
          scales: {
            x: { beginAtZero: true, max: 100 },
          },
        },
      });
    } else {
      this.accuracyContainer.innerHTML = "<pre>" + sorted.map(s => `${s.symbol}: ${(s.accuracy*100).toFixed(2)}%`).join("\n") + "</pre>";
    }
  }

  renderTimelines(accuracies, horizon) {
    // Clear
    this.timelineContainer.innerHTML = "";
    // We stored timelines in computePerStockAccuracy as an array per stock if needed; but to keep simple,
    // we'll build placeholder small canvases indicating overall accuracy (green/red stripes not re-derived here).
    for (const a of accuracies) {
      const div = document.createElement("div");
      div.style.marginBottom = "8px";
      const title = document.createElement("div");
      title.textContent = `${a.symbol} — Accuracy ${(a.accuracy * 100).toFixed(1)}%`;
      title.style.fontWeight = "600";
      div.appendChild(title);
      const canvas = document.createElement("canvas");
      canvas.width = 800;
      canvas.height = 20;
      canvas.style.border = "1px solid #ddd";
      div.appendChild(canvas);
      // fill canvas proportionally: left portion green = accuracy, right red = 1-accuracy
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#2e7d32"; // green
      ctx.fillRect(0, 0, canvas.width * a.accuracy, canvas.height);
      ctx.fillStyle = "#c62828"; // red
      ctx.fillRect(canvas.width * a.accuracy, 0, canvas.width * (1 - a.accuracy), canvas.height);
      this.timelineContainer.appendChild(div);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  // Basic sanity check - ensure tf tensor creation functions exist in the module's tf import
  if (typeof tf === "undefined" || typeof tf.tensor3d !== "function") {
    alert("TensorFlow.js not loaded correctly in module scope. Ensure you are using the provided index.html (no global tf script) and that your browser supports ES Modules.");
    console.error("tf or tf.tensor3d missing in module scope:", tf);
  }
  new StockApp();
});
