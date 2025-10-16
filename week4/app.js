// app.js

class StockApp {
  constructor() {
    this.fileInput = document.getElementById("csvFile");
    this.loadBtn = document.getElementById("btnLoad");
    this.trainBtn = document.getElementById("btnTrain");
    this.progress = document.getElementById("progress");
    this.chartDiv = document.getElementById("accuracyChart");

    this.loader = new DataLoader();
    this.loadBtn.onclick = () => this.loadCsv();
    this.trainBtn.onclick = () => this.train();
  }

  async loadCsv() {
    try {
      const file = this.fileInput.files[0];
      if (!file) return alert("Select a CSV file first!");
      this.progress.innerText = "Loading CSV...";
      await this.loader.parseCsvFile(file);
      this.dataset = this.loader.buildSamples();
      this.progress.innerText = `Loaded ${this.dataset.X_train.shape[0]} train samples.`;
      this.trainBtn.disabled = false;
    } catch (err) {
      alert("Error loading CSV: " + err.message);
      console.error(err);
    }
  }

  async train() {
    const { X_train, y_train, X_test, y_test, symbols } = this.dataset;
    const [seq, feat] = X_train.shape.slice(1);
    const model = new GRUModel({ inputShape: [seq, feat], denseUnits: y_train.shape[1] });
    model.build();

    this.progress.innerText = "Training...";
    await model.fit(X_train, y_train, {
      epochs: 25,
      batchSize: 32,
      onEpoch: (e, logs) => {
        this.progress.innerText = `Epoch ${e + 1}: loss=${logs.loss.toFixed(4)}, acc=${logs.binaryAccuracy.toFixed(4)}`;
      },
    });

    this.progress.innerText = "Predicting...";
    const preds = model.predict(X_test);
    const accs = await model.computeAccuracy(preds, y_test, symbols);

    this.showAccuracy(accs);
    this.progress.innerText = "Done!";
  }

  showAccuracy(accs) {
    const sorted = accs.sort((a, b) => b.accuracy - a.accuracy);
    const labels = sorted.map(a => a.symbol);
    const data = sorted.map(a => (a.accuracy * 100).toFixed(2));

    this.chartDiv.innerHTML = `<canvas id="chart"></canvas>`;
    const ctx = document.getElementById("chart").getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{ label: "Accuracy %", data }],
      },
      options: { scales: { y: { beginAtZero: true, max: 100 } } },
    });
  }
}

window.addEventListener("DOMContentLoaded", () => new StockApp());
