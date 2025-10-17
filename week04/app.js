// app.js
// Orchestrates UI, DataLoader, GRUModel. Trains with responsive UI, shows sorted accuracy bar chart and per-stock timelines.
// Uses global DataLoader and GRUModel (exposed by previous files), and Chart.js.

(function () {
  // DOM-ready initialization
  window.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("csvFile");
    const loadBtn = document.getElementById("btnLoad");
    const trainBtn = document.getElementById("btnTrain");
    const progress = document.getElementById("progress");
    const accuracyChartDiv = document.getElementById("accuracyChart");

    let loader = null;
    let modelWrapper = null;
    let dataset = null; // {X_train, y_train, X_test, y_test, symbols}

    // utility log
    function log(msg) {
      console.log(msg);
      progress.innerText = msg;
    }

    loadBtn.addEventListener("click", async () => {
      try {
        const file = fileInput.files[0];
        if (!file) { alert("Please select a CSV file"); return; }
        loader = new window.DataLoader();
        log("Parsing CSV...");
        await loader.parseCsvFile(file);
        log("Building samples...");
        dataset = loader.buildSamples(); // throws on failure
        log(`Loaded. Train samples: ${dataset.X_train.shape[0]}, Test samples: ${dataset.X_test.shape[0]}`);
        trainBtn.disabled = false;
      } catch (err) {
        console.error(err);
        alert("Error loading CSV: " + (err.message || err));
        log("Error loading CSV");
      }
    });

    trainBtn.addEventListener("click", async () => {
      try {
        if (!dataset) { alert("Load data first"); return; }
        trainBtn.disabled = true;
        loadBtn.disabled = true;

        const { X_train, y_train, X_test, y_test, symbols } = dataset;
        // input shape inferred
        const seq = X_train.shape[1];
        const feat = X_train.shape[2];

        log("Building model...");
        modelWrapper = new window.GRUModel({ inputShape: [seq, feat], outputSize: y_train.shape[1], learningRate: 0.001 });
        modelWrapper.build();

        log("Training (this may take a while)...");
        await modelWrapper.fit(X_train, y_train, {
          epochs: 80,
          batchSize: Math.max(8, Math.min(32, Math.floor(X_train.shape[0] / 10))),
          validationSplit: 0.1,
          onEpoch: (epoch, logs) => {
            const ep = epoch + 1;
            const loss = (logs.loss || 0).toFixed(4);
            const acc = logs.binaryAccuracy ? (logs.binaryAccuracy * 100).toFixed(2) : "0.00";
            const valLoss = logs.val_loss ? logs.val_loss.toFixed(4) : "-";
            const valAcc = logs.val_binaryAccuracy ? (logs.val_binaryAccuracy * 100).toFixed(2) : "-";
            log(`Epoch ${ep}/80\nLoss=${loss} | Acc=${acc}%\nValLoss=${valLoss} | ValAcc=${valAcc}%`);
          }
        });

        log("Predicting and smoothing...");
        const rawPredsOrPromise = modelWrapper.predict(X_test, { smoothing: true, window: 3 });
        // compute per-stock accuracy (handles Promise or tensor)
        const res = await modelWrapper.computePerStockAccuracy(rawPredsOrPromise, y_test, symbols);

        renderAccuracyChart(res.accuracies);
        renderTimelines(res.accuracies, res.horizon);

        log("✅ Done! Final accuracies rendered.");
        trainBtn.disabled = false;
        loadBtn.disabled = false;

        // cleanup tensors in loader if needed
        // (keep them if user wants to re-evaluate; uncomment to free)
        // loader.dispose();
      } catch (err) {
        console.error(err);
        alert("Training error: " + (err.message || err));
        log("Training error");
        trainBtn.disabled = false;
        loadBtn.disabled = false;
      }
    });

    // Render horizontal bar chart sorted best->worst
    let chartInstance = null;
    function renderAccuracyChart(accuracies) {
      const sorted = accuracies.slice().sort((a, b) => b.accuracy - a.accuracy);
      const labels = sorted.map(s => `${s.symbol} (${(s.accuracy * 100).toFixed(1)}%)`);
      const data = sorted.map(s => +(s.accuracy * 100).toFixed(2));

      accuracyChartDiv.innerHTML = `<canvas id="accuracyCanvas"></canvas><div id="timelines"></div>`;
      const ctx = document.getElementById("accuracyCanvas").getContext("2d");
      if (chartInstance) chartInstance.destroy();
      chartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels,
          datasets: [{
            label: "Accuracy %",
            data,
            backgroundColor: "rgba(54,162,235,0.5)",
            borderColor: "rgba(54,162,235,0.9)",
            borderWidth: 1
          }]
        },
        options: {
          indexAxis: "x",
          responsive: true,
          scales: {
            y: { beginAtZero: true, max: 100 }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => `Accuracy %: ${ctx.parsed.y}`
              }
            }
          }
        }
      });
    }

    // Render per-stock simple timeline bars (green=correct, red=wrong) from accuracy.timeline if available
    function renderTimelines(accuracies, horizon) {
      const timelinesDiv = document.getElementById("timelines");
      if (!timelinesDiv) return;
      timelinesDiv.innerHTML = "<h3>Per-stock correctness timelines (green=correct, red=wrong)</h3>";
      // if timeline arrays present, use them; else show proportional bar
      for (const a of accuracies.sort((a, b) => b.accuracy - a.accuracy)) {
        const wrapper = document.createElement("div");
        wrapper.style.margin = "10px 0";
        const title = document.createElement("div");
        title.innerText = `${a.symbol} — Accuracy ${(a.accuracy * 100).toFixed(1)}%`;
        title.style.fontWeight = "600";
        wrapper.appendChild(title);

        const canvas = document.createElement("canvas");
        canvas.width = 900;
        canvas.height = 20;
        canvas.style.border = "1px solid #ddd";
        canvas.style.display = "block";
        canvas.style.marginTop = "6px";
        wrapper.appendChild(canvas);

        const ctx = canvas.getContext("2d");
        // If timeline data exists, draw exact sequence; otherwise draw proportion
        const tl = a.timeline;
        if (Array.isArray(tl) && tl.length > 0) {
          const cellW = canvas.width / tl.length;
          for (let i = 0; i < tl.length; i++) {
            ctx.fillStyle = tl[i] ? "#2e7d32" : "#c62828";
            ctx.fillRect(i * cellW, 0, Math.ceil(cellW), canvas.height);
          }
        } else {
          // proportional bar: left = accuracy green, right = red
          ctx.fillStyle = "#2e7d32";
          ctx.fillRect(0, 0, canvas.width * a.accuracy, canvas.height);
          ctx.fillStyle = "#c62828";
          ctx.fillRect(canvas.width * a.accuracy, 0, canvas.width * (1 - a.accuracy), canvas.height);
        }
        timelinesDiv.appendChild(wrapper);
      }
    }

  });
})();
