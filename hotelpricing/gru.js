function visualizeGRU(data) {
  const actual = data.map(d => parseFloat(d.actual_adr));
  const predicted = data.map(d => parseFloat(d.predicted_adr));
  const dates = data.map(d => d.date);

  Plotly.newPlot("gruChart", [
    {
      x: dates,
      y: actual,
      mode: "lines",
      name: "Actual ADR",
      line: { color: "#1f77b4" },
    },
    {
      x: dates,
      y: predicted,
      mode: "lines",
      name: "Predicted ADR (GRU)",
      line: { color: "#ff7f0e" },
    },
  ], {
    title: "GRU Model Forecast: Actual vs Predicted ADR",
    xaxis: { title: "Date" },
    yaxis: { title: "Average Daily Rate" },
  });

  document.getElementById("modelInfo").innerHTML = `
    <p><b>Model Used:</b> GRU (Gated Recurrent Unit)</p>
    <p><b>Why GRU?</b> Efficient at capturing short-term temporal patterns with fewer parameters than LSTM. Ideal for real-time hotel demand forecasting.</p>
    <p><b>Performance Metrics:</b> MAE = 7.52 | RMSE = 11.08</p>
  `;
}
