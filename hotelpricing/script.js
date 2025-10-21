function visualizeEDA(data) {
  // ADR Distribution
  const adr = data.map(d => parseFloat(d.adr)).filter(v => !isNaN(v));
  Plotly.newPlot("adrPlot", [{
    x: adr,
    type: "histogram",
    marker: { color: "#1f77b4" },
  }], {
    title: "ADR (Average Daily Rate) Distribution",
    xaxis: { title: "ADR" },
    yaxis: { title: "Count" },
  });

  // Bookings by Month
  const monthCounts = {};
  data.forEach(d => {
    if (d.arrival_date_month) {
      monthCounts[d.arrival_date_month] = (monthCounts[d.arrival_date_month] || 0) + 1;
    }
  });
  Plotly.newPlot("monthPlot", [{
    x: Object.keys(monthCounts),
    y: Object.values(monthCounts),
    type: "bar",
    marker: { color: "#2ca02c" },
  }], { title: "Bookings by Month" });

  // Cancellation Rate by Month
  const cancelByMonth = {};
  data.forEach(d => {
    const m = d.arrival_date_month;
    if (!m) return;
    if (!cancelByMonth[m]) cancelByMonth[m] = { total: 0, canceled: 0 };
    cancelByMonth[m].total++;
    if (d.is_canceled === 1) cancelByMonth[m].canceled++;
  });
  const months = Object.keys(cancelByMonth);
  const cancelRates = months.map(m => (cancelByMonth[m].canceled / cancelByMonth[m].total) * 100);
  Plotly.newPlot("cancelPlot", [{
    x: months,
    y: cancelRates,
    type: "bar",
    marker: { color: "#d62728" },
  }], { title: "Cancellation Rate by Month (%)" });

  // Missing Values
  const missingCounts = {};
  Object.keys(data[0]).forEach(key => {
    missingCounts[key] = data.filter(d => d[key] === "" || d[key] === null).length;
  });
  Plotly.newPlot("missingPlot", [{
    x: Object.keys(missingCounts),
    y: Object.values(missingCounts),
    type: "bar",
    marker: { color: "#9467bd" },
  }], {
    title: "Missing Values per Column",
    xaxis: { title: "Columns" },
    yaxis: { title: "Missing Count" },
  });
}

// Correlation Heatmap
function visualizeCorrelation(data) {
  // Extract numeric columns
  const numericKeys = Object.keys(data[0]).filter(key =>
    data.every(d => !isNaN(parseFloat(d[key])))
  );
  if (numericKeys.length < 2) return;

  // Build correlation matrix
  const values = numericKeys.map(key => data.map(d => parseFloat(d[key])));
  const corrMatrix = numericKeys.map((rowKey, i) =>
    numericKeys.map((colKey, j) => {
      const x = values[i];
      const y = values[j];
      const n = x.length;
      const meanX = x.reduce((a,b)=>a+b,0)/n;
      const meanY = y.reduce((a,b)=>a+b,0)/n;
      const cov = x.reduce((sum, xi, idx) => sum + (xi-meanX)*(y[idx]-meanY), 0)/n;
      const stdX = Math.sqrt(x.reduce((sum, xi) => sum + Math.pow(xi-meanX,2),0)/n);
      const stdY = Math.sqrt(y.reduce((sum, yi, idx) => sum + Math.pow(yi-meanY,2),0)/n);
      return cov/(stdX*stdY);
    })
  );

  Plotly.newPlot("corrPlot", [{
    z: corrMatrix,
    x: numericKeys,
    y: numericKeys,
    type: "heatmap",
    colorscale: "Viridis",
    zmin: -1,
    zmax: 1
  }], { title: "Correlation Heatmap" });
}
