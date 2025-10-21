// Interactive EDA charts using Plotly

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
  }], {
    title: "Bookings by Month",
    xaxis: { title: "Month" },
    yaxis: { title: "Count" },
  });

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
  const cancelRates = months.map(
    m => (cancelByMonth[m].canceled / cancelByMonth[m].total) * 100
  );
  Plotly.newPlot("cancelPlot", [{
    x: months,
    y: cancelRates,
    type: "bar",
    marker: { color: "#d62728" },
  }], {
    title: "Cancellation Rate by Month (%)",
    xaxis: { title: "Month" },
    yaxis: { title: "Cancellation %" },
  });
}
