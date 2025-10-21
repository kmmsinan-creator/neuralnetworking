window.onload = () => {
  console.log("🌐 Hotel Pricing Dashboard Loaded");

  // Download EDA report
  document.getElementById("downloadReport").addEventListener("click", () => {
    if (!window.uploadedData) {
      alert("Please load data first!");
      return;
    }

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    doc.text("Hotel Room Pricing Optimization - EDA Report", 10, 10);
    doc.text("Dataset Records: " + window.uploadedData.length, 10, 20);
    doc.text("Generated using GRU Demand Forecast Dashboard", 10, 30);
    doc.save("Hotel_EDA_Report.pdf");
  });
};
