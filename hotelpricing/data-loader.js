let uploadedFile = null;

// When file is selected
document.getElementById("csvFile").addEventListener("change", function (e) {
  uploadedFile = e.target.files[0];
});

// Load and visualize
document.getElementById("loadButton").addEventListener("click", function () {
  if (!uploadedFile) {
    alert("Please select a CSV file first!");
    return;
  }

  Papa.parse(uploadedFile, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: function (results) {
      const data = results.data;
      console.log("✅ CSV Loaded:", data.slice(0, 5));

      window.uploadedData = data;

      // Visualize EDA if ADR column exists
      if (data[0].adr && typeof visualizeEDA === "function") visualizeEDA(data);

      // Visualize GRU if predicted_adr exists
      if (data[0].predicted_adr && typeof visualizeGRU === "function") visualizeGRU(data);

      // Visualize correlation heatmap
      if (typeof visualizeCorrelation === "function") visualizeCorrelation(data);
    },
  });
});
