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
    complete: function (results) {
      const data = results.data.filter(row => Object.keys(row).length > 1);
      console.log("✅ CSV Loaded:", data.slice(0, 5));

      window.uploadedData = data;

      if (data[0].adr) {
        visualizeEDA(data);
      }
      if (data[0].predicted_adr) {
        visualizeGRU(data);
      }
    },
  });
});
