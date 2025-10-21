// Handles CSV upload and parsing using PapaParse

document.getElementById("csvFile").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;

  Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    complete: function (results) {
      const data = results.data;
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
