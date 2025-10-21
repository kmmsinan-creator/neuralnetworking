// Main application logic with file upload
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const dataPreview = document.getElementById('dataPreview');
    const previewTable = document.getElementById('previewTable');
    const controlsSection = document.getElementById('controlsSection');
    const predictButton = document.getElementById('predictBtn');
    const basePriceInput = document.getElementById('basePrice');
    const seasonSelect = document.getElementById('season');
    const isHolidayCheckbox = document.getElementById('isHoliday');
    const advanceBookingInput = document.getElementById('advanceBooking');
    const outputDiv = document.getElementById('output');
    const occupancyResultP = document.getElementById('occupancyResult');
    const priceResultP = document.getElementById('priceResult');
    const logicResultP = document.getElementById('logicResult');

    const dataLoader = new DataLoader();
    const model = new SimulatedGRU();

    // File upload handler
    fileInput.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;

        uploadStatus.textContent = 'Loading dataset...';
        uploadStatus.className = '';

        try {
            await dataLoader.loadFile(file);
            
            uploadStatus.textContent = '✅ Dataset loaded successfully!';
            uploadStatus.className = 'success';
            
            // Show data preview
            previewTable.innerHTML = dataLoader.getPreviewHTML();
            dataPreview.style.display = 'block';
            
            // Enable simulation controls
            controlsSection.style.display = 'block';
            
            console.log('Dataset features:', dataLoader.features);
            console.log('Dataset stats:', dataLoader.stats);

        } catch (error) {
            uploadStatus.textContent = `❌ Error: ${error.message}`;
            uploadStatus.className = 'error';
            console.error('File loading error:', error);
        }
    });

    // Prediction handler
    predictButton.addEventListener('click', function() {
        if (!dataLoader.dataset) {
            alert('Please upload a dataset first.');
            return;
        }

        // Get user inputs
        const basePrice = parseFloat(basePriceInput.value);
        const season = seasonSelect.value;
        const isHoliday = isHolidayCheckbox.checked;
        const leadTime = parseInt(advanceBookingInput.value);

        // Input validation
        if (isNaN(basePrice) || isNaN(leadTime)) {
            alert("Please enter valid numbers for price and lead time.");
            return;
        }

        // 1. GET PREDICTION FROM (SIMULATED) GRU MODEL
        const predictedOccupancy = model.predict(season, isHoliday, leadTime);
        const occupancyPercent = (predictedOccupancy * 100).toFixed(1);

        // 2. SMART PRICING LOGIC
        let suggestedPrice;
        let logicExplanation;

        if (predictedOccupancy > 0.75) {
            suggestedPrice = basePrice * 1.4;
            logicExplanation = `High predicted occupancy (${occupancyPercent}%). Capitalize on demand by increasing price.`;
        } else if (predictedOccupancy > 0.5) {
            suggestedPrice = basePrice * 1.1;
            logicExplanation = `Moderate predicted occupancy (${occupancyPercent}%). Slight price increase to maximize revenue.`;
        } else {
            suggestedPrice = basePrice * 0.8;
            logicExplanation = `Low predicted occupancy (${occupancyPercent}%). Competitive discount to attract bookings and reduce empty rooms.`;
        }

        // 3. UPDATE THE UI WITH RESULTS
        occupancyResultP.textContent = `${occupancyPercent}%`;
        priceResultP.textContent = `$${suggestedPrice.toFixed(2)}`;
        priceResultP.style.color = suggestedPrice > basePrice ? '#e74c3c' : '#27ae60';
        logicResultP.textContent = logicExplanation;

        outputDiv.innerHTML = `<p>Based on <strong>${dataLoader.stats.rowCount} data points</strong> and GRU model analysis:</p>`;
    });
});
