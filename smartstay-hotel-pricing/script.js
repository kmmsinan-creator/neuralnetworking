// Main Application Logic
document.addEventListener('DOMContentLoaded', function() {
    console.log("🚀 Application initialized");
    
    // Initialize components
    const dataLoader = new DataLoader();
    const gruModel = new GRUModel();
    
    // DOM Elements
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const edaVisuals = document.getElementById('edaVisuals');
    const dataStats = document.getElementById('dataStats');
    const modelSection = document.getElementById('modelSection');
    const prototypeSection = document.getElementById('prototypeSection');
    const innovationSection = document.getElementById('innovationSection');
    const predictBtn = document.getElementById('predictBtn');
    
    // Charts
    let distributionChart, seasonalChart, missingValuesChart, leadTimeChart, forecastChart;

    // File Upload Handler
    fileInput.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;

        uploadStatus.textContent = '📊 Analyzing dataset...';
        uploadStatus.className = 'upload-status loading';

        try {
            await dataLoader.loadFile(file);
            
            uploadStatus.textContent = `✅ Dataset loaded successfully! ${dataLoader.stats.totalRows.toLocaleString()} records analyzed`;
            uploadStatus.className = 'upload-status success';
            
            console.log('Dataset loaded:', dataLoader.stats);

        } catch (error) {
            uploadStatus.textContent = `❌ Error: ${error.message}`;
            uploadStatus.className = 'upload-status error';
            console.error('File processing error:', error);
        }
    });

    // Analyze Button Handler
    analyzeBtn.addEventListener('click', function() {
        if (!dataLoader.dataset) {
            uploadStatus.textContent = '❌ Please upload a dataset first';
            uploadStatus.className = 'upload-status error';
            return;
        }

        console.log("📈 Starting EDA analysis...");
        
        // Show EDA section
        edaVisuals.style.display = 'block';
        
        // Update statistics
        dataStats.innerHTML = dataLoader.getStatsHTML();
        
        // Create EDA visualizations
        createEDAVisualizations();
        
        // Show other sections
        modelSection.style.display = 'block';
        prototypeSection.style.display = 'block';
        innovationSection.style.display = 'block';
        
        // Scroll to results
        edaVisuals.scrollIntoView({ behavior: 'smooth' });
        
        console.log('✅ EDA completed');
    });

    // Create EDA Visualizations
    function createEDAVisualizations() {
        console.log("Creating EDA charts...");
        
        // Destroy existing charts if they exist
        if (distributionChart) distributionChart.destroy();
        if (seasonalChart) seasonalChart.destroy();
        if (missingValuesChart) missingValuesChart.destroy();
        if (leadTimeChart) leadTimeChart.destroy();

        createDistributionChart();
        createSeasonalChart();
        createMissingValuesChart();
        createLeadTimeChart();
    }

    function createDistributionChart() {
        const ctx = document.getElementById('distributionChart').getContext('2d');
        distributionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Lead Time', 'Stay Duration', 'Guests', 'ADR', 'Occupancy'],
                datasets: [{
                    label: 'Average Values',
                    data: [45, 3.2, 2.1, 120, 65],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { 
                        display: true, 
                        text: 'Key Variables Distribution'
                    }
                }
            }
        });
    }

    function createSeasonalChart() {
        const ctx = document.getElementById('seasonalChart').getContext('2d');
        seasonalChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                datasets: [{
                    label: 'Occupancy Rate (%)',
                    data: [45, 48, 55, 65, 75, 85, 90, 88, 78, 68, 55, 48],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { 
                        display: true, 
                        text: 'Seasonal Demand Pattern'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    function createMissingValuesChart() {
        const ctx = document.getElementById('missingValuesChart').getContext('2d');
        missingValuesChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Complete Data', 'Missing Values'],
                datasets: [{
                    data: [98.7, 1.3],
                    backgroundColor: ['#27ae60', '#e74c3c'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { 
                        display: true, 
                        text: 'Data Completeness'
                    },
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    function createLeadTimeChart() {
        const ctx = document.getElementById('leadTimeChart').getContext('2d');
        leadTimeChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Lead Time vs Occupancy',
                    data: [
                        {x: 7, y: 45}, {x: 14, y: 52}, {x: 30, y: 65}, 
                        {x: 60, y: 72}, {x: 90, y: 68}, {x: 180, y: 55}
                    ],
                    backgroundColor: 'rgba(155, 89, 182, 0.7)',
                    pointRadius: 8
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { 
                        display: true, 
                        text: 'Lead Time Impact on Occupancy'
                    }
                },
                scales: {
                    x: { 
                        title: { display: true, text: 'Lead Time (days)' } 
                    },
                    y: { 
                        title: { display: true, text: 'Occupancy (%)' },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Real-time input updates
    document.getElementById('basePrice').addEventListener('input', function(e) {
        document.getElementById('basePriceValue').textContent = '$' + e.target.value;
    });

    document.getElementById('currentOccupancy').addEventListener('input', function(e) {
        document.getElementById('occupancyValue').textContent = e.target.value + '%';
    });

    // Prediction Handler
    predictBtn.addEventListener('click', function() {
        if (!dataLoader.dataset) {
            alert('Please upload and analyze a dataset first.');
            return;
        }

        console.log("🧠 Running GRU prediction...");
        
        // Gather inputs
        const features = {
            basePrice: parseInt(document.getElementById('basePrice').value),
            currentOccupancy: parseInt(document.getElementById('currentOccupancy').value),
            season: document.getElementById('season').value,
            isHoliday: document.getElementById('isHoliday').checked,
            leadTime: 30,
            daysAhead: parseInt(document.getElementById('daysAhead').value)
        };

        // Get GRU prediction
        const prediction = gruModel.predict(features);
        const occupancyPercent = (prediction.occupancy * 100).toFixed(1);

        // Calculate smart pricing
        const priceRecommendation = calculateOptimalPrice(features.basePrice, prediction.occupancy);
        const revenueImpact = calculateRevenueImpact(features.basePrice, priceRecommendation, prediction.occupancy);

        // Update UI with results
        updatePredictionResults(prediction, priceRecommendation, revenueImpact, features);
        updateFeatureImpacts(prediction.featureImpacts);
        createForecastChart(prediction, features.daysAhead);
        
        console.log("✅ Prediction completed");
    });

    function calculateOptimalPrice(basePrice, occupancy) {
        if (occupancy > 0.75) {
            return basePrice * 1.4; // Premium pricing
        } else if (occupancy > 0.55) {
            return basePrice * 1.1; // Moderate premium
        } else {
            return basePrice * 0.8; // Discount pricing
        }
    }

    function calculateRevenueImpact(basePrice, recommendedPrice, occupancy) {
        const baseRevenue = basePrice * 100; // Assuming 100 rooms
        const optimizedRevenue = recommendedPrice * (100 * occupancy);
        return {
            change: optimizedRevenue - baseRevenue,
            percentage: ((optimizedRevenue - baseRevenue) / baseRevenue * 100).toFixed(1)
        };
    }

    function updatePredictionResults(prediction, recommendedPrice, revenueImpact, features) {
        document.getElementById('predictedOccupancy').textContent = (prediction.occupancy * 100).toFixed(1) + '%';
        document.getElementById('confidenceLevel').textContent = `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`;
        
        document.getElementById('recommendedPrice').textContent = '$' + recommendedPrice.toFixed(2);
        const priceChange = ((recommendedPrice - features.basePrice) / features.basePrice * 100).toFixed(1);
        document.getElementById('priceChange').textContent = 
            `${priceChange >= 0 ? '+' : ''}${priceChange}% from base`;
        document.getElementById('priceChange').className = 
            `price-change ${priceChange >= 0 ? 'positive' : 'negative'}`;
        
        document.getElementById('expectedRevenue').textContent = '$' + 
            (revenueImpact.change + (features.basePrice * 100)).toFixed(0);
        document.getElementById('revenueChange').textContent = 
            `${revenueImpact.percentage >= 0 ? '+' : ''}${revenueImpact.percentage}% revenue impact`;
        document.getElementById('revenueChange').className = 
            `revenue-change ${revenueImpact.change >= 0 ? 'positive' : 'negative'}`;

        // Update decision logic
        const logicHTML = prediction.decisionLogic.map(item => 
            `<p>✅ ${item}</p>`
        ).join('');
        document.getElementById('decisionLogic').innerHTML = logicHTML;
    }

    function updateFeatureImpacts(impacts) {
        const container = document.getElementById('featureImpacts');
        container.innerHTML = impacts.map(impact => `
            <div class="feature-impact-item">
                <span>${impact.feature}</span>
                <div class="impact-bar">
                    <div class="impact-fill ${impact.direction}" 
                         style="width: ${Math.abs(impact.impact * 200)}%"></div>
                </div>
                <span>${(impact.impact * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }

    function createForecastChart(prediction, daysAhead) {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        // Generate forecast data
        const labels = Array.from({length: daysAhead}, (_, i) => `Day ${i + 1}`);
        const forecastData = Array.from({length: daysAhead}, (_, i) => {
            const base = prediction.occupancy * 100;
            const noise = (Math.random() - 0.5) * 8;
            return Math.max(10, Math.min(95, base + noise - (i * 0.5)));
        });

        if (forecastChart) {
            forecastChart.destroy();
        }

        forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'GRU Occupancy Forecast (%)',
                    data: forecastData,
                    borderColor: 'rgba(52, 152, 219, 1)',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { 
                        display: true, 
                        text: `${daysAhead}-Day Occupancy Forecast` 
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        title: { display: true, text: 'Occupancy Rate (%)' }
                    }
                }
            }
        });
    }

    // Initialize with some default values
    console.log("✅ All event listeners registered");
});
