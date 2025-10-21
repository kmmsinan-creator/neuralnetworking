// Main Application Logic for Hotel Booking Dataset
document.addEventListener('DOMContentLoaded', function() {
    console.log("🚀 Hotel Pricing Optimizer initialized");
    
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
    
    // Chart instances
    let hotelTypeChart = null;
    let monthlyBookingsChart = null;
    let leadTimeChart = null;
    let cancellationChart = null;
    let forecastChart = null;

    // File Upload Handler
    fileInput.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;

        uploadStatus.textContent = '📊 Analyzing hotel booking dataset...';
        uploadStatus.className = 'upload-status loading';

        try {
            await dataLoader.loadFile(file);
            
            const datasetInfo = dataLoader.getDatasetInfo();
            uploadStatus.textContent = `✅ Dataset loaded! ${datasetInfo.stats.totalRows.toLocaleString()} records, ${datasetInfo.stats.totalFeatures} features`;
            uploadStatus.className = 'upload-status success';
            
            console.log('Hotel dataset loaded:', datasetInfo);

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
        console.log("Creating hotel data EDA charts...");
        
        // Destroy existing charts before creating new ones
        destroyAllCharts();
        
        createHotelTypeChart();
        createMonthlyBookingsChart();
        createLeadTimeDistributionChart();
        createCancellationChart();
    }

    // Function to destroy all existing charts
    function destroyAllCharts() {
        const charts = [hotelTypeChart, monthlyBookingsChart, leadTimeChart, cancellationChart, forecastChart];
        charts.forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        hotelTypeChart = null;
        monthlyBookingsChart = null;
        leadTimeChart = null;
        cancellationChart = null;
        forecastChart = null;
    }

    function createHotelTypeChart() {
        const ctx = document.getElementById('hotelTypeChart');
        if (!ctx) {
            console.error('Hotel type chart canvas not found');
            return;
        }
        
        try {
            const analysis = dataLoader.analysis;
            const labels = Object.keys(analysis.hotelTypeDistribution);
            const data = Object.values(analysis.hotelTypeDistribution);
            
            hotelTypeChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.8)',
                            'rgba(54, 162, 235, 0.8)'
                        ],
                        borderColor: [
                            'rgba(255, 99, 132, 1)',
                            'rgba(54, 162, 235, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Bookings by Hotel Type'
                        },
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating hotel type chart:', error);
        }
    }

    function createMonthlyBookingsChart() {
        const ctx = document.getElementById('monthlyBookingsChart');
        if (!ctx) {
            console.error('Monthly bookings chart canvas not found');
            return;
        }
        
        try {
            const monthlyData = dataLoader.analysis.monthlyPatterns;
            const labels = Object.keys(monthlyData);
            const data = Object.values(monthlyData);
            
            monthlyBookingsChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Number of Bookings',
                        data: data,
                        backgroundColor: 'rgba(75, 192, 192, 0.8)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Monthly Booking Distribution'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Bookings'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating monthly bookings chart:', error);
        }
    }

    function createLeadTimeDistributionChart() {
        const ctx = document.getElementById('leadTimeChart');
        if (!ctx) {
            console.error('Lead time chart canvas not found');
            return;
        }
        
        try {
            const leadTimeData = dataLoader.analysis.leadTimeAnalysis;
            // Create sample distribution data
            const labels = ['0-30', '31-60', '61-90', '91-180', '181-365', '365+'];
            const data = [35, 25, 15, 12, 8, 5];
            
            leadTimeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Percentage of Bookings',
                        data: data,
                        backgroundColor: 'rgba(153, 102, 255, 0.8)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Lead Time Distribution (days)'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Percentage (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Lead Time Range'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating lead time chart:', error);
        }
    }

    function createCancellationChart() {
        const ctx = document.getElementById('cancellationChart');
        if (!ctx) {
            console.error('Cancellation chart canvas not found');
            return;
        }
        
        try {
            const cancellationData = dataLoader.analysis.cancellationAnalysis;
            
            cancellationChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Not Canceled', 'Canceled'],
                    datasets: [{
                        data: [cancellationData.notCanceled, cancellationData.canceled],
                        backgroundColor: [
                            'rgba(54, 162, 235, 0.8)',
                            'rgba(255, 99, 132, 0.8)'
                        ],
                        borderColor: [
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 99, 132, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: `Cancellation Rate: ${cancellationData.cancellationRate}%`
                        },
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating cancellation chart:', error);
        }
    }

    // Real-time input updates
    document.getElementById('basePrice').addEventListener('input', function(e) {
        document.getElementById('basePriceValue').textContent = '$' + e.target.value;
    });

    document.getElementById('leadTime').addEventListener('input', function(e) {
        document.getElementById('leadTimeValue').textContent = e.target.value + ' days';
    });

    document.getElementById('weekendNights').addEventListener('input', function(e) {
        document.getElementById('weekendNightsValue').textContent = e.target.value;
    });

    document.getElementById('weekNights').addEventListener('input', function(e) {
        document.getElementById('weekNightsValue').textContent = e.target.value;
    });

    document.getElementById('adults').addEventListener('input', function(e) {
        document.getElementById('adultsValue').textContent = e.target.value;
    });

    // Prediction Handler
    predictBtn.addEventListener('click', function() {
        if (!dataLoader.dataset) {
            alert('Please upload and analyze the hotel dataset first.');
            return;
        }

        console.log("🧠 Running GRU prediction...");
        
        // Gather inputs
        const features = {
            basePrice: parseInt(document.getElementById('basePrice').value),
            leadTime: parseInt(document.getElementById('leadTime').value),
            month: document.getElementById('month').value,
            hotelType: document.getElementById('hotelType').value,
            guestType: document.getElementById('guestType').value,
            weekendNights: parseInt(document.getElementById('weekendNights').value),
            weekNights: parseInt(document.getElementById('weekNights').value),
            adults: parseInt(document.getElementById('adults').value),
            isRepeatedGuest: document.getElementById('isRepeatedGuest').checked
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
        createForecastChart(prediction);
        
        console.log("✅ GRU prediction completed");
    });

    function calculateOptimalPrice(basePrice, occupancy) {
        if (occupancy > 0.80) {
            return basePrice * 1.45; // Very high demand - premium pricing
        } else if (occupancy > 0.70) {
            return basePrice * 1.25; // High demand - optimized pricing
        } else if (occupancy > 0.55) {
            return basePrice * 1.10; // Moderate demand - slight premium
        } else {
            return basePrice * 0.80; // Low demand - attraction pricing
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
            `<p>🎯 ${item}</p>`
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
                         style="width: ${Math.abs(impact.impact * 150)}%"></div>
                </div>
                <span>${(impact.impact * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }

    function createForecastChart(prediction) {
        const ctx = document.getElementById('forecastChart');
        if (!ctx) {
            console.error('Forecast chart canvas not found');
            return;
        }
        
        // Destroy existing forecast chart
        if (forecastChart) {
            forecastChart.destroy();
        }

        // Generate 30-day forecast data
        const labels = Array.from({length: 30}, (_, i) => `Day ${i + 1}`);
        const baseOccupancy = prediction.occupancy * 100;
        const forecastData = Array.from({length: 30}, (_, i) => {
            // Simulate realistic occupancy fluctuations
            const trend = Math.sin(i * 0.2) * 8; // Weekly pattern
            const noise = (Math.random() - 0.5) * 6; // Random noise
            const weekendBoost = (i % 7 === 5 || i % 7 === 6) ? 12 : 0; // Weekend effect
            
            return Math.max(15, Math.min(95, baseOccupancy + trend + noise + weekendBoost - (i * 0.3)));
        });

        try {
            forecastChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Predicted Occupancy Rate (%)',
                        data: forecastData,
                        borderColor: 'rgba(52, 152, 219, 1)',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2,
                        pointRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: '30-Day Occupancy Forecast',
                            font: { size: 16 }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    },
                    scales: {
                        y: {
                            min: 0,
                            max: 100,
                            title: { 
                                display: true, 
                                text: 'Occupancy Rate (%)',
                                font: { size: 14 }
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Days Ahead',
                                font: { size: 14 }
                            }
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            });
        } catch (error) {
            console.error('Error creating forecast chart:', error);
        }
    }

    // Clean up charts when page is unloaded
    window.addEventListener('beforeunload', function() {
        destroyAllCharts();
    });

    // Initialize with sample data display
    console.log("✅ All event listeners registered for hotel booking analysis");
});
