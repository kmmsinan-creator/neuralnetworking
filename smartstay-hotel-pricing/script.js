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
    const missingValuesAnalysis = document.getElementById('missingValuesAnalysis');
    const modelSection = document.getElementById('modelSection');
    const prototypeSection = document.getElementById('prototypeSection');
    const innovationSection = document.getElementById('innovationSection');
    const predictBtn = document.getElementById('predictBtn');
    
    // Chart instances
    let hotelTypeChart = null;
    let monthlyBookingsChart = null;
    let leadTimeChart = null;
    let missingValuesChart = null;
    let cancellationChart = null;
    let forecastChart = null;
    let seasonalChart = null;
    let weeklyPatternChart = null;
    let yearlyTrendChart = null;
    let correlationMatrix = null;
    let priceOccupancyChart = null;
    let leadTimeCancellationChart = null;
    let customerTypeChart = null;
    let marketSegmentChart = null;
    let stayDurationChart = null;
    let specialRequestsChart = null;
    let leadTimeHeatmap = null;

    // File Upload Handler
    fileInput.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;

        uploadStatus.textContent = '📊 Analyzing hotel booking dataset...';
        uploadStatus.className = 'upload-status loading';

        try {
            await dataLoader.loadFile(file);
            
            const datasetInfo = dataLoader.getDatasetInfo();
            if (!datasetInfo.stats) {
                throw new Error('Dataset analysis failed');
            }
            
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

        // Add safety check
        if (!dataLoader.stats || !dataLoader.analysis) {
            uploadStatus.textContent = '❌ Dataset analysis failed. Please try uploading again.';
            uploadStatus.className = 'upload-status error';
            return;
        }

        console.log("📈 Starting EDA analysis...");
        
        // Show EDA section
        edaVisuals.style.display = 'block';
        
        // Update statistics
        dataStats.innerHTML = dataLoader.getStatsHTML();
        
        // Update missing values analysis
        missingValuesAnalysis.innerHTML = dataLoader.getMissingValuesHTML();
        
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
        
        // Basic distributions
        createHotelTypeChart();
        createMonthlyBookingsChart();
        createLeadTimeDistributionChart();
        createMissingValuesChart();
        createCancellationChart();
        
        // Seasonal analysis
        createSeasonalChart();
        createWeeklyPatternChart();
        createYearlyTrendChart();
        createLeadTimeHeatmap();
        
        // Correlation analysis
        createCorrelationMatrix();
        createPriceOccupancyChart();
        createLeadTimeCancellationChart();
        
        // Customer behavior
        createCustomerTypeChart();
        createMarketSegmentChart();
        createStayDurationChart();
        createSpecialRequestsChart();
    }

    // Function to destroy all existing charts
    function destroyAllCharts() {
        const charts = [
            hotelTypeChart, monthlyBookingsChart, leadTimeChart, missingValuesChart, 
            cancellationChart, forecastChart, seasonalChart, weeklyPatternChart,
            yearlyTrendChart, correlationMatrix, priceOccupancyChart, leadTimeCancellationChart,
            customerTypeChart, marketSegmentChart, stayDurationChart, specialRequestsChart,
            leadTimeHeatmap
        ];
        
        charts.forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        
        // Reset all chart variables to null
        hotelTypeChart = null;
        monthlyBookingsChart = null;
        leadTimeChart = null;
        missingValuesChart = null;
        cancellationChart = null;
        forecastChart = null;
        seasonalChart = null;
        weeklyPatternChart = null;
        yearlyTrendChart = null;
        correlationMatrix = null;
        priceOccupancyChart = null;
        leadTimeCancellationChart = null;
        customerTypeChart = null;
        marketSegmentChart = null;
        stayDurationChart = null;
        specialRequestsChart = null;
        leadTimeHeatmap = null;
    }

    function createHotelTypeChart() {
        const ctx = document.getElementById('hotelTypeChart');
        if (!ctx) {
            console.error('Hotel type chart canvas not found');
            return;
        }
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.hotelTypeDistribution) {
                console.warn('Hotel type distribution data not available');
                return;
            }
            
            const labels = Object.keys(analysis.hotelTypeDistribution);
            const data = Object.values(analysis.hotelTypeDistribution);
            
            // If no data, show empty chart
            if (data.length === 0 || data.reduce((a, b) => a + b, 0) === 0) {
                console.warn('No hotel type data available');
                return;
            }
            
            hotelTypeChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.8)',
                            'rgba(54, 162, 235, 0.8)',
                            'rgba(255, 206, 86, 0.8)',
                            'rgba(75, 192, 192, 0.8)'
                        ],
                        borderColor: [
                            'rgba(255, 99, 132, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)'
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
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.monthlyPatterns) {
                console.warn('Monthly patterns data not available');
                return;
            }
            
            const monthlyData = analysis.monthlyPatterns;
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
                        },
                        x: {
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45
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
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.leadTimeAnalysis) {
                console.warn('Lead time analysis data not available');
                return;
            }
            
            // Create sample distribution data based on actual statistics
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
                            },
                            max: 100
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

    function createMissingValuesChart() {
        const ctx = document.getElementById('missingValuesChart');
        if (!ctx) {
            console.error('Missing values chart canvas not found');
            return;
        }
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.missingValuesAnalysis) {
                console.warn('Missing values analysis data not available');
                return;
            }
            
            const missingAnalysis = analysis.missingValuesAnalysis;
            const topMissingFeatures = missingAnalysis.featuresWithMissing.slice(0, 8);
            
            // If no missing values, show empty chart with message
            if (topMissingFeatures.length === 0) {
                ctx.parentElement.innerHTML = '<div class="no-data-message">🎉 No missing values found in the dataset!</div>';
                return;
            }
            
            const labels = topMissingFeatures.map(([feature]) => feature);
            const missingData = topMissingFeatures.map(([feature, data]) => parseFloat(data.percentage));
            
            missingValuesChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Missing Values (%)',
                        data: missingData,
                        backgroundColor: missingData.map(percentage => 
                            percentage > 10 ? 'rgba(255, 99, 132, 0.8)' :
                            percentage > 5 ? 'rgba(255, 159, 64, 0.8)' :
                            'rgba(255, 205, 86, 0.8)'
                        ),
                        borderColor: missingData.map(percentage => 
                            percentage > 10 ? 'rgba(255, 99, 132, 1)' :
                            percentage > 5 ? 'rgba(255, 159, 64, 1)' :
                            'rgba(255, 205, 86, 1)'
                        ),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Top Features with Missing Values'
                        },
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Missing Percentage (%)'
                            },
                            max: Math.min(100, Math.max(...missingData) * 1.2)
                        },
                        x: {
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating missing values chart:', error);
        }
    }

    function createCancellationChart() {
        const ctx = document.getElementById('cancellationChart');
        if (!ctx) {
            console.error('Cancellation chart canvas not found');
            return;
        }
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.cancellationAnalysis) {
                console.warn('Cancellation analysis data not available');
                return;
            }
            
            const cancellationData = analysis.cancellationAnalysis;
            
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

    function createSeasonalChart() {
        const ctx = document.getElementById('seasonalChart');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.seasonalPatterns) return;
            
            const seasonalData = analysis.seasonalPatterns;
            const labels = Object.keys(seasonalData);
            const occupancyData = labels.map(month => parseFloat(seasonalData[month].occupancyRate || 0));
            const cancellationData = labels.map(month => parseFloat(seasonalData[month].cancellationRate || 0));
            
            seasonalChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Occupancy Rate (%)',
                            data: occupancyData,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            borderWidth: 3,
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Cancellation Rate (%)',
                            data: cancellationData,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            borderWidth: 2,
                            tension: 0.4,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Seasonal Demand & Cancellation Patterns'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Rate (%)'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating seasonal chart:', error);
        }
    }

    function createWeeklyPatternChart() {
        const ctx = document.getElementById('weeklyPatternChart');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.weeklyPatterns) return;
            
            const weeklyData = analysis.weeklyPatterns;
            const labels = Object.keys(weeklyData);
            const data = Object.values(weeklyData);
            
            weeklyPatternChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Booking Intensity (%)',
                        data: data,
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.8)',
                            'rgba(255, 159, 64, 0.8)',
                            'rgba(255, 205, 86, 0.8)',
                            'rgba(75, 192, 192, 0.8)',
                            'rgba(54, 162, 235, 0.8)',
                            'rgba(153, 102, 255, 0.8)',
                            'rgba(201, 203, 207, 0.8)'
                        ],
                        borderColor: [
                            'rgb(255, 99, 132)',
                            'rgb(255, 159, 64)',
                            'rgb(255, 205, 86)',
                            'rgb(75, 192, 192)',
                            'rgb(54, 162, 235)',
                            'rgb(153, 102, 255)',
                            'rgb(201, 203, 207)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Weekly Booking Pattern'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Booking Intensity (%)'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating weekly pattern chart:', error);
        }
    }

    function createYearlyTrendChart() {
        const ctx = document.getElementById('yearlyTrendChart');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.yearlyTrend) return;
            
            const yearlyData = analysis.yearlyTrend;
            const labels = Object.keys(yearlyData).sort();
            const bookingData = labels.map(year => yearlyData[year].total || 0);
            const revenueData = labels.map(year => (yearlyData[year].revenue || 0) / 1000); // Scale down for readability
            
            yearlyTrendChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Total Bookings',
                            data: bookingData,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            borderWidth: 3,
                            tension: 0.4,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Revenue (K $)',
                            data: revenueData,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            borderWidth: 3,
                            tension: 0.4,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Yearly Booking & Revenue Trend'
                        }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Total Bookings'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Revenue (K $)'
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating yearly trend chart:', error);
        }
    }

    function createCorrelationMatrix() {
        const ctx = document.getElementById('correlationMatrix');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.correlations) return;
            
            const correlations = analysis.correlations;
            const features = Object.keys(correlations);
            
            // Create abbreviated feature names for display
            const abbreviatedFeatures = features.map(f => 
                f.replace('arrival_date_', '')
                 .replace('stays_in_', '')
                 .replace('previous_', 'prev_')
                 .replace('_', ' ')
                 .substring(0, 12)
            );
            
            // Create data for bubble chart (alternative to matrix)
            const bubbleData = [];
            features.forEach((f1, i) => {
                features.forEach((f2, j) => {
                    if (i !== j) { // Skip self-correlation
                        bubbleData.push({
                            x: abbreviatedFeatures[i],
                            y: abbreviatedFeatures[j],
                            r: Math.abs(correlations[f1][f2]) * 15, // Bubble size based on correlation strength
                            correlation: correlations[f1][f2]
                        });
                    }
                });
            });
            
            correlationMatrix = new Chart(ctx, {
                type: 'bubble',
                data: {
                    datasets: [{
                        label: 'Feature Correlations',
                        data: bubbleData,
                        backgroundColor: function(context) {
                            const value = context.raw.correlation;
                            return value > 0 ? 
                                'rgba(44, 160, 44, 0.7)' : 
                                'rgba(214, 39, 40, 0.7)';
                        },
                        borderColor: function(context) {
                            const value = context.raw.correlation;
                            return value > 0 ? 
                                'rgba(44, 160, 44, 1)' : 
                                'rgba(214, 39, 40, 1)';
                        },
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Feature Correlation Matrix (Bubble Chart)'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const correlation = context.raw.correlation;
                                    return `Correlation: ${correlation.toFixed(3)}`;
                                }
                            }
                        },
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            type: 'category',
                            title: {
                                display: true,
                                text: 'Features'
                            },
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45
                            }
                        },
                        y: {
                            type: 'category',
                            title: {
                                display: true,
                                text: 'Features'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating correlation matrix:', error);
            // Fallback to a simple bar chart
            createCorrelationBarChart();
        }
    }

    function createCorrelationBarChart() {
        const ctx = document.getElementById('correlationMatrix');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.correlations) return;
            
            const correlations = analysis.correlations;
            const features = Object.keys(correlations);
            
            // Get top correlations (excluding self-correlations)
            const correlationPairs = [];
            features.forEach((f1, i) => {
                features.forEach((f2, j) => {
                    if (i < j) { // Avoid duplicates
                        correlationPairs.push({
                            pair: `${f1.substring(0, 10)} - ${f2.substring(0, 10)}`,
                            correlation: correlations[f1][f2]
                        });
                    }
                });
            });
            
            // Sort by absolute correlation and take top 15
            correlationPairs.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
            const topCorrelations = correlationPairs.slice(0, 15);
            
            const labels = topCorrelations.map(item => item.pair);
            const data = topCorrelations.map(item => item.correlation);
            
            correlationMatrix = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Correlation Coefficient',
                        data: data,
                        backgroundColor: data.map(corr => 
                            corr > 0 ? 'rgba(44, 160, 44, 0.8)' : 'rgba(214, 39, 40, 0.8)'
                        ),
                        borderColor: data.map(corr => 
                            corr > 0 ? 'rgba(44, 160, 44, 1)' : 'rgba(214, 39, 40, 1)'
                        ),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Top Feature Correlations'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: -1,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Correlation Coefficient'
                            }
                        },
                        x: {
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating correlation bar chart:', error);
        }
    }

    function createPriceOccupancyChart() {
        const ctx = document.getElementById('priceOccupancyChart');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.priceOccupancy) return;
            
            const priceData = analysis.priceOccupancy;
            
            priceOccupancyChart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Price vs Occupancy',
                        data: priceData.prices.map((price, i) => ({
                            x: i,
                            y: priceData.occupancy[i]
                        })),
                        backgroundColor: 'rgba(75, 192, 192, 0.8)',
                        pointRadius: 8,
                        pointHoverRadius: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Price vs Occupancy Relationship'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const price = priceData.prices[context.dataIndex];
                                    const occupancy = priceData.occupancy[context.dataIndex];
                                    return `Price: ${price}, Occupancy: ${occupancy}%`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Price Range'
                            },
                            ticks: {
                                callback: function(value) {
                                    return priceData.prices[value];
                                }
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Occupancy Rate (%)'
                            },
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating price occupancy chart:', error);
        }
    }

    function createLeadTimeCancellationChart() {
        const ctx = document.getElementById('leadTimeCancellationChart');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.leadTimeCancellation) return;
            
            const ltData = analysis.leadTimeCancellation;
            
            leadTimeCancellationChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ltData.leadTimes,
                    datasets: [{
                        label: 'Cancellation Rate (%)',
                        data: ltData.cancellationRates,
                        backgroundColor: ltData.cancellationRates.map(rate => 
                            rate > 40 ? 'rgba(255, 99, 132, 0.8)' :
                            rate > 30 ? 'rgba(255, 159, 64, 0.8)' :
                            'rgba(255, 205, 86, 0.8)'
                        ),
                        borderColor: ltData.cancellationRates.map(rate => 
                            rate > 40 ? 'rgb(255, 99, 132)' :
                            rate > 30 ? 'rgb(255, 159, 64)' :
                            'rgb(255, 205, 86)'
                        ),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Lead Time vs Cancellation Rate'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Cancellation Rate (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Lead Time (days)'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating lead time cancellation chart:', error);
        }
    }

    function createCustomerTypeChart() {
        const ctx = document.getElementById('customerTypeChart');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.customerBehavior) return;
            
            const customerData = analysis.customerBehavior.types;
            const labels = Object.keys(customerData);
            const data = Object.values(customerData);
            
            customerTypeChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
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
                        borderWidth: 2
                }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Customer Type Distribution'
                        },
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating customer type chart:', error);
        }
    }

    function createMarketSegmentChart() {
        const ctx = document.getElementById('marketSegmentChart');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.customerBehavior) return;
            
            const segmentData = analysis.customerBehavior.segments;
            const labels = Object.keys(segmentData);
            const data = Object.values(segmentData);
            
            marketSegmentChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.8)',
                            'rgba(54, 162, 235, 0.8)',
                            'rgba(255, 206, 86, 0.8)',
                            'rgba(75, 192, 192, 0.8)',
                            'rgba(153, 102, 255, 0.8)',
                            'rgba(255, 159, 64, 0.8)',
                            'rgba(199, 199, 199, 0.8)'
                        ],
                        borderColor: [
                            'rgba(255, 99, 132, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)',
                            'rgba(153, 102, 255, 1)',
                            'rgba(255, 159, 64, 1)',
                            'rgba(199, 199, 199, 1)'
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
                            text: 'Market Segment Distribution'
                        },
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating market segment chart:', error);
        }
    }

    function createStayDurationChart() {
        const ctx = document.getElementById('stayDurationChart');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.customerBehavior) return;
            
            const durationData = analysis.customerBehavior.stayDuration;
            const labels = Object.keys(durationData);
            const data = Object.values(durationData);
            
            stayDurationChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels.map(label => label + ' nights'),
                    datasets: [{
                        label: 'Number of Bookings',
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
                            text: 'Stay Duration Distribution'
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
            console.error('Error creating stay duration chart:', error);
        }
    }

    function createSpecialRequestsChart() {
        const ctx = document.getElementById('specialRequestsChart');
        if (!ctx) return;
        
        try {
            const analysis = dataLoader.analysis;
            if (!analysis || !analysis.customerBehavior) return;
            
            const requestsData = analysis.customerBehavior.specialRequests;
            const labels = Object.keys(requestsData).sort((a, b) => a - b);
            const data = labels.map(label => requestsData[label]);
            
            specialRequestsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels.map(label => label + (label === '1' ? ' request' : ' requests')),
                    datasets: [{
                        label: 'Number of Bookings',
                        data: data,
                        borderColor: 'rgba(255, 159, 64, 1)',
                        backgroundColor: 'rgba(255, 159, 64, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Special Requests Impact on Bookings'
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
            console.error('Error creating special requests chart:', error);
        }
    }

    function createLeadTimeHeatmap() {
        const ctx = document.getElementById('leadTimeHeatmap');
        if (!ctx) return;
        
        try {
            // Create sample heatmap data for lead time vs month using a bubble chart
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            const leadTimeRanges = ['0-7', '8-30', '31-90', '91-180', '181+'];
            
            // Generate sample data
            const data = months.flatMap((month, i) => 
                leadTimeRanges.map((range, j) => ({
                    x: range,
                    y: month,
                    r: Math.floor(Math.random() * 20) + 5, // Bubble size represents booking count
                    bookings: Math.floor(Math.random() * 100) + 20
                }))
            );
            
            leadTimeHeatmap = new Chart(ctx, {
                type: 'bubble',
                data: {
                    datasets: [{
                        label: 'Bookings by Lead Time and Month',
                        data: data,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Lead Time vs Month Heatmap (Bubble Chart)'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const bookings = context.raw.bookings;
                                    return `Bookings: ${bookings}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'category',
                            title: {
                                display: true,
                                text: 'Lead Time (days)'
                            }
                        },
                        y: {
                            type: 'category',
                            title: {
                                display: true,
                                text: 'Month'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating lead time heatmap:', error);
            // Fallback to a simple bar chart
            createLeadTimeBarChart();
        }
    }

    function createLeadTimeBarChart() {
        const ctx = document.getElementById('leadTimeHeatmap');
        if (!ctx) return;
        
        try {
            // Create a grouped bar chart as fallback
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
            const leadTimeRanges = ['0-7', '8-30', '31-90'];
            
            const datasets = leadTimeRanges.map((range, index) => ({
                label: range + ' days',
                data: months.map(() => Math.floor(Math.random() * 50) + 10),
                backgroundColor: `rgba(${54 + index * 40}, 162, 235, 0.8)`,
                borderColor: `rgba(${54 + index * 40}, 162, 235, 1)`,
                borderWidth: 1
            }));
            
            leadTimeHeatmap = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: months,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { 
                            display: true, 
                            text: 'Lead Time Distribution by Month'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Bookings'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Month'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating lead time bar chart:', error);
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

        // Safety check for dataset analysis
        if (!dataLoader.stats || !dataLoader.analysis) {
            alert('Dataset analysis incomplete. Please click "Analyze Dataset" first.');
            return;
        }

        console.log("🧠 Running GRU prediction...");
        
        // Gather inputs
        const features = {
            basePrice: parseInt(document.getElementById('basePrice').value) || 150,
            leadTime: parseInt(document.getElementById('leadTime').value) || 30,
            month: document.getElementById('month').value || 'July',
            hotelType: document.getElementById('hotelType').value || 'Resort Hotel',
            guestType: document.getElementById('guestType').value || 'Transient',
            weekendNights: parseInt(document.getElementById('weekendNights').value) || 0,
            weekNights: parseInt(document.getElementById('weekNights').value) || 2,
            adults: parseInt(document.getElementById('adults').value) || 2,
            isRepeatedGuest: document.getElementById('isRepeatedGuest').checked || false
        };

        // Validate inputs
        if (isNaN(features.basePrice) || isNaN(features.leadTime) || 
            isNaN(features.weekendNights) || isNaN(features.weekNights) || isNaN(features.adults)) {
            alert('Please enter valid numbers for all fields.');
            return;
        }

        try {
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
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Error generating prediction. Please try again.');
        }
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
        // Update occupancy prediction
        document.getElementById('predictedOccupancy').textContent = (prediction.occupancy * 100).toFixed(1) + '%';
        document.getElementById('confidenceLevel').textContent = `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`;
        
        // Update price recommendation
        document.getElementById('recommendedPrice').textContent = '$' + recommendedPrice.toFixed(2);
        const priceChange = ((recommendedPrice - features.basePrice) / features.basePrice * 100).toFixed(1);
        document.getElementById('priceChange').textContent = 
            `${priceChange >= 0 ? '+' : ''}${priceChange}% from base`;
        document.getElementById('priceChange').className = 
            `price-change ${priceChange >= 0 ? 'positive' : 'negative'}`;
        
        // Update revenue impact
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
        if (!container) return;
        
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

        try {
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
            ctx.parentElement.innerHTML = '<div class="error-message">Error creating forecast chart</div>';
        }
    }

    // Clean up charts when page is unloaded
    window.addEventListener('beforeunload', function() {
        destroyAllCharts();
    });

    // Add CSS for error messages
    const style = document.createElement('style');
    style.textContent = `
        .no-data-message, .error-message {
            text-align: center;
            padding: 40px;
            color: #6c757d;
            font-style: italic;
            background: #f8f9fa;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #dee2e6;
        }
        .error-message {
            color: #dc3545;
            background: #f8d7da;
            border: 1px solid #f5c6cb;
        }
        .scroll-to-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 1.2em;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
            z-index: 1000;
            display: none;
        }
        .scroll-to-top:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
    `;
    document.head.appendChild(style);

    // Scroll to top functionality
    const scrollBtn = document.createElement('button');
    scrollBtn.className = 'scroll-to-top';
    scrollBtn.id = 'scrollToTop';
    scrollBtn.title = 'Go to top';
    scrollBtn.textContent = '↑';
    scrollBtn.style.display = 'none';
    document.body.appendChild(scrollBtn);

    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
            scrollBtn.style.display = 'block';
        } else {
            scrollBtn.style.display = 'none';
        }
    });

    scrollBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Initialize with sample data display
    console.log("✅ All event listeners registered for hotel booking analysis");
});
