// SmartStay Hotel Pricing Optimization - Main JavaScript
class SmartStayApp {
    constructor() {
        this.forecastChart = null;
        this.currentPrediction = null;
        this.initializeApp();
    }

    initializeApp() {
        // Initialize all components when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {
            this.initializeChart();
            this.setupEventListeners();
            this.initializeAnimations();
            this.updateForecast(); // Initial update
            
            console.log('🏨 SmartStay App Initialized');
        });
    }

    initializeChart() {
        const ctx = document.getElementById('forecastChart');
        if (!ctx) return;

        this.forecastChart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                datasets: [{
                    label: 'Predicted Occupancy %',
                    data: [65, 68, 72, 75, 82, 95, 88],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y',
                    pointBackgroundColor: '#667eea',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }, {
                    label: 'Recommended Price ($)',
                    data: [159, 162, 168, 175, 189, 229, 209],
                    borderColor: '#764ba2',
                    backgroundColor: 'rgba(118, 75, 162, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y1',
                    pointBackgroundColor: '#764ba2',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Occupancy %',
                            color: '#667eea',
                            font: {
                                weight: 'bold'
                            }
                        },
                        min: 0,
                        max: 100,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)',
                        },
                        ticks: {
                            color: '#667eea',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Price ($)',
                            color: '#764ba2',
                            font: {
                                weight: 'bold'
                            }
                        },
                        min: 100,
                        max: 250,
                        grid: {
                            drawOnChartArea: false,
                        },
                        ticks: {
                            color: '#764ba2',
                            callback: function(value) {
                                return '$' + value;
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)',
                        },
                        ticks: {
                            color: '#4a5568'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#2d3748',
                            font: {
                                size: 12,
                                weight: 'bold'
                            },
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#2d3748',
                        bodyColor: '#4a5568',
                        borderColor: '#e2e8f0',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    if (context.dataset.yAxisID === 'y1') {
                                        label += '$' + context.parsed.y;
                                    } else {
                                        label += context.parsed.y + '%';
                                    }
                                }
                                return label;
                            }
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    setupEventListeners() {
        // Slider event listeners
        const sliders = ['demandSlider', 'seasonSlider', 'competitionSlider'];
        sliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            if (slider) {
                slider.addEventListener('input', () => this.updateForecast());
                // Add visual feedback on interaction
                slider.addEventListener('mousedown', () => slider.classList.add('active'));
                slider.addEventListener('mouseup', () => slider.classList.remove('active'));
                slider.addEventListener('touchstart', () => slider.classList.add('active'));
                slider.addEventListener('touchend', () => slider.classList.remove('active'));
            }
        });

        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Navbar scroll effect
        window.addEventListener('scroll', () => {
            const navbar = document.querySelector('.navbar');
            if (navbar) {
                if (window.scrollY > 100) {
                    navbar.style.background = 'rgba(255, 255, 255, 0.98)';
                    navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.1)';
                } else {
                    navbar.style.background = 'rgba(255, 255, 255, 0.95)';
                    navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
                }
            }
        });

        // Keyboard navigation for sliders
        document.addEventListener('keydown', (e) => {
            const activeElement = document.activeElement;
            if (activeElement.type === 'range') {
                const step = 5;
                if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
                    activeElement.value = Math.min(parseInt(activeElement.max), parseInt(activeElement.value) + step);
                    this.updateForecast();
                } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
                    activeElement.value = Math.max(parseInt(activeElement.min), parseInt(activeElement.value) - step);
                    this.updateForecast();
                }
            }
        });
    }

    initializeAnimations() {
        // Intersection Observer for fade-in animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        // Observe all sections and cards
        document.querySelectorAll('.section, .card, .tech-card, .value-card').forEach(el => {
            el.classList.add('fade-in');
            observer.observe(el);
        });
    }

    // Smart pricing algorithm
    calculateOptimalPrice(demand, season, competition) {
        // Base pricing model parameters
        const basePrice = 120;
        const maxPrice = 250;
        const minPrice = 80;
        
        // Demand factor (0.5 to 2.0) - exponential growth for high demand
        const demandFactor = 0.5 + Math.pow(demand / 100, 1.5) * 1.5;
        
        // Season factor (0.8 to 1.5) - sigmoid function for smooth transitions
        const seasonFactor = 0.8 + (0.7 / (1 + Math.exp(-0.1 * (season - 50))));
        
        // Competition adjustment with market positioning
        const competitionRatio = competition / basePrice;
        let competitionAdjustment;
        if (competitionRatio < 0.8) {
            // Lower competition - premium positioning
            competitionAdjustment = 1.1;
        } else if (competitionRatio > 1.2) {
            // Higher competition - value positioning
            competitionAdjustment = 0.9;
        } else {
            // Similar competition - match with slight premium
            competitionAdjustment = 1.0 + (competitionRatio - 1.0) * 0.2;
        }
        
        // Calculate optimal price with market elasticity consideration
        let optimalPrice = basePrice * demandFactor * seasonFactor * competitionAdjustment;
        
        // Apply psychological pricing (ending with 9)
        optimalPrice = Math.round(optimalPrice - 1) + 0.99;
        
        // Apply constraints
        optimalPrice = Math.max(minPrice, Math.min(maxPrice, optimalPrice));
        
        return optimalPrice;
    }

    calculateOccupancy(demand, season, price) {
        const baseOccupancy = 50;
        const demandEffect = (demand / 100) * 40;
        const seasonEffect = (season / 100) * 20;
        
        // Price elasticity effect (higher prices reduce demand)
        const priceElasticity = -0.015; // 1.5% reduction per $10 above base
        const basePrice = 120;
        const priceEffect = (price - basePrice) * priceElasticity * 10;
        
        let occupancy = baseOccupancy + demandEffect + seasonEffect + priceEffect;
        
        // Ensure realistic bounds
        return Math.max(10, Math.min(95, occupancy));
    }

    updateForecast() {
        const demandValue = parseInt(document.getElementById('demandSlider').value);
        const seasonValue = parseInt(document.getElementById('seasonSlider').value);
        const competitionValue = parseInt(document.getElementById('competitionSlider').value);
        
        // Update slider value displays with animation
        this.animateValueChange('demandValue', `${demandValue}%`);
        this.animateValueChange('seasonValue', `${seasonValue}%`);
        this.animateValueChange('competitionValue', `$${competitionValue}`);
        
        // Calculate new values using the pricing model
        const price = this.calculateOptimalPrice(demandValue, seasonValue, competitionValue);
        const occupancy = this.calculateOccupancy(demandValue, seasonValue, price);
        
        // Calculate revenue impact
        const rooms = 100; // assuming 100 rooms
        const weeklyRevenue = Math.round(rooms * (occupancy / 100) * price * 7);
        const fixedPriceRevenue = Math.round(rooms * (occupancy / 100) * 150 * 7); // fixed at $150
        const revenueIncrease = Math.round(((weeklyRevenue - fixedPriceRevenue) / fixedPriceRevenue) * 100);
        
        // Update UI with animations
        this.animateValueChange('recommendedPrice', `$${price.toFixed(0)}`);
        this.animateValueChange('occupancyRate', `${Math.round(occupancy)}%`);
        this.animateValueChange('projectedRevenue', `$${weeklyRevenue.toLocaleString()}`);
        
        const comparisonElement = document.getElementById('revenueComparison');
        if (comparisonElement) {
            if (revenueIncrease > 0) {
                comparisonElement.textContent = `+${revenueIncrease}%`;
                comparisonElement.className = 'impact-value positive';
                this.animatePositiveChange(comparisonElement);
            } else {
                comparisonElement.textContent = `${revenueIncrease}%`;
                comparisonElement.className = 'impact-value negative';
                this.animateNegativeChange(comparisonElement);
            }
        }
        
        // Update chart with new forecast data
        this.updateChartData(occupancy, price, demandValue);
        
        // Store current prediction for potential API submission
        this.currentPrediction = {
            demand: demandValue,
            season: seasonValue,
            competition: competitionValue,
            optimalPrice: price,
            expectedOccupancy: occupancy,
            weeklyRevenue: weeklyRevenue,
            revenueIncrease: revenueIncrease,
            timestamp: new Date().toISOString()
        };
        
        // Log prediction for debugging
        console.log('📊 Prediction Update:', this.currentPrediction);
    }

    animateValueChange(elementId, newValue) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        element.style.transform = 'scale(1.1)';
        element.style.color = '#667eea';
        
        setTimeout(() => {
            element.textContent = newValue;
            element.style.transform = 'scale(1)';
            setTimeout(() => {
                element.style.color = '';
            }, 300);
        }, 150);
    }

    animatePositiveChange(element) {
        element.style.transform = 'scale(1.2)';
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 300);
    }

    animateNegativeChange(element) {
        element.style.transform = 'scale(0.9)';
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 300);
    }

    updateChartData(occupancy, price, demand) {
        if (!this.forecastChart) return;
        
        // Generate realistic forecast data based on inputs
        const baseOccupancyData = this.generateOccupancyForecast(occupancy, demand);
        const basePriceData = this.generatePriceForecast(price, demand);
        
        // Add some realistic variation
        const occupancyData = baseOccupancyData.map(value => 
            Math.max(10, Math.min(95, value + (Math.random() * 8 - 4)))
        );
        
        const priceData = basePriceData.map(value => 
            Math.max(80, Math.min(250, value + (Math.random() * 15 - 7.5)))
        );
        
        // Update chart with smooth transition
        this.forecastChart.data.datasets[0].data = occupancyData;
        this.forecastChart.data.datasets[1].data = priceData;
        this.forecastChart.update('active');
    }

    generateOccupancyForecast(baseOccupancy, demand) {
        // Generate weekly pattern based on base occupancy and demand
        const pattern = [0.85, 0.88, 0.92, 1.0, 1.05, 1.15, 1.02];
        const demandMultiplier = 0.8 + (demand / 100) * 0.4;
        
        return pattern.map(factor => 
            Math.min(95, baseOccupancy * factor * demandMultiplier)
        );
    }

    generatePriceForecast(basePrice, demand) {
        // Generate weekly price pattern
        const pattern = [0.85, 0.88, 0.92, 1.0, 1.05, 1.2, 1.1];
        const demandMultiplier = 0.9 + (demand / 100) * 0.2;
        
        return pattern.map(factor => basePrice * factor * demandMultiplier);
    }

    // Dataset analysis functions (for future expansion)
    async analyzeDataset() {
        try {
            // This would typically make an API call to get real analysis
            // For now, we'll use simulated data
            const analysis = {
                totalBookings: 119390,
                cancellationRate: 0.37,
                averageLeadTime: 104,
                averageADR: 102,
                mostPopularMonth: 'August',
                busiestDay: 'Saturday',
                revenueByMonth: this.generateRevenueData(),
                cancellationPatterns: this.generateCancellationPatterns()
            };
            
            return analysis;
        } catch (error) {
            console.error('Error analyzing dataset:', error);
            return null;
        }
    }

    generateRevenueData() {
        // Simulate monthly revenue data
        return Array.from({length: 12}, (_, i) => ({
            month: new Date(2023, i).toLocaleString('default', { month: 'long' }),
            revenue: Math.floor(Math.random() * 500000) + 200000,
            occupancy: Math.floor(Math.random() * 40) + 50
        }));
    }

    generateCancellationPatterns() {
        return {
            byLeadTime: [
                { leadTime: '0-7 days', cancellationRate: 0.15 },
                { leadTime: '8-30 days', cancellationRate: 0.25 },
                { leadTime: '31-90 days', cancellationRate: 0.35 },
                { leadTime: '90+ days', cancellationRate: 0.45 }
            ],
            byMarketSegment: [
                { segment: 'Direct', cancellationRate: 0.20 },
                { segment: 'Corporate', cancellationRate: 0.15 },
                { segment: 'Online TA', cancellationRate: 0.40 },
                { segment: 'Offline TA/TO', cancellationRate: 0.35 }
            ]
        };
    }

    // Export current prediction (for potential download or API submission)
    exportPrediction() {
        if (!this.currentPrediction) {
            alert('No prediction data available. Please adjust the sliders first.');
            return;
        }
        
        const dataStr = JSON.stringify(this.currentPrediction, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `smartstay-prediction-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        URL.revokeObjectURL(link.href);
    }

    // Reset to default values
    resetToDefaults() {
        document.getElementById('demandSlider').value = 65;
        document.getElementById('seasonSlider').value = 70;
        document.getElementById('competitionSlider').value = 150;
        this.updateForecast();
    }
}

// Utility functions
const Utils = {
    formatCurrency: (amount) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(amount);
    },
    
    formatPercentage: (value) => {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 1,
            maximumFractionDigits: 1
        }).format(value / 100);
    },
    
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Initialize the application
const smartStayApp = new SmartStayApp();

// Make app globally available for console debugging
window.smartStayApp = smartStayApp;

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SmartStayApp, Utils };
}
