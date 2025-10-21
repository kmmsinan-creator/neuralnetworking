// Enhanced Data Loader for Hotel Booking Dataset - Full Data Processing
class DataLoader {
    constructor() {
        this.dataset = null;
        this.features = null;
        this.stats = null;
        this.analysis = null;
        this.fullDataset = null;
    }

    async loadFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    const content = e.target.result;
                    this.parseCSV(content);
                    this.performEDA();
                    resolve(this.dataset);
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    parseCSV(content) {
        const lines = content.split('\n').filter(line => line.trim());
        if (lines.length === 0) throw new Error('CSV file is empty');
        
        const headers = lines[0].split(',').map(h => h.trim());
        
        const data = [];
        // Process ALL rows for complete analysis
        for (let i = 1; i < lines.length; i++) {
            const values = this.parseCSVLine(lines[i]);
            const row = {};
            headers.forEach((header, index) => {
                let value = values[index] || '';
                // Convert to number if possible
                if (!isNaN(value) && value !== '' && value !== 'NULL' && value !== 'null') {
                    value = Number(value);
                } else if (value === 'NULL' || value === 'null') {
                    value = null;
                } else if (value === '') {
                    value = null;
                }
                row[header] = value;
            });
            data.push(row);
        }
        
        this.fullDataset = data;
        // Use full dataset for analysis
        this.dataset = data;
        this.features = headers;
    }

    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        
        result.push(current.trim());
        return result;
    }

    performEDA() {
        if (!this.dataset || this.dataset.length === 0) return;
        
        // Calculate missing values first
        const missingValues = this.calculateMissingValues();
        const dataQuality = this.calculateDataQuality(missingValues);
        
        this.stats = {
            totalRows: this.dataset.length,
            totalFeatures: this.features.length,
            numericalFeatures: this.getNumericalFeatures(),
            categoricalFeatures: this.getCategoricalFeatures(),
            missingValues: missingValues,
            basicStats: this.calculateBasicStats(missingValues),
            bookingAnalysis: this.analyzeBookings(),
            dataQuality: dataQuality
        };

        this.analysis = {
            hotelTypeDistribution: this.analyzeHotelType(),
            monthlyPatterns: this.analyzeMonthlyPatterns(),
            leadTimeAnalysis: this.analyzeLeadTime(),
            cancellationAnalysis: this.analyzeCancellations(),
            missingValuesAnalysis: this.analyzeMissingValues(missingValues),
            featureDistributions: this.analyzeFeatureDistributions()
        };
    }

    getNumericalFeatures() {
        const numericalFeatures = [
            'lead_time', 'arrival_date_year', 'arrival_date_week_number',
            'arrival_date_day_of_month', 'stays_in_weekend_nights',
            'stays_in_week_nights', 'adults', 'children', 'babies',
            'previous_cancellations', 'previous_bookings_not_canceled',
            'booking_changes', 'days_in_waiting_list', 'adr',
            'required_car_parking_spaces', 'total_of_special_requests'
        ];
        return numericalFeatures.filter(feature => this.features.includes(feature));
    }

    getCategoricalFeatures() {
        const categoricalFeatures = [
            'hotel', 'is_canceled', 'arrival_date_month', 'meal', 'country',
            'market_segment', 'distribution_channel', 'is_repeated_guest',
            'reserved_room_type', 'assigned_room_type', 'deposit_type',
            'agent', 'company', 'customer_type', 'reservation_status'
        ];
        return categoricalFeatures.filter(feature => this.features.includes(feature));
    }

    calculateMissingValues() {
        const missing = {};
        this.features.forEach(feature => {
            const missingCount = this.dataset.filter(row => 
                row[feature] === '' || row[feature] === null || row[feature] === undefined || 
                row[feature] === 'NULL' || row[feature] === 'null' || 
                (typeof row[feature] === 'number' && isNaN(row[feature]))
            ).length;
            missing[feature] = {
                count: missingCount,
                percentage: (missingCount / this.dataset.length * 100).toFixed(2)
            };
        });
        return missing;
    }

    analyzeMissingValues(missingValues) {
        const featuresWithMissing = Object.entries(missingValues)
            .filter(([feature, data]) => data.count > 0)
            .sort((a, b) => b[1].count - a[1].count);

        const totalMissingCells = Object.values(missingValues).reduce((sum, item) => sum + item.count, 0);
        const totalCells = this.dataset.length * this.features.length;

        return {
            totalMissingCells: totalMissingCells,
            totalCells: totalCells,
            missingPercentage: (totalMissingCells / totalCells * 100).toFixed(2),
            featuresWithMissing: featuresWithMissing.slice(0, 10), // Top 10 features with missing values
            completeFeatures: this.features.filter(feature => missingValues[feature].count === 0)
        };
    }

    calculateDataQuality(missingValues) {
        const totalCells = this.dataset.length * this.features.length;
        const missingCells = Object.values(missingValues).reduce((sum, item) => sum + item.count, 0);
        const qualityScore = ((totalCells - missingCells) / totalCells * 100).toFixed(2);
        
        return {
            score: qualityScore,
            rating: qualityScore >= 95 ? 'Excellent' : 
                   qualityScore >= 90 ? 'Good' : 
                   qualityScore >= 80 ? 'Fair' : 'Poor',
            missingCells: missingCells,
            totalCells: totalCells
        };
    }

    calculateBasicStats(missingValues) {
        const stats = {};
        this.getNumericalFeatures().forEach(feature => {
            const values = this.dataset.map(row => row[feature]).filter(val => typeof val === 'number' && !isNaN(val));
            if (values.length > 0) {
                stats[feature] = {
                    mean: (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2),
                    min: Math.min(...values),
                    max: Math.max(...values),
                    median: this.calculateMedian(values),
                    std: this.calculateStandardDeviation(values),
                    count: values.length,
                    missing: missingValues[feature] ? missingValues[feature].count : 0
                };
            }
        });
        return stats;
    }

    calculateMedian(values) {
        const sorted = values.slice().sort((a, b) => a - b);
        const middle = Math.floor(sorted.length / 2);
        if (sorted.length % 2 === 0) {
            return ((sorted[middle - 1] + sorted[middle]) / 2).toFixed(2);
        }
        return sorted[middle].toFixed(2);
    }

    calculateStandardDeviation(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const squareDiffs = values.map(value => Math.pow(value - mean, 2));
        const variance = squareDiffs.reduce((a, b) => a + b, 0) / values.length;
        return Math.sqrt(variance).toFixed(2);
    }

    analyzeBookings() {
        const totalBookings = this.dataset.length;
        const canceledBookings = this.dataset.filter(row => row.is_canceled === 1).length;
        const resortBookings = this.dataset.filter(row => row.hotel === 'Resort Hotel').length;
        const cityBookings = this.dataset.filter(row => row.hotel === 'City Hotel').length;
        
        return {
            totalBookings,
            canceledBookings,
            cancellationRate: ((canceledBookings / totalBookings) * 100).toFixed(2),
            resortBookings,
            cityBookings,
            resortPercentage: ((resortBookings / totalBookings) * 100).toFixed(2),
            cityPercentage: ((cityBookings / totalBookings) * 100).toFixed(2)
        };
    }

    analyzeHotelType() {
        const hotelCounts = {};
        this.dataset.forEach(row => {
            const hotel = row.hotel;
            if (hotel) {
                hotelCounts[hotel] = (hotelCounts[hotel] || 0) + 1;
            }
        });
        return hotelCounts;
    }

    analyzeMonthlyPatterns() {
        const monthlyData = {};
        const months = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December'];
        
        months.forEach(month => {
            monthlyData[month] = this.dataset.filter(row => row.arrival_date_month === month).length;
        });
        
        return monthlyData;
    }

    analyzeLeadTime() {
        const leadTimes = this.dataset.map(row => row.lead_time).filter(val => typeof val === 'number' && !isNaN(val));
        if (leadTimes.length === 0) {
            return {
                average: '0',
                min: 0,
                max: 0,
                distribution: []
            };
        }
        
        return {
            average: (leadTimes.reduce((a, b) => a + b, 0) / leadTimes.length).toFixed(2),
            min: Math.min(...leadTimes),
            max: Math.max(...leadTimes),
            distribution: this.createDistribution(leadTimes, 10)
        };
    }

    analyzeCancellations() {
        const canceled = this.dataset.filter(row => row.is_canceled === 1).length;
        const notCanceled = this.dataset.filter(row => row.is_canceled === 0).length;
        
        return {
            canceled,
            notCanceled,
            cancellationRate: ((canceled / this.dataset.length) * 100).toFixed(2)
        };
    }

    analyzeFeatureDistributions() {
        const distributions = {};
        
        // Analyze key numerical features
        ['lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights'].forEach(feature => {
            if (this.features.includes(feature)) {
                const values = this.dataset.map(row => row[feature]).filter(val => typeof val === 'number' && !isNaN(val));
                if (values.length > 0) {
                    distributions[feature] = this.createDistribution(values, 8);
                }
            }
        });
        
        return distributions;
    }

    createDistribution(values, bins) {
        if (values.length === 0) {
            return {
                bins: Array(bins).fill(0),
                labels: Array.from({length: bins}, (_, i) => `Bin ${i + 1}`),
                min: 0,
                max: 0
            };
        }
        
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binSize = (max - min) / bins;
        
        const distribution = Array(bins).fill(0);
        values.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
            distribution[binIndex]++;
        });
        
        return {
            bins: distribution,
            labels: Array.from({length: bins}, (_, i) => 
                Math.round(min + i * binSize) + '-' + Math.round(min + (i + 1) * binSize)
            ),
            min: min,
            max: max
        };
    }

    getStatsHTML() {
        if (!this.stats) return '';
        
        return `
            <div class="stat-item"><strong>Total Records:</strong> ${this.stats.totalRows.toLocaleString()}</div>
            <div class="stat-item"><strong>Total Features:</strong> ${this.stats.totalFeatures}</div>
            <div class="stat-item"><strong>Numerical Features:</strong> ${this.stats.numericalFeatures.length}</div>
            <div class="stat-item"><strong>Categorical Features:</strong> ${this.stats.categoricalFeatures.length}</div>
            <div class="stat-item"><strong>Data Quality Score:</strong> ${this.stats.dataQuality.score}% (${this.stats.dataQuality.rating})</div>
            <div class="stat-item"><strong>Missing Values:</strong> ${this.stats.dataQuality.missingCells.toLocaleString()} / ${this.stats.dataQuality.totalCells.toLocaleString()} cells</div>
            <div class="stat-item"><strong>Cancellation Rate:</strong> ${this.stats.bookingAnalysis.cancellationRate}%</div>
            <div class="stat-item"><strong>Average Lead Time:</strong> ${this.stats.basicStats.lead_time ? this.stats.basicStats.lead_time.mean + ' days' : 'N/A'}</div>
            <div class="stat-item"><strong>Average ADR:</strong> $${this.stats.basicStats.adr ? this.stats.basicStats.adr.mean : 'N/A'}</div>
        `;
    }

    getMissingValuesHTML() {
        if (!this.analysis || !this.analysis.missingValuesAnalysis) return '';
        
        const missingAnalysis = this.analysis.missingValuesAnalysis;
        
        let html = `
            <div class="missing-values-section">
                <h4>Missing Values Analysis</h4>
                <div class="missing-summary">
                    <div class="missing-stat">
                        <strong>Overall Data Quality:</strong> ${this.stats.dataQuality.score}%
                    </div>
                    <div class="missing-stat">
                        <strong>Missing Cells:</strong> ${missingAnalysis.totalMissingCells.toLocaleString()} / ${missingAnalysis.totalCells.toLocaleString()}
                    </div>
                    <div class="missing-stat">
                        <strong>Complete Features:</strong> ${missingAnalysis.completeFeatures.length} / ${this.features.length}
                    </div>
                </div>
        `;
        
        if (missingAnalysis.featuresWithMissing.length > 0) {
            html += `
                <div class="missing-features">
                    <h5>Top Features with Missing Values:</h5>
                    <div class="missing-features-list">
            `;
            
            missingAnalysis.featuresWithMissing.forEach(([feature, data]) => {
                html += `
                    <div class="missing-feature-item">
                        <span class="feature-name">${feature}</span>
                        <span class="missing-count">${data.count.toLocaleString()} (${data.percentage}%)</span>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        } else {
            html += `<p>No missing values found in the dataset! 🎉</p>`;
        }
        
        html += `</div>`;
        return html;
    }

    getDatasetInfo() {
        return {
            features: this.features,
            stats: this.stats,
            analysis: this.analysis
        };
    }
}
