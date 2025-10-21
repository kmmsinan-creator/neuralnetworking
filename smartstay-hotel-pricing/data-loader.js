// Enhanced Data Loader for Hotel Booking Dataset
class DataLoader {
    constructor() {
        this.dataset = null;
        this.features = null;
        this.stats = null;
        this.analysis = null;
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
        // Process up to 2000 rows for performance (enough for meaningful analysis)
        const maxRows = Math.min(lines.length, 2000);
        for (let i = 1; i < maxRows; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, index) => {
                let value = values[index] || '';
                // Convert to number if possible
                if (!isNaN(value) && value !== '' && value !== 'NULL') {
                    value = Number(value);
                } else if (value === 'NULL') {
                    value = null;
                }
                row[header] = value;
            });
            data.push(row);
        }
        
        this.dataset = data;
        this.features = headers;
    }

    performEDA() {
        if (!this.dataset || this.dataset.length === 0) return;
        
        this.stats = {
            totalRows: this.dataset.length,
            totalFeatures: this.features.length,
            numericalFeatures: this.getNumericalFeatures(),
            categoricalFeatures: this.getCategoricalFeatures(),
            missingValues: this.calculateMissingValues(),
            basicStats: this.calculateBasicStats(),
            bookingAnalysis: this.analyzeBookings()
        };

        this.analysis = {
            hotelTypeDistribution: this.analyzeHotelType(),
            monthlyPatterns: this.analyzeMonthlyPatterns(),
            leadTimeAnalysis: this.analyzeLeadTime(),
            cancellationAnalysis: this.analyzeCancellations()
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
                row[feature] === '' || row[feature] === null || row[feature] === undefined || row[feature] === 'NULL'
            ).length;
            missing[feature] = {
                count: missingCount,
                percentage: (missingCount / this.dataset.length * 100).toFixed(2)
            };
        });
        return missing;
    }

    calculateBasicStats() {
        const stats = {};
        this.getNumericalFeatures().forEach(feature => {
            const values = this.dataset.map(row => row[feature]).filter(val => typeof val === 'number' && !isNaN(val));
            if (values.length > 0) {
                stats[feature] = {
                    mean: (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2),
                    min: Math.min(...values),
                    max: Math.max(...values),
                    median: this.calculateMedian(values),
                    count: values.length
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
            hotelCounts[hotel] = (hotelCounts[hotel] || 0) + 1;
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
        const leadTimes = this.dataset.map(row => row.lead_time).filter(val => typeof val === 'number');
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

    createDistribution(values, bins) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binSize = (max - min) / bins;
        
        const distribution = Array(bins).fill(0);
        values.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
            distribution[binIndex]++;
        });
        
        return distribution;
    }

    getStatsHTML() {
        if (!this.stats) return '';
        
        const totalMissing = Object.values(this.stats.missingValues).reduce((sum, item) => sum + item.count, 0);
        const dataQuality = ((this.stats.totalRows * this.stats.totalFeatures - totalMissing) / 
                           (this.stats.totalRows * this.stats.totalFeatures) * 100).toFixed(1);
        
        return `
            <div class="stat-item"><strong>Total Records Analyzed:</strong> ${this.stats.totalRows.toLocaleString()}</div>
            <div class="stat-item"><strong>Total Features:</strong> ${this.stats.totalFeatures}</div>
            <div class="stat-item"><strong>Numerical Features:</strong> ${this.stats.numericalFeatures.length}</div>
            <div class="stat-item"><strong>Categorical Features:</strong> ${this.stats.categoricalFeatures.length}</div>
            <div class="stat-item"><strong>Data Quality Score:</strong> ${dataQuality}%</div>
            <div class="stat-item"><strong>Cancellation Rate:</strong> ${this.stats.bookingAnalysis.cancellationRate}%</div>
            <div class="stat-item"><strong>Average Lead Time:</strong> ${this.stats.basicStats.lead_time ? this.stats.basicStats.lead_time.mean + ' days' : 'N/A'}</div>
            <div class="stat-item"><strong>Average ADR:</strong> $${this.stats.basicStats.adr ? this.stats.basicStats.adr.mean : 'N/A'}</div>
        `;
    }

    getDatasetInfo() {
        return {
            features: this.features,
            stats: this.stats,
            analysis: this.analysis
        };
    }
}
