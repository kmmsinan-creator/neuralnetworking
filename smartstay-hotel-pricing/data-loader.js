// Enhanced Data Loader with EDA capabilities
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
        const headers = lines[0].split(',').map(h => h.trim());
        
        const data = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, index) => {
                // Basic data type detection and conversion
                let value = values[index] || '';
                if (!isNaN(value) && value !== '') {
                    value = Number(value);
                } else if (value.toLowerCase() === 'true' || value.toLowerCase() === 'false') {
                    value = value.toLowerCase() === 'true';
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
            basicStats: this.calculateBasicStats()
        };

        this.analysis = {
            seasonalPatterns: this.analyzeSeasonalPatterns(),
            correlation: this.analyzeCorrelations(),
            distributions: this.analyzeDistributions()
        };
    }

    getNumericalFeatures() {
        return this.features.filter(feature => {
            return this.dataset.some(row => typeof row[feature] === 'number');
        });
    }

    getCategoricalFeatures() {
        return this.features.filter(feature => {
            return this.dataset.some(row => typeof row[feature] === 'string' || typeof row[feature] === 'boolean');
        });
    }

    calculateMissingValues() {
        const missing = {};
        this.features.forEach(feature => {
            const missingCount = this.dataset.filter(row => 
                row[feature] === '' || row[feature] === null || row[feature] === undefined
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
            const values = this.dataset.map(row => row[feature]).filter(val => typeof val === 'number');
            stats[feature] = {
                mean: values.reduce((a, b) => a + b, 0) / values.length,
                min: Math.min(...values),
                max: Math.max(...values),
                std: this.calculateStandardDeviation(values)
            };
        });
        return stats;
    }

    calculateStandardDeviation(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const squareDiffs = values.map(value => Math.pow(value - mean, 2));
        return Math.sqrt(squareDiffs.reduce((a, b) => a + b, 0) / values.length);
    }

    analyzeSeasonalPatterns() {
        // Simulate seasonal analysis
        return {
            summer: { occupancy: 0.85, priceMultiplier: 1.3 },
            winter: { occupancy: 0.45, priceMultiplier: 0.8 },
            spring: { occupancy: 0.65, priceMultiplier: 1.0 },
            fall: { occupancy: 0.70, priceMultiplier: 1.1 }
        };
    }

    analyzeCorrelations() {
        // Simulate correlation analysis
        return {
            'lead_time vs occupancy': -0.45,
            'season vs occupancy': 0.72,
            'holiday vs occupancy': 0.68,
            'advance_booking vs price': 0.35
        };
    }

    analyzeDistributions() {
        const distributions = {};
        this.getNumericalFeatures().forEach(feature => {
            const values = this.dataset.map(row => row[feature]).filter(val => typeof val === 'number');
            distributions[feature] = this.createDistribution(values, 10);
        });
        return distributions;
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
        
        return {
            bins: distribution,
            labels: Array.from({length: bins}, (_, i) => 
                (min + i * binSize).toFixed(1) + '-' + (min + (i + 1) * binSize).toFixed(1)
            )
        };
    }

    getStatsHTML() {
        if (!this.stats) return '';
        
        return `
            <div class="stat-item"><strong>Total Records:</strong> ${this.stats.totalRows.toLocaleString()}</div>
            <div class="stat-item"><strong>Features:</strong> ${this.stats.totalFeatures}</div>
            <div class="stat-item"><strong>Numerical Features:</strong> ${this.stats.numericalFeatures.length}</div>
            <div class="stat-item"><strong>Categorical Features:</strong> ${this.stats.categoricalFeatures.length}</div>
            <div class="stat-item"><strong>Data Quality:</strong> ${this.calculateDataQuality()}%</div>
        `;
    }

    calculateDataQuality() {
        const totalCells = this.stats.totalRows * this.stats.totalFeatures;
        const missingCells = Object.values(this.stats.missingValues).reduce((sum, item) => sum + item.count, 0);
        return ((totalCells - missingCells) / totalCells * 100).toFixed(1);
    }
}
