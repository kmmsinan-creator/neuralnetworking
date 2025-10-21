// Enhanced data loader with file upload capability
class DataLoader {
    constructor() {
        this.dataset = null;
        this.features = null;
        this.stats = null;
    }

    async loadFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    const content = e.target.result;
                    
                    if (file.name.endsWith('.csv')) {
                        this.parseCSV(content);
                    } else if (file.name.endsWith('.json')) {
                        this.parseJSON(content);
                    } else {
                        throw new Error('Unsupported file format');
                    }
                    
                    this.analyzeDataset();
                    resolve(this.dataset);
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            
            if (file.name.endsWith('.csv')) {
                reader.readAsText(file);
            } else {
                reader.readAsText(file);
            }
        });
    }

    parseCSV(content) {
        const lines = content.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',').map(h => h.trim());
        
        const data = [];
        for (let i = 1; i < Math.min(lines.length, 100); i++) { // Limit preview to 100 rows
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index] || '';
            });
            data.push(row);
        }
        
        this.dataset = data;
        this.features = headers;
    }

    parseJSON(content) {
        const data = JSON.parse(content);
        // Handle both array of objects and nested structures
        if (Array.isArray(data)) {
            this.dataset = data.slice(0, 100); // Limit preview
            this.features = Object.keys(data[0] || {});
        } else {
            throw new Error('JSON format not supported. Expected array of objects.');
        }
    }

    analyzeDataset() {
        if (!this.dataset || this.dataset.length === 0) return;
        
        this.stats = {
            rowCount: this.dataset.length,
            featureCount: this.features.length,
            sampleFeatures: this.features.slice(0, 6), // Show first 6 features
            missingValues: this.calculateMissingValues()
        };
    }

    calculateMissingValues() {
        const missing = {};
        this.features.forEach(feature => {
            const missingCount = this.dataset.filter(row => 
                row[feature] === '' || row[feature] === null || row[feature] === undefined
            ).length;
            missing[feature] = missingCount;
        });
        return missing;
    }

    getPreviewHTML() {
        if (!this.dataset) return '';
        
        let html = `<div class="dataset-stats">
            <p><strong>Dataset loaded:</strong> ${this.stats.rowCount} rows, ${this.stats.featureCount} features</p>
            <p><strong>Key features detected:</strong> ${this.stats.sampleFeatures.join(', ')}${this.stats.featureCount > 6 ? '...' : ''}</p>
        </div>`;
        
        html += '<table class="preview-table"><thead><tr>';
        
        // Show first 6 columns for preview
        const previewFeatures = this.features.slice(0, 6);
        previewFeatures.forEach(feature => {
            html += `<th>${feature}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        // Show first 5 rows
        for (let i = 0; i < Math.min(5, this.dataset.length); i++) {
            html += '<tr>';
            previewFeatures.forEach(feature => {
                const value = this.dataset[i][feature];
                html += `<td title="${feature}: ${value}">${this.truncateValue(value)}</td>`;
            });
            html += '</tr>';
        }
        html += '</tbody></table>';
        
        return html;
    }

    truncateValue(value, maxLength = 20) {
        const str = String(value);
        return str.length > maxLength ? str.substring(0, maxLength) + '...' : str;
    }
}
