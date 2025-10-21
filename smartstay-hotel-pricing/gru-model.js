// Enhanced GRU Model Simulation with Realistic Behavior
class GRUModel {
    constructor() {
        this.isTrained = false;
        this.trainingHistory = [];
        this.featureImportance = {
            'lead_time': 0.25,
            'seasonality': 0.22,
            'is_holiday': 0.18,
            'advance_booking': 0.15,
            'previous_occupancy': 0.12,
            'weekend_stays': 0.08
        };
    }

    simulateTraining() {
        console.log("🧠 Simulating GRU model training on 119,391 records...");
        
        // Simulate training process
        for (let epoch = 1; epoch <= 5; epoch++) {
            const loss = 0.15 - (epoch * 0.02);
            const accuracy = 0.70 + (epoch * 0.05);
            this.trainingHistory.push({ epoch, loss: loss.toFixed(4), accuracy: accuracy.toFixed(4) });
        }
        
        this.isTrained = true;
        console.log("✅ GRU model training completed!");
        console.log("Training History:", this.trainingHistory);
    }

    predict(features) {
        if (!this.isTrained) {
            this.simulateTraining();
        }

        const {
            basePrice = 150,
            currentOccupancy = 65,
            season = 'medium',
            isHoliday = false,
            leadTime = 30,
            daysAhead = 7
        } = features;

        // Simulate GRU's temporal pattern recognition
        let basePrediction = currentOccupancy / 100;
        
        // Season effect (GRU learns seasonal patterns)
        const seasonEffects = { low: -0.2, medium: 0.0, high: 0.3 };
        basePrediction += seasonEffects[season] || 0;

        // Holiday effect (GRU captures event-based spikes)
        if (isHoliday) basePrediction += 0.25;

        // Lead time effect (GRU understands booking windows)
        const leadTimeEffect = this.calculateLeadTimeEffect(leadTime);
        basePrediction += leadTimeEffect;

        // Temporal decay for longer forecasts (GRU's sequence understanding)
        const temporalDecay = (daysAhead - 7) * 0.01;
        basePrediction -= temporalDecay;

        // Add some realistic noise based on model confidence
        const confidence = this.calculateConfidence(basePrediction);
        const noise = (1 - confidence) * (Math.random() - 0.5) * 0.1;
        basePrediction += noise;

        // Ensure realistic bounds
        basePrediction = Math.max(0.05, Math.min(0.95, basePrediction));

        return {
            occupancy: basePrediction,
            confidence: confidence,
            featureImpacts: this.calculateFeatureImpacts(features),
            decisionLogic: this.generateDecisionLogic(features, basePrediction)
        };
    }

    calculateLeadTimeEffect(leadTime) {
        // GRU learns non-linear relationships with lead time
        if (leadTime < 7) return 0.05;  // Last-minute bookings
        if (leadTime <= 30) return 0.15; // Optimal booking window
        if (leadTime <= 90) return 0.10; // Advanced planning
        return 0.02; // Very advanced (might have higher cancellation)
    }

    calculateConfidence(prediction) {
        // GRU is more confident around typical occupancy ranges
        const midRange = 0.6;
        const distanceFromMid = Math.abs(prediction - midRange);
        return Math.max(0.7, 1 - distanceFromMid * 2);
    }

    calculateFeatureImpacts(features) {
        const impacts = [];
        
        Object.entries(this.featureImportance).forEach(([feature, weight]) => {
            let impact = 0;
            
            switch(feature) {
                case 'lead_time':
                    impact = this.calculateLeadTimeEffect(features.leadTime) * weight * 3;
                    break;
                case 'seasonality':
                    const seasonWeight = { low: -0.6, medium: 0, high: 0.9 }[features.season] || 0;
                    impact = seasonWeight * weight;
                    break;
                case 'is_holiday':
                    impact = features.isHoliday ? 0.25 * weight * 2 : -0.05 * weight;
                    break;
                case 'advance_booking':
                    impact = (features.daysAhead / 30) * weight;
                    break;
                case 'previous_occupancy':
                    impact = (features.currentOccupancy / 100 - 0.5) * weight * 2;
                    break;
            }
            
            impacts.push({
                feature: feature.replace('_', ' '),
                impact: impact,
                direction: impact >= 0 ? 'positive' : 'negative'
            });
        });

        return impacts;
    }

    generateDecisionLogic(features, prediction) {
        const logic = [];
        const occupancyPercent = (prediction * 100).toFixed(1);

        if (prediction > 0.75) {
            logic.push(`🏨 High demand forecast (${occupancyPercent}% occupancy)`);
            logic.push(`📈 Market conditions favor premium pricing`);
            logic.push(`💰 Recommend price increase to maximize revenue`);
        } else if (prediction > 0.55) {
            logic.push(`⚖️ Moderate demand (${occupancyPercent}% occupancy)`);
            logic.push(`🎯 Balanced pricing strategy recommended`);
            logic.push(`📊 Small premium justified by market position`);
        } else {
            logic.push(`📉 Low demand period (${occupancyPercent}% occupancy)`);
            logic.push(`🎪 Competitive pricing needed to attract bookings`);
            logic.push(`💡 Discount strategy to minimize empty rooms`);
        }

        // Add feature-specific insights
        if (features.isHoliday) {
            logic.push(`🎄 Holiday period detected - high demand expected`);
        }
        
        if (features.leadTime < 7) {
            logic.push(`⏰ Short lead time - last-minute booking pattern`);
        } else if (features.leadTime > 90) {
            logic.push(`📅 Long lead time - advanced planning pattern`);
        }

        return logic;
    }

    getTrainingHistory() {
        return this.trainingHistory;
    }
}
