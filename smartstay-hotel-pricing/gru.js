// This file simulates the predictive logic of a trained GRU model.
// In reality, this would be a TensorFlow.js model loaded and executed.

class SimulatedGRU {
    predict(season, isHoliday, leadTime) {
        console.log("Simulating GRU prediction...");

        // Simulate the GRU's learning of complex temporal patterns:
        // 1. Base occupancy influenced by season
        let baseOccupancy = 0.5; // 50% base
        if (season === 'high') baseOccupancy += 0.3;
        if (season === 'low') baseOccupancy -= 0.2;

        // 2. Significant boost from holidays
        if (isHoliday) baseOccupancy += 0.25;

        // 3. Non-linear relationship with lead time (very short or very long lead times might indicate different things)
        // Let's simulate that medium-long lead times are best.
        let leadTimeEffect = 0;
        if (leadTime > 7 && leadTime <= 90) {
            leadTimeEffect = 0.15;
        } else if (leadTime > 90) {
            // Very long lead times might have lower conversion? Or higher? Let's simulate a slight drop.
            leadTimeEffect = 0.05;
        }
        // Short lead times (<=7 days) get no extra boost.

        let finalOccupancy = baseOccupancy + leadTimeEffect;

        // Ensure occupancy is between 5% and 95%
        finalOccupancy = Math.max(0.05, Math.min(0.95, finalOccupancy));

        // Add a small random noise to simulate real-world prediction variance
        finalOccupancy += (Math.random() * 0.1 - 0.05);
        finalOccupancy = Math.max(0.05, Math.min(0.95, finalOccupancy));

        return finalOccupancy;
    }
}
