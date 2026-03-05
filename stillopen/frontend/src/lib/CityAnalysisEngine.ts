import { geocodeCity, searchPlacesByBbox } from './CitySearchService';
import { useCityStore } from '../store/CityResultsStore';

export class CityAnalysisEngine {
    private isCancelled = false;
    private maxResultsToProcess = 500; // Cap to prevent crashing browser or backend easily
    private batchSize = 50;

    public cancel() {
        this.isCancelled = true;
    }

    public async runAnalysis(cityName: string) {
        this.isCancelled = false;
        const store = useCityStore.getState();

        try {
            store.startAnalysis(cityName);

            // 1. Geocode the city to a bounding box
            const geoResult = await geocodeCity(cityName);
            if (!geoResult) {
                store.setError(`Could not find city bounds for "${cityName}".`);
                return;
            }

            console.log(`Starting analysis for ${geoResult.displayName}`, geoResult.bbox);

            // 2. Paginate through results within the bounding box
            let offset = 0;
            let totalFetched = 0;
            let hasMore = true;

            while (hasMore && !this.isCancelled && totalFetched < this.maxResultsToProcess) {
                // The backend search route already runs predict_status internally and returns 'status' and 'confidence'.
                // All we need to do is paginate through the bbox.
                const batch = await searchPlacesByBbox(geoResult.bbox, this.batchSize, offset);

                if (batch.length === 0) {
                    hasMore = false;
                    break;
                }

                store.addResults(batch);
                totalFetched += batch.length;
                offset += this.batchSize;

                // Update rough progress. (Without knowing total hits upfront, we estimate out of maxResultsToProcess)
                const pct = Math.min(99, Math.round((totalFetched / this.maxResultsToProcess) * 100));
                store.setProgress(pct);

                // Small delay to let React render the batch and not overheat the backend
                await new Promise(resolve => setTimeout(resolve, 500));
            }

            if (!this.isCancelled) {
                store.finishAnalysis();
            }

        } catch (err: unknown) {
            console.error("CityAnalysisEngine error:", err);
            store.setError(err instanceof Error ? err.message : "An unknown error occurred during analysis.");
        }
    }
}

// Singleton helper
export const cityAnalysisEngine = new CityAnalysisEngine();
