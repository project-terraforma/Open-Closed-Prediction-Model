import { create } from 'zustand';
import type { SearchResultType } from '@/components/SearchResults';

export interface CityAnalysisState {
    cityName: string | null;
    isAnalyzing: boolean;
    progress: number;
    results: SearchResultType[];
    error: string | null;

    // Actions
    startAnalysis: (cityName: string) => void;
    addResults: (newResults: SearchResultType[]) => void;
    setProgress: (progress: number) => void;
    finishAnalysis: () => void;
    setError: (error: string) => void;
    reset: () => void;
}

export const useCityStore = create<CityAnalysisState>((set) => ({
    cityName: null,
    isAnalyzing: false,
    progress: 0,
    results: [],
    error: null,

    startAnalysis: (cityName) => set({
        cityName,
        isAnalyzing: true,
        progress: 0,
        results: [],
        error: null
    }),

    addResults: (newResults) => set((state) => ({
        results: [...state.results, ...newResults]
    })),

    setProgress: (progress) => set({ progress }),

    finishAnalysis: () => set({ isAnalyzing: false, progress: 100 }),

    setError: (error) => set({ error, isAnalyzing: false }),

    reset: () => set({
        cityName: null,
        isAnalyzing: false,
        progress: 0,
        results: [],
        error: null
    }),
}));
