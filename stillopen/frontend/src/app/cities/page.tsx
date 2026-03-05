"use client";

import { useEffect, useMemo, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useCityStore } from "../../store/CityResultsStore";
import { cityAnalysisEngine } from "../../lib/CityAnalysisEngine";
import { Loader2, AlertCircle, Search, MapPin, CheckCircle, XCircle } from "lucide-react";
import dynamic from "next/dynamic";
import { useState } from "react";

const CityMap = dynamic(() => import("../../components/CityMap"), {
    ssr: false,
    loading: () => (
        <div className="w-full h-full bg-gray-100 animate-pulse flex items-center justify-center text-gray-500 font-semibold">
            Loading Map...
        </div>
    ),
});

function CitiesContent() {
    const searchParams = useSearchParams();
    const cityQuery = searchParams.get("q");
    const [cityInput, setCityInput] = useState(cityQuery || "");

    const { cityName, isAnalyzing, progress, results, error } = useCityStore();

    useEffect(() => {
        if (cityQuery) {
            setCityInput(cityQuery);
            cityAnalysisEngine.runAnalysis(cityQuery);
        }
        return () => {
            cityAnalysisEngine.cancel();
            useCityStore.getState().reset();
        };
    }, [cityQuery]);

    const stats = useMemo(() => {
        const open = results.filter(r => r.status?.toLowerCase() === "open").length;
        const closed = results.filter(r => r.status?.toLowerCase() === "closed").length;
        const unknown = results.length - open - closed;
        return { open, closed, unknown, total: results.length };
    }, [results]);

    const mapCenter = useMemo((): [number, number] | undefined => {
        const withCoords = results.filter(r => r.lat && r.lon);
        if (withCoords.length === 0) return undefined;
        const avgLat = withCoords.reduce((s, r) => s + r.lat!, 0) / withCoords.length;
        const avgLon = withCoords.reduce((s, r) => s + r.lon!, 0) / withCoords.length;
        return [avgLat, avgLon];
    }, [results]);

    const handleCitySearch = (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (cityInput.trim()) {
            window.location.href = `/cities?q=${encodeURIComponent(cityInput)}`;
        }
    };

    return (
        <div className="w-full flex-1 flex flex-col relative" style={{ height: "calc(100vh - 64px)" }}>
            {/* Full-screen map */}
            <div className="absolute inset-0">
                <CityMap results={results} center={mapCenter} />
            </div>

            {/* Floating Search Panel */}
            <div className="absolute top-4 left-4 z-[1000] w-96 max-w-[calc(100vw-2rem)]">
                <form onSubmit={handleCitySearch} className="bg-white/95 backdrop-blur-md rounded-2xl shadow-2xl border border-gray-100 overflow-hidden">
                    <div className="p-4">
                        <h2 className="text-sm font-black text-gray-900 uppercase tracking-wider mb-3 flex items-center gap-2">
                            <MapPin className="w-4 h-4 text-emerald-500" />
                            Cities Mode
                        </h2>
                        <div className="relative flex">
                            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none z-10">
                                <Search className="h-4 w-4 text-gray-400" />
                            </div>
                            <input
                                type="text"
                                className="block w-full pl-10 pr-4 py-2.5 text-sm text-gray-900 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500 transition-all placeholder-gray-400"
                                placeholder="Enter a city (e.g. Santa Cruz, CA)..."
                                value={cityInput}
                                onChange={(e) => setCityInput(e.target.value)}
                            />
                        </div>
                    </div>

                    {/* Stats */}
                    {results.length > 0 && (
                        <div className="border-t border-gray-100 px-4 py-3 grid grid-cols-3 gap-2 text-center">
                            <div className="flex flex-col items-center">
                                <div className="flex items-center gap-1 text-emerald-600">
                                    <CheckCircle className="w-3.5 h-3.5" />
                                    <span className="text-lg font-black">{stats.open}</span>
                                </div>
                                <span className="text-[10px] text-gray-400 font-bold uppercase tracking-widest">Open</span>
                            </div>
                            <div className="flex flex-col items-center">
                                <div className="flex items-center gap-1 text-rose-500">
                                    <XCircle className="w-3.5 h-3.5" />
                                    <span className="text-lg font-black">{stats.closed}</span>
                                </div>
                                <span className="text-[10px] text-gray-400 font-bold uppercase tracking-widest">Closed</span>
                            </div>
                            <div className="flex flex-col items-center">
                                <span className="text-lg font-black text-gray-700">{stats.total}</span>
                                <span className="text-[10px] text-gray-400 font-bold uppercase tracking-widest">Total</span>
                            </div>
                        </div>
                    )}

                    {/* Progress */}
                    {isAnalyzing && (
                        <div className="px-4 pb-3">
                            <div className="flex items-center gap-2 mb-1.5">
                                <Loader2 className="w-3 h-3 animate-spin text-emerald-500" />
                                <span className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">
                                    Analyzing {cityName}...
                                </span>
                            </div>
                            <div className="w-full bg-gray-100 rounded-full h-1.5 overflow-hidden">
                                <div
                                    className="bg-emerald-500 h-1.5 rounded-full transition-all duration-300"
                                    style={{ width: `${progress}%` }}
                                />
                            </div>
                        </div>
                    )}
                </form>

                {/* Error */}
                {error && (
                    <div className="mt-3 bg-rose-50/95 backdrop-blur-sm text-rose-700 p-3 rounded-xl flex items-center gap-2 text-sm shadow-lg border border-rose-100">
                        <AlertCircle className="w-4 h-4 shrink-0" />
                        <p className="text-xs">{error}</p>
                    </div>
                )}
            </div>

            {/* Legend */}
            <div className="absolute bottom-6 right-4 z-[1000] bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-gray-100 px-4 py-3">
                <div className="flex items-center gap-4 text-xs font-semibold">
                    <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-full bg-emerald-500 border-2 border-white shadow"></span>
                        <span className="text-gray-600">Open</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-full bg-rose-500 border-2 border-white shadow"></span>
                        <span className="text-gray-600">Closed</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-full bg-gray-400 border-2 border-white shadow"></span>
                        <span className="text-gray-600">Unknown</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function CitiesPage() {
    return (
        <Suspense
            fallback={
                <div className="w-full flex-1 flex items-center justify-center">
                    <Loader2 className="w-8 h-8 animate-spin text-emerald-500" />
                </div>
            }
        >
            <CitiesContent />
        </Suspense>
    );
}
