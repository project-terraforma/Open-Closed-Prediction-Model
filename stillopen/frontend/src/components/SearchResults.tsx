"use client";
import { useEffect, useState } from "react";
import { searchPlaces } from "../lib/api";
import StatusBadge from "./StatusBadge";
import Link from "next/link";
import { Loader2, Globe, Clock, MapPin } from "lucide-react";
import dynamic from "next/dynamic";

const ResultsMap = dynamic(() => import("./ResultsMap"), {
    ssr: false,
    loading: () => <div className="h-full w-full bg-gray-100 animate-pulse rounded-2xl flex items-center justify-center text-gray-500 font-semibold">Loading Map...</div>,
});

interface SearchResultType {
    id: string;
    name: string;
    address: string;
    category?: string;
    lat?: number;
    lon?: number;
    source?: string;
    metadata_json?: Record<string, unknown>;
    status: string;
    confidence: number;
    website?: string;
    opening_hours?: string;
    photo_url?: string;
}

export default function SearchResults({ query }: { query: string }) {
    const [results, setResults] = useState<SearchResultType[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        let active = true;
        if (!query) {
            setTimeout(() => { if (active) setResults([]); }, 0);
            return;
        }
        setTimeout(() => {
            if (active) setLoading(true);
        }, 0);
        searchPlaces(query)
            .then(data => {
                if (active) setResults(data);
            })
            .catch(err => console.error(err))
            .finally(() => {
                if (active) setLoading(false);
            });

        return () => { active = false; };
    }, [query]);

    if (loading) {
        return <div className="flex justify-center p-12 w-full"><Loader2 className="w-8 h-8 animate-spin text-emerald-500" /></div>;
    }

    if (results.length === 0 && query) {
        return <div className="text-gray-500 text-center w-full p-12 bg-white rounded-2xl border border-gray-100 shadow-sm">No results found for &quot;{query}&quot;. Try searching for something else.</div>;
    }

    if (results.length === 0 && !query) {
        return null;
    }

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-7xl mx-auto h-[75vh]">
            <div className="flex flex-col gap-4 overflow-y-auto pr-2 pb-10">
                {results.map((res) => (
                    <Link href={`/place/${res.id}`} key={res.id} className="block w-full bg-white rounded-2xl shadow-sm border border-gray-100 p-5 hover:shadow-lg hover:border-emerald-200 transition-all group overflow-hidden relative">
                        <div className="flex gap-5">
                            {res.photo_url && (
                                // eslint-disable-next-line @next/next/no-img-element
                                <img src={res.photo_url} alt={res.name} className="w-24 h-24 sm:w-32 sm:h-32 object-cover rounded-xl shadow-sm flex-shrink-0" />
                            )}
                            <div className="flex flex-col flex-1 min-w-0 py-1">
                                <div className="flex justify-between items-start gap-2">
                                    <h2 className="text-xl font-bold text-gray-900 group-hover:text-emerald-600 transition-colors truncate">{res.name}</h2>
                                    <StatusBadge status={res.status} />
                                </div>

                                {res.category && (
                                    <span className="text-xs uppercase tracking-widest text-emerald-600 font-bold mt-1 block">
                                        {res.category}
                                    </span>
                                )}

                                <div className="mt-3 space-y-1.5 flex-1">
                                    <p className="text-gray-500 text-sm flex items-start gap-1.5">
                                        <MapPin className="w-4 h-4 mt-0.5 shrink-0 text-gray-400" />
                                        <span className="truncate">{res.address}</span>
                                    </p>
                                    {res.opening_hours && (
                                        <p className="text-gray-500 text-sm flex items-center gap-1.5">
                                            <Clock className="w-4 h-4 shrink-0 text-gray-400" />
                                            <span className="truncate">{res.opening_hours}</span>
                                        </p>
                                    )}
                                    {res.website && (
                                        <a href={res.website} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()} className="text-blue-500 hover:text-blue-600 text-sm flex items-center gap-1.5 w-fit">
                                            <Globe className="w-4 h-4 shrink-0" />
                                            <span className="truncate max-w-[200px] hover:underline">Website</span>
                                        </a>
                                    )}
                                </div>

                                <div className="flex justify-between items-end mt-4">
                                    {(res.status === 'open' || res.status === 'closed') && (
                                        <span className="text-[10px] text-gray-400 font-bold uppercase tracking-widest bg-gray-50 px-2 py-1 rounded-md">
                                            Conf: {(res.confidence * 100).toFixed(0)}%
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                    </Link>
                ))}
            </div>

            <div className="hidden lg:block h-full relative border border-gray-100 rounded-2xl shadow-xl overflow-hidden bg-white">
                <ResultsMap results={results} />
            </div>
        </div>
    );
}
