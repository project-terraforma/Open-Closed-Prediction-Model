"use client";
import { useEffect, useState } from "react";
import { searchPlaces } from "../lib/api";
import StatusBadge from "./StatusBadge";
import Link from "next/link";
import { Loader2 } from "lucide-react";

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
        return <div className="flex justify-center p-12"><Loader2 className="w-8 h-8 animate-spin text-emerald-500" /></div>;
    }

    if (results.length === 0 && query) {
        return <div className="text-gray-500 text-center p-12 bg-white rounded-2xl border border-gray-100 shadow-sm">No results found for &quot;{query}&quot;. Try searching for something else.</div>;
    }

    if (results.length === 0 && !query) {
        return null;
    }

    return (
        <div className="flex flex-col gap-4">
            {results.map((res) => (
                <Link href={`/place/${res.id}`} key={res.id} className="block w-full bg-white rounded-xl shadow-sm border border-gray-100 p-6 hover:shadow-md hover:border-emerald-200 transition-all group">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <div>
                            <h2 className="text-xl font-bold text-gray-900 group-hover:text-emerald-600 transition-colors">{res.name}</h2>
                            <p className="text-gray-500 mt-1">{res.address}</p>
                        </div>
                        <div className="flex flex-col items-start sm:items-end gap-2">
                            <StatusBadge status={res.status} />
                            {(res.status === 'open' || res.status === 'closed') && (
                                <span className="text-xs text-gray-400 font-semibold uppercase tracking-wider">
                                    Confidence: {(res.confidence * 100).toFixed(0)}%
                                </span>
                            )}
                        </div>
                    </div>
                </Link>
            ))}
        </div>
    )
}
