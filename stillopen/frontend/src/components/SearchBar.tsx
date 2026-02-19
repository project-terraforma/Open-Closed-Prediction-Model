"use client";
import { useState, useEffect, useRef } from "react";
import { Search, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { searchPlaces } from "../lib/api";
import StatusBadge from "./StatusBadge";

import type { SearchResultType } from "./SearchResults";

export default function SearchBar({ compact = false }: { compact?: boolean }) {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<SearchResultType[]>([]);
    const [loading, setLoading] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const router = useRouter();

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setShowDropdown(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    useEffect(() => {
        if (query.trim().length < 2) {
            setResults([]);
            return;
        }

        const timeoutId = setTimeout(async () => {
            setLoading(true);
            try {
                const data = await searchPlaces(query);
                setResults(data);
                setShowDropdown(true);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        }, 300);
        return () => clearTimeout(timeoutId);
    }, [query]);

    const handleSearchSubmit = () => {
        if (query.trim()) {
            router.push(`/search?q=${encodeURIComponent(query)}`);
            setShowDropdown(false);
        }
    }

    return (
        <div className={`relative w-full ${compact ? 'max-w-md' : 'max-w-2xl px-6'}`} ref={dropdownRef}>
            <div className="relative group w-full flex">
                <div className={`absolute inset-y-0 left-0 flex items-center pointer-events-none z-10 ${compact ? 'pl-4' : 'pl-10'}`}>
                    {loading ? (
                        <Loader2 className="h-5 w-5 text-emerald-500 animate-spin" />
                    ) : (
                        <Search className="h-5 w-5 text-gray-400 group-focus-within:text-emerald-500 transition-colors" />
                    )}
                </div>
                <input
                    type="text"
                    className={`block w-full text-gray-900 border border-gray-200 bg-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all shadow-sm ${compact ? 'pl-12 pr-4 py-2 rounded-lg text-sm' : 'pl-16 pr-6 py-6 rounded-full text-lg shadow-xl shadow-gray-100/50 hover:shadow-2xl hover:shadow-gray-200/50'}`}
                    placeholder="Search for a business or place..."
                    value={query}
                    onChange={(e) => {
                        setQuery(e.target.value);
                        if (e.target.value.length > 0) setShowDropdown(true);
                    }}
                    onFocus={() => {
                        if (results.length > 0) setShowDropdown(true);
                    }}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                            handleSearchSubmit();
                        }
                    }}
                />
            </div>

            <AnimatePresence>
                {showDropdown && results.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.98 }}
                        className={`absolute z-50 w-full left-0 right-0 mt-2 ${compact ? '' : 'px-6'}`}
                    >
                        <div className="bg-white rounded-xl shadow-2xl border border-gray-100 overflow-hidden ring-1 ring-black ring-opacity-5 max-h-96 overflow-y-auto">
                            {results.map((result) => (
                                <button
                                    key={result.id}
                                    className="w-full px-6 py-4 text-left hover:bg-emerald-50 transition-colors flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-gray-50 last:border-0 group"
                                    onClick={() => {
                                        setQuery(result.name);
                                        setShowDropdown(false);
                                        router.push(`/place/${result.id}`);
                                    }}
                                >
                                    <div className="flex flex-col flex-1 pl-4 sm:pl-0 truncate">
                                        <span className="font-bold text-gray-900 group-hover:text-emerald-700 transition-colors truncate">{result.name}</span>
                                        <span className="text-xs text-gray-500 font-medium truncate max-w-xs">{result.address}</span>
                                    </div>
                                    <div className="mt-2 sm:mt-0 flex flex-col items-start sm:items-end flex-shrink-0 ml-4">
                                        <StatusBadge status={result.status} />
                                        {(result.status === 'open' || result.status === 'closed') && (
                                            <span className="text-[10px] text-gray-400 mt-1 uppercase tracking-wider font-semibold">
                                                Conf: {(result.confidence * 100).toFixed(0)}%
                                            </span>
                                        )}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
