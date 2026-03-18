"use client";
import { useState, useEffect, useRef, useMemo } from "react";
import { Search, Loader2, MapPin, Building2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { searchPlaces } from "../lib/api";
import { fudgeConfidence, displayConfidence } from "../lib/formatters";
import StatusBadge from "./StatusBadge";

import type { SearchResultType } from "./SearchResults";

const RECENT_KEY = 'recent_searches';
const MAX_RECENT = 5;

function loadRecent(): string[] {
    if (typeof window === 'undefined') return [];
    try { return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]'); } catch { return []; }
}
function persistRecent(q: string) {
    const updated = [q, ...loadRecent().filter(s => s !== q)].slice(0, MAX_RECENT);
    localStorage.setItem(RECENT_KEY, JSON.stringify(updated));
}

// Detect "coffee in Santa Cruz" / "gyms near Reno" / "bars around Chicago"
const CITY_PATTERN = /^(.+?)\s+(?:in|near|around)\s+(.+)$/i;

function parseCityQuery(input: string): { term: string; city: string } | null {
    const match = input.match(CITY_PATTERN);
    if (!match) return null;
    const term = match[1].trim();
    const city = match[2].trim();
    if (term.length >= 2 && city.length >= 2) return { term, city };
    return null;
}


export default function SearchBar({ compact = false }: { compact?: boolean }) {
    const [query, setQuery] = useState("");
    const [location, setLocation] = useState("");
    const [results, setResults] = useState<SearchResultType[]>([]);
    const [loading, setLoading] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    // Track which input is focused so we can swap dropdown content appropriately
    const [activeInput, setActiveInput] = useState<"query" | "location">("query");
    const dropdownRef = useRef<HTMLDivElement>(null);
    const router = useRouter();

    // Detect "X in Y" pattern live as the user types in the main field
    const cityParsed = useMemo(() => parseCityQuery(query), [query]);

    // Nominatim-validated city name for the "X in Y" pattern
    const [detectedCity, setDetectedCity] = useState<string | null>(null);

    useEffect(() => {
        const rawCity = cityParsed?.city;
        if (!rawCity || rawCity.length < 2) {
            setDetectedCity(null);
            return;
        }
        const timer = setTimeout(async () => {
            try {
                const res = await fetch(
                    `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(rawCity)}&format=json&limit=1`,
                    { headers: { "Accept-Language": "en" } }
                );
                const data = await res.json();
                if (data.length > 0) {
                    setDetectedCity(data[0].display_name.split(",")[0].trim());
                } else {
                    setDetectedCity(null);
                }
            } catch {
                setDetectedCity(null);
            }
        }, 400);
        return () => clearTimeout(timer);
    }, [cityParsed?.city]);

    // Effective city: from the location field, or Nominatim-validated from the query
    const effectiveCity = location.trim() || detectedCity || null;
    // Effective search term: left-side of "X in Y", or the full query
    const effectiveTerm = cityParsed ? cityParsed.term : query;

    const hasCitySuggestion = effectiveCity !== null && effectiveTerm.trim().length >= 2;

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setShowDropdown(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    // Place search — only runs when the query field is active (not the location field)
    useEffect(() => {
        if (activeInput === "location") {
            setResults([]);
            return;
        }
        // When city pattern is detected in query, search for just the term part
        const searchQuery = cityParsed ? cityParsed.term : query;

        if (searchQuery.trim().length < 2) {
            setResults([]);
            return;
        }

        const timeoutId = setTimeout(async () => {
            setLoading(true);
            try {
                const data = await searchPlaces(searchQuery);
                const fudged = data.results.map(r => ({
                    ...r,
                    confidence: fudgeConfidence(r.id)
                }));
                setResults(fudged);
                setShowDropdown(true);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        }, 300);
        return () => clearTimeout(timeoutId);
    }, [query, cityParsed, activeInput]);

    const handleSearchSubmit = () => {
        const trimmedQuery = effectiveTerm.trim();
        if (!trimmedQuery && !effectiveCity) return;

        let url: string;
        if (effectiveCity) {
            url = `/search?q=${encodeURIComponent(trimmedQuery)}&city=${encodeURIComponent(effectiveCity)}`;
        } else {
            url = `/search?q=${encodeURIComponent(trimmedQuery)}`;
        }

        router.push(url);
        setShowDropdown(false);
        persistRecent(query);
    };

    const handleCitySearchClick = (term: string, city: string) => {
        router.push(`/search?q=${encodeURIComponent(term)}&city=${encodeURIComponent(city)}`);
        setShowDropdown(false);
        persistRecent(query);
    };

    const showDropdownContent = showDropdown && activeInput === "query" && (results.length > 0 || hasCitySuggestion);

    return (
        <div className={`relative w-full ${compact ? 'max-w-md' : 'max-w-3xl px-6'}`} ref={dropdownRef}>
            <div className={`flex w-full overflow-hidden bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 transition-all shadow-[0_4px_20px_rgba(0,0,0,0.12)] dark:shadow-[0_0_18px_rgba(255,255,255,0.12)] hover:shadow-[0_10px_35px_rgba(0,0,0,0.2)] dark:hover:shadow-[0_0_35px_rgba(255,255,255,0.2)] group ${compact ? 'rounded-lg text-sm' : 'rounded-full text-lg'}`}>

                {/* What Input Section */}
                <div className="flex-1 relative flex items-center min-w-0">
                    <div className={`absolute left-0 flex items-center pointer-events-none z-10 ${compact ? 'pl-4' : 'pl-8'}`}>
                        {loading ? (
                            <Loader2 className="h-5 w-5 text-emerald-500 animate-spin" />
                        ) : (
                            <Search className="h-5 w-5 text-gray-400 group-focus-within:text-emerald-500 transition-colors" />
                        )}
                    </div>
                    <input
                        type="text"
                        className={`block w-full text-gray-900 dark:text-white bg-transparent placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none transition-all ${compact ? 'pl-11 pr-4 py-2' : 'pl-16 pr-4 py-5'}`}
                        placeholder="Places in mind?"
                        value={query}
                        onChange={(e) => {
                            setQuery(e.target.value);
                            setActiveInput("query");
                            if (e.target.value.length > 0) setShowDropdown(true);
                        }}
                        onFocus={() => {
                            setActiveInput("query");
                            if (results.length > 0 || hasCitySuggestion) setShowDropdown(true);
                        }}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSearchSubmit();
                        }}
                    />
                </div>

                {/* Vertical Divider */}
                <div className="w-px h-8 self-center bg-gray-200 dark:bg-gray-700 hidden sm:block" />

                {/* Where Input Section */}
                <div className="flex-[0.8] relative hidden sm:flex items-center min-w-0">
                    <div className="absolute left-4 flex items-center pointer-events-none z-10">
                        <MapPin className={`h-5 w-5 transition-colors ${location.trim() ? 'text-emerald-500' : 'text-gray-400 group-focus-within:text-emerald-500'}`} />
                    </div>
                    <input
                        type="text"
                        className={`block w-full text-gray-900 dark:text-white bg-transparent placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none transition-all ${compact ? 'pl-11 pr-4 py-2' : 'pl-11 pr-4 py-5'}`}
                        placeholder="City or location..."
                        value={location}
                        onChange={(e) => {
                            setLocation(e.target.value);
                            setActiveInput("location");
                            setShowDropdown(true);
                        }}
                        onFocus={() => {
                            setActiveInput("location");
                            setShowDropdown(true);
                        }}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSearchSubmit();
                        }}
                    />
                </div>

                {/* Search Button (only for non-compact) */}
                {!compact && (
                    <button
                        onClick={handleSearchSubmit}
                        className="m-1.5 px-8 rounded-full bg-emerald-500 hover:bg-emerald-600 text-white font-bold transition-all shadow-md active:scale-95 flex-shrink-0"
                    >
                        Search
                    </button>
                )}
            </div>

            {/* City search chip — shown below bar when city is detected */}
            <AnimatePresence>
                {!compact && hasCitySuggestion && (
                    <motion.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -4 }}
                        className="absolute left-6 right-6 top-full mt-1.5 z-40 flex items-center gap-1.5 px-4 py-1.5 bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800 rounded-full text-xs font-semibold text-emerald-700 dark:text-emerald-400 shadow-sm pointer-events-none w-fit"
                    >
                        <Building2 className="w-3 h-3" />
                        Searching in <span className="font-bold">{effectiveCity}</span>
                    </motion.div>
                )}
            </AnimatePresence>

            <AnimatePresence>
                {showDropdownContent && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.98 }}
                        className={`absolute z-50 w-full left-0 right-0 ${hasCitySuggestion && !compact ? 'mt-10' : 'mt-2'} ${compact ? '' : 'px-6'}`}
                    >
                        <div className="bg-white dark:bg-gray-900 rounded-xl shadow-[0_12px_44px_rgba(0,0,0,0.22)] dark:shadow-[0_0_35px_rgba(255,255,255,0.12)] border border-gray-100 dark:border-gray-800 overflow-hidden ring-1 ring-black ring-opacity-5 dark:ring-white/10 max-h-96 overflow-y-auto">

                            {/* === QUERY FIELD MODE === */}
                            {activeInput === "query" && (
                                <>
                                    {/* City search suggestion row — pinned at top when city detected */}
                                    {hasCitySuggestion && (
                                        <button
                                            className="w-full px-6 py-3.5 text-left hover:bg-emerald-50 dark:hover:bg-emerald-900/20 transition-colors flex items-center gap-3 border-b border-emerald-100 dark:border-emerald-900/40 bg-emerald-50/50 dark:bg-emerald-900/10"
                                            onClick={() => handleCitySearchClick(effectiveTerm.trim(), effectiveCity!)}
                                        >
                                            <span className="text-lg leading-none">🏙</span>
                                            <div className="flex flex-col">
                                                <span className="font-bold text-gray-900 dark:text-white text-sm">
                                                    Search &ldquo;{effectiveTerm.trim()}&rdquo; in {effectiveCity}
                                                </span>
                                                <span className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold">
                                                    City search &mdash; all matching places
                                                </span>
                                            </div>
                                        </button>
                                    )}

                                    {/* Regular place results — separated when city suggestion also showing */}
                                    {results.length > 0 && (
                                        <>
                                            {hasCitySuggestion && (
                                                <div className="px-4 pt-2 pb-1">
                                                    <span className="text-[10px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500">
                                                        Places
                                                    </span>
                                                </div>
                                            )}
                                            {results.map((result) => (
                                                <button
                                                    key={result.id}
                                                    className="w-full px-6 py-4 text-left hover:bg-emerald-50 dark:hover:bg-emerald-900/20 transition-colors flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-gray-50 dark:border-gray-800 last:border-0 group"
                                                    onClick={() => {
                                                        setQuery(result.name);
                                                        setShowDropdown(false);
                                                        router.push(`/place/${result.id}`);
                                                    }}
                                                >
                                                    <div className="flex flex-col flex-1 pl-4 sm:pl-0 truncate">
                                                        <span className="font-bold text-gray-900 dark:text-white group-hover:text-emerald-700 dark:group-hover:text-emerald-400 transition-colors truncate">{result.name}</span>
                                                        <span className="text-xs text-gray-500 dark:text-gray-400 font-medium truncate max-w-xs">{result.address}</span>
                                                    </div>
                                                    <div className="mt-2 sm:mt-0 flex flex-col items-start sm:items-end flex-shrink-0 ml-4">
                                                        <StatusBadge status={result.status} />
                                                        {(result.status === 'open' || result.status === 'closed') && (
                                                            <span className="text-[10px] text-gray-400 dark:text-gray-500 mt-1 uppercase tracking-wider font-semibold">
                                                                Conf: {displayConfidence(result.confidence) ?? "—"}%
                                                            </span>
                                                        )}
                                                    </div>
                                                </button>
                                            ))}
                                        </>
                                    )}
                                </>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
