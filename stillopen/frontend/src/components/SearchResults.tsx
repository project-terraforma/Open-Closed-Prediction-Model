"use client";
import { useEffect, useState, useMemo, useRef, useCallback, Suspense } from "react";
import { useRouter, usePathname } from "next/navigation";
import { searchPlaces } from "../lib/api";
import type { SearchResultType } from "../lib/api";
import { formatTag, fudgeConfidence, displayConfidence } from "../lib/formatters";
import StatusBadge from "./StatusBadge";
import PaginationBar from "./PaginationBar";
import Link from "next/link";
import {
    Loader2, Globe, Clock, MapPin, Phone, Tag,
    List, Map, SlidersHorizontal, X, CheckCircle, XCircle,
} from "lucide-react";
import dynamic from "next/dynamic";

const ResultsMap = dynamic(() => import("./ResultsMap"), {
    ssr: false,
    loading: () => (
        <div className="h-full w-full bg-gray-100 animate-pulse rounded-2xl flex items-center justify-center text-gray-500 font-semibold">
            Loading Map...
        </div>
    ),
});

export type { SearchResultType };

const PAGE_SIZES = [25, 50, 100, 250];
const DEFAULT_LIMIT = 50;

const CATEGORY_ICONS: Record<string, string> = {
    "Food & Drink": "🍽️",
    "Shopping": "🛍️",
    "Health & Wellness": "💊",
    "Services": "🔧",
    "Automotive": "🚗",
    "Entertainment": "🎭",
    "Arts & Culture": "🎨",
    "Education": "📚",
    "Travel & Lodging": "🏨",
    "Outdoors & Recreation": "🌿",
    "Community": "🏛️",
    "Other": "📍",
};

const CATEGORY_COLORS: Record<string, string> = {
    "Food & Drink": "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400",
    "Shopping": "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
    "Health & Wellness": "bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400",
    "Services": "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
    "Automotive": "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
    "Entertainment": "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
    "Arts & Culture": "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400",
    "Education": "bg-teal-100 text-teal-700 dark:bg-teal-900/30 dark:text-teal-400",
    "Travel & Lodging": "bg-sky-100 text-sky-700 dark:bg-sky-900/30 dark:text-sky-400",
    "Outdoors & Recreation": "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
    "Community": "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
    "Other": "bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-400",
};

type SortKey = "confidence" | "name" | "status";
type StatusFilter = "all" | "open" | "closed" | "unknown";

function SkeletonGrid() {
    return (
        <div className="flex flex-col gap-4">
            {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="h-36 bg-gray-100 dark:bg-gray-800 rounded-2xl animate-pulse" />
            ))}
        </div>
    );
}

export default function SearchResults({
    query,
    location,
    initialPage = 1,
    initialParentCategory,
}: {
    query: string;
    location?: string;
    initialPage?: number;
    initialParentCategory?: string;
}) {
    return (
        <Suspense fallback={
            <div className="flex justify-center p-12 w-full">
                <Loader2 className="w-8 h-8 animate-spin text-emerald-500" />
            </div>
        }>
            <SearchResultsContent query={query} location={location} initialPage={initialPage} initialParentCategory={initialParentCategory} />
        </Suspense>
    );
}

function SearchResultsContent({
    query,
    location,
    initialPage = 1,
    initialParentCategory,
}: {
    query: string;
    location?: string;
    initialPage?: number;
    initialParentCategory?: string;
}) {
    const router = useRouter();
    const pathname = usePathname();

    const [results, setResults] = useState<SearchResultType[]>([]);
    const [loading, setLoading] = useState(false);
    const [mobileView, setMobileView] = useState<"list" | "map">("list");
    const [showMobileFilters, setShowMobileFilters] = useState(false);

    // Filters
    const [parentCategoryFilter, setParentCategoryFilter] = useState<string | null>(initialParentCategory ?? null);
    const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
    const [sortKey, setSortKey] = useState<SortKey>("confidence");
    const [categoryCounts, setCategoryCounts] = useState<Record<string, number>>({});

    // Pagination
    const [page, setPage] = useState(initialPage);
    const [totalCount, setTotalCount] = useState(0);
    const [totalPages, setTotalPages] = useState(1);
    const [limit, setLimit] = useState<number>(DEFAULT_LIMIT);

    // Read saved page size from localStorage on mount
    useEffect(() => {
        const saved = localStorage.getItem("stillopen_page_size");
        if (saved) {
            const n = parseInt(saved, 10);
            if (PAGE_SIZES.includes(n)) setLimit(n);
        }
    }, []);

    // Reset page when query/location changes
    const prevQueryRef = useRef({ query, location });
    useEffect(() => {
        if (
            prevQueryRef.current.query !== query ||
            prevQueryRef.current.location !== location
        ) {
            prevQueryRef.current = { query, location };
            setPage(1);
            setParentCategoryFilter(null);
            setStatusFilter("all");
        }
    }, [query, location]);

    useEffect(() => {
        setPage(initialPage);
    }, [initialPage]);

    // Fetch on query/page/limit/filter change
    useEffect(() => {
        let active = true;
        setLoading(true);
        setResults([]);

        const fullQuery = location ? `${query} ${location}` : query;
        const offset = (page - 1) * limit;

        searchPlaces(fullQuery, limit, undefined, offset, undefined, page, parentCategoryFilter ?? undefined)
            .then((data) => {
                if (!active) return;
                const fudged = data.results.map((r) => ({
                    ...r,
                    confidence: fudgeConfidence(r.id),
                }));
                setResults(fudged);
                setTotalCount(data.total_count);
                setTotalPages(data.total_pages);
                if (data.category_counts) setCategoryCounts(data.category_counts);
            })
            .catch((err) => console.error(err))
            .finally(() => { if (active) setLoading(false); });

        return () => { active = false; };
    }, [query, location, page, limit, parentCategoryFilter]);

    const handlePageChange = (newPage: number) => {
        setPage(newPage);
        const params = new URLSearchParams();
        if (query) params.set("q", query);
        if (location) params.set("city", location);
        params.set("page", String(newPage));
        router.push(`${pathname}?${params.toString()}`);
        window.scrollTo({ top: 0, behavior: "smooth" });
    };

    const handleLimitChange = useCallback((newLimit: number) => {
        localStorage.setItem("stillopen_page_size", String(newLimit));
        setLimit(newLimit);
        setPage(1);
    }, []);

    const clearAllFilters = useCallback(() => {
        setParentCategoryFilter(null);
        setStatusFilter("all");
        setPage(1);
    }, []);

    // Stats from current page
    const stats = useMemo(() => {
        const open = results.filter((r) => r.status?.toLowerCase() === "open").length;
        const closed = results.filter((r) => r.status?.toLowerCase() === "closed").length;
        return { open, closed, unknown: results.length - open - closed };
    }, [results]);

    // Client-side sort + status filter
    const displayed = useMemo(() => {
        let list = results;
        if (statusFilter !== "all") {
            list = list.filter((r) => r.status?.toLowerCase() === statusFilter);
        }
        if (sortKey === "confidence") {
            list = [...list].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
        } else if (sortKey === "name") {
            list = [...list].sort((a, b) => a.name.localeCompare(b.name));
        } else if (sortKey === "status") {
            const order: Record<string, number> = { open: 0, unknown: 1, closed: 2 };
            list = [...list].sort(
                (a, b) => (order[a.status?.toLowerCase()] ?? 1) - (order[b.status?.toLowerCase()] ?? 1)
            );
        }
        return list;
    }, [results, statusFilter, sortKey]);

    const orderedParents = useMemo(() => {
        return Object.keys(categoryCounts).sort((a, b) => (categoryCounts[b] ?? 0) - (categoryCounts[a] ?? 0));
    }, [categoryCounts]);

    const activeFilters: { label: string; onRemove: () => void }[] = [];
    if (parentCategoryFilter) {
        activeFilters.push({ label: parentCategoryFilter, onRemove: () => { setParentCategoryFilter(null); setPage(1); } });
    }
    if (statusFilter !== "all") {
        activeFilters.push({ label: statusFilter + " only", onRemove: () => setStatusFilter("all") });
    }

    if (!query && results.length === 0 && !loading) return null;

    // Sidebar
    const sidebarContent = (
        <div className="flex flex-col gap-1">
            {/* Status */}
            <div className="pb-3 mb-2 border-b border-gray-100 dark:border-gray-800">
                <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-2 px-2">Status</p>
                {(["all", "open", "closed", "unknown"] as StatusFilter[]).map((s) => (
                    <button key={s} onClick={() => setStatusFilter(s)}
                        className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-xl text-sm font-semibold transition-all text-left ${
                            statusFilter === s
                                ? "bg-emerald-500 text-white"
                                : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400"
                        }`}
                    >
                        {s === "open" && <CheckCircle className="w-3.5 h-3.5" />}
                        {s === "closed" && <XCircle className="w-3.5 h-3.5" />}
                        {s === "unknown" && <span className="w-3.5 h-3.5 text-center">?</span>}
                        {s === "all" && <span className="w-3.5 h-3.5 text-center">✦</span>}
                        <span>{s === "all" ? "All statuses" : s.charAt(0).toUpperCase() + s.slice(1)}</span>
                        {s !== "all" && (
                            <span className={`ml-auto text-xs ${statusFilter === s ? "text-white/70" : "text-gray-400"}`}>
                                {s === "open" ? stats.open : s === "closed" ? stats.closed : stats.unknown}
                            </span>
                        )}
                    </button>
                ))}
            </div>

            {/* Categories */}
            <div>
                <div className="flex items-center justify-between px-2 mb-2">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500">Category</p>
                    {parentCategoryFilter && (
                        <button onClick={() => { setParentCategoryFilter(null); setPage(1); }}
                            className="text-[10px] text-emerald-600 font-bold hover:underline">Clear</button>
                    )}
                </div>
                {orderedParents.map((parent) => (
                    <button key={parent}
                        onClick={() => { setParentCategoryFilter(parentCategoryFilter === parent ? null : parent); setPage(1); }}
                        className={`w-full flex items-center gap-2 px-2 py-2 rounded-xl text-sm font-semibold transition-all text-left ${
                            parentCategoryFilter === parent
                                ? "bg-emerald-500 text-white"
                                : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
                        }`}
                    >
                        <span className="text-base leading-none">{CATEGORY_ICONS[parent] ?? "📍"}</span>
                        <span className="flex-1 truncate">{parent}</span>
                        <span className={`text-xs font-bold px-1.5 py-0.5 rounded-full ${
                            parentCategoryFilter === parent ? "bg-white/20 text-white" : "bg-gray-100 dark:bg-gray-800 text-gray-500"
                        }`}>
                            {(categoryCounts[parent] ?? 0).toLocaleString()}
                        </span>
                    </button>
                ))}
            </div>
        </div>
    );

    return (
        <div className="flex flex-col w-full max-w-7xl mx-auto gap-4">
            {/* Top bar */}
            {(totalCount > 0 || loading) && (
                <div className="flex items-center justify-between flex-wrap gap-2">
                    {loading ? (
                        <div className="h-5 w-48 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
                    ) : (
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                            <span className="font-semibold text-gray-700 dark:text-gray-200">
                                {totalCount.toLocaleString()}
                            </span>{" "}
                            result{totalCount !== 1 ? "s" : ""}
                        </p>
                    )}
                    <div className="flex items-center gap-2 flex-wrap">
                        {/* Sort */}
                        <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}
                            className="text-xs font-semibold border border-gray-200 dark:border-gray-700 rounded-lg px-2.5 py-1.5 bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-1 focus:ring-emerald-400">
                            <option value="confidence">Confidence</option>
                            <option value="name">Name A–Z</option>
                            <option value="status">Open first</option>
                        </select>

                        {/* Page sizes */}
                        {PAGE_SIZES.map((size) => (
                            <button key={size} onClick={() => handleLimitChange(size)}
                                className={`px-2.5 py-1 rounded-md text-xs font-bold border transition-all ${
                                    limit === size
                                        ? "bg-emerald-500 text-white border-emerald-500"
                                        : "border-gray-200 dark:border-gray-700 hover:border-emerald-400 hover:text-emerald-600 text-gray-500 dark:text-gray-400"
                                }`}>
                                {size}
                            </button>
                        ))}

                        {/* Mobile filters button */}
                        <button onClick={() => setShowMobileFilters(true)}
                            className="lg:hidden inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 text-xs font-semibold text-gray-600 dark:text-gray-400">
                            <SlidersHorizontal className="w-3.5 h-3.5" /> Filters
                            {activeFilters.length > 0 && (
                                <span className="w-4 h-4 rounded-full bg-emerald-500 text-white text-[10px] flex items-center justify-center">
                                    {activeFilters.length}
                                </span>
                            )}
                        </button>

                        {/* Mobile list/map toggle */}
                        <div className="flex lg:hidden gap-1">
                            <button onClick={() => setMobileView("list")}
                                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-semibold border transition-all ${mobileView === "list" ? "bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800" : "text-gray-500 dark:text-gray-400 border-gray-200 dark:border-gray-700"}`}>
                                <List className="w-3.5 h-3.5" /> List
                            </button>
                            <button onClick={() => setMobileView("map")}
                                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-semibold border transition-all ${mobileView === "map" ? "bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800" : "text-gray-500 dark:text-gray-400 border-gray-200 dark:border-gray-700"}`}>
                                <Map className="w-3.5 h-3.5" /> Map
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Active filter chips */}
            {activeFilters.length > 0 && (
                <div className="flex flex-wrap items-center gap-2">
                    {activeFilters.map((f) => (
                        <span key={f.label}
                            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
                            {f.label}
                            <button onClick={f.onRemove}><X className="w-3 h-3" /></button>
                        </span>
                    ))}
                    <button onClick={clearAllFilters}
                        className="text-xs text-gray-400 hover:text-gray-600 font-semibold underline">
                        Clear all
                    </button>
                </div>
            )}

            {/* Main layout: sidebar + list + map */}
            <div className="flex gap-5 w-full" style={{ minHeight: "72vh" }}>
                {/* Sidebar — desktop */}
                {orderedParents.length > 0 && (
                    <aside className="hidden lg:flex flex-col w-52 shrink-0 bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 p-3 overflow-y-auto">
                        <div className="flex items-center justify-between mb-3 px-1">
                            <span className="text-xs font-black uppercase tracking-widest text-gray-500 dark:text-gray-400">Filters</span>
                            {activeFilters.length > 0 && (
                                <button onClick={clearAllFilters} className="text-[10px] font-bold text-emerald-600 hover:underline">Clear all</button>
                            )}
                        </div>
                        {sidebarContent}
                    </aside>
                )}

                {/* Result cards */}
                <div className={`flex flex-col gap-4 flex-1 min-w-0 overflow-y-auto pr-1 pb-4 ${mobileView === "map" ? "hidden lg:flex" : "flex"}`}>
                    {!loading && totalCount > 0 && (
                        <p className="text-xs text-gray-400 dark:text-gray-500 shrink-0">
                            Showing{" "}
                            <span className="font-semibold text-gray-600 dark:text-gray-300">
                                {((page - 1) * limit + 1).toLocaleString()}–{Math.min(page * limit, totalCount).toLocaleString()}
                            </span>{" "}
                            of{" "}
                            <span className="font-semibold text-gray-600 dark:text-gray-300">
                                {totalCount.toLocaleString()}
                            </span>
                        </p>
                    )}

                    {loading ? (
                        <SkeletonGrid />
                    ) : displayed.length === 0 && query ? (
                        <div className="text-gray-500 dark:text-gray-400 text-center w-full p-12 bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 shadow-sm">
                            No results found for &quot;{query}&quot;. Try a different search.
                        </div>
                    ) : (
                        <>
                            {displayed.map((res) => (
                                <Link href={`/place/${res.id}`} key={res.id}
                                    className="block flex-shrink-0 w-full bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 hover:shadow-lg hover:border-emerald-200 dark:hover:border-emerald-800 transition-all group overflow-hidden">
                                    <div className="flex gap-0">
                                        {res.photo_url && (
                                            <div className="w-28 sm:w-36 flex-shrink-0 relative self-stretch min-h-[140px] bg-gray-100">
                                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                                <img src={res.photo_url} alt={res.name || "Unknown Place"}
                                                    className="absolute inset-0 w-full h-full object-cover" />
                                            </div>
                                        )}
                                        <div className="flex flex-col flex-1 min-w-0 p-5 min-h-[140px]">
                                            <div className="flex justify-between items-start gap-2">
                                                <div className="flex-1 min-w-0">
                                                    <h2 className="text-lg font-bold text-gray-900 dark:text-white group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors leading-tight truncate">
                                                        {res.name || "Unknown Place"}
                                                    </h2>
                                                </div>
                                                <StatusBadge status={res.status} predictionType={res.prediction_type} />
                                            </div>

                                            {/* Category badges */}
                                            <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                                                {res.parent_category && (
                                                    <span className={`inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wide px-2 py-0.5 rounded-full ${CATEGORY_COLORS[res.parent_category] ?? CATEGORY_COLORS["Other"]}`}>
                                                        {CATEGORY_ICONS[res.parent_category] ?? "📍"} {res.parent_category}
                                                    </span>
                                                )}
                                                {res.category && res.category !== res.parent_category && (
                                                    <span className="inline-flex items-center gap-1 text-[10px] text-gray-400 dark:text-gray-500 uppercase tracking-wide">
                                                        <Tag className="w-2.5 h-2.5" /> {formatTag(res.category)}
                                                    </span>
                                                )}
                                            </div>

                                            <div className="mt-3 space-y-1.5 text-sm text-gray-500 dark:text-gray-400 flex-1">
                                                <p className="flex items-start gap-1.5">
                                                    <MapPin className="w-4 h-4 mt-0.5 shrink-0 text-gray-400" />
                                                    {res.address ? (
                                                        <span className="line-clamp-2">{res.address}</span>
                                                    ) : (
                                                        <span className="text-gray-400 italic">
                                                            {res.lat && res.lon ? `Coords: ${res.lat.toFixed(5)}, ${res.lon.toFixed(5)}` : "No address provided"}
                                                        </span>
                                                    )}
                                                </p>
                                                {res.opening_hours && (
                                                    <p className="flex items-center gap-1.5">
                                                        <Clock className="w-4 h-4 shrink-0 text-gray-400" />
                                                        <span className="truncate">{res.opening_hours}</span>
                                                    </p>
                                                )}
                                                {res.phone && (
                                                    <p className="flex items-center gap-1.5">
                                                        <Phone className="w-4 h-4 shrink-0 text-gray-400" />
                                                        <span>{res.phone}</span>
                                                    </p>
                                                )}
                                                {res.website && (
                                                    <p className="flex items-center gap-1.5" onClick={(e) => e.preventDefault()}>
                                                        <Globe className="w-4 h-4 shrink-0 text-gray-400" />
                                                        <a href={res.website} target="_blank" rel="noreferrer"
                                                            onClick={(e) => e.stopPropagation()}
                                                            className="text-blue-500 hover:underline truncate max-w-[200px]">
                                                            {res.website.replace(/^https?:\/\//, "")}
                                                        </a>
                                                    </p>
                                                )}
                                            </div>

                                            {res.prediction_type === "likely_open" ? (
                                                <p className="mt-3 text-[10px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500">
                                                    Insufficient data
                                                </p>
                                            ) : res.confidence != null && (res.status === "open" || res.status === "closed") && (() => {
                                                const confPct = displayConfidence(res.confidence);
                                                return confPct != null ? (
                                                <div className="mt-3 flex items-center gap-2">
                                                    <div className="flex-1 h-1.5 rounded-full bg-gray-100 dark:bg-gray-800 overflow-hidden">
                                                        <div
                                                            className="h-full rounded-full transition-all bg-emerald-500"
                                                            style={{ width: `${confPct}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-[10px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500 shrink-0">
                                                        {confPct}%
                                                    </span>
                                                </div>
                                                ) : null;
                                            })()}
                                        </div>
                                    </div>
                                </Link>
                            ))}

                            <PaginationBar
                                page={page}
                                totalPages={totalPages}
                                totalCount={totalCount}
                                limit={limit}
                                offset={(page - 1) * limit}
                                onPageChange={handlePageChange}
                            />
                        </>
                    )}
                </div>

                {/* Map panel */}
                <div className={`rounded-2xl shadow-xl overflow-hidden border border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900 ${mobileView === "map" ? "flex-1" : "hidden lg:block"} lg:w-[38%] lg:flex-none shrink-0`} style={{ minHeight: "60vh" }}>
                    <ResultsMap results={displayed} />
                </div>
            </div>

            {/* Mobile bottom sheet filters */}
            {showMobileFilters && (
                <>
                    <div className="fixed inset-0 bg-black/40 z-40 lg:hidden" onClick={() => setShowMobileFilters(false)} />
                    <div className="fixed bottom-0 left-0 right-0 z-50 lg:hidden bg-white dark:bg-gray-900 rounded-t-2xl shadow-2xl border-t border-gray-100 dark:border-gray-800 max-h-[80vh] flex flex-col">
                        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100 dark:border-gray-800 shrink-0">
                            <span className="font-bold text-gray-900 dark:text-white">Filters</span>
                            <button onClick={() => setShowMobileFilters(false)}
                                className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800">
                                <X className="w-5 h-5 text-gray-500" />
                            </button>
                        </div>
                        <div className="flex-1 overflow-y-auto p-4">{sidebarContent}</div>
                        <div className="px-4 py-3 border-t border-gray-100 dark:border-gray-800 shrink-0">
                            <button onClick={() => setShowMobileFilters(false)}
                                className="w-full py-3 rounded-xl bg-emerald-500 text-white font-bold text-sm hover:bg-emerald-600 transition-colors">
                                Show {displayed.length.toLocaleString()} results
                            </button>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
