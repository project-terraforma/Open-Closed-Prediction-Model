"use client";
import { useEffect, useState, useMemo, useRef, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import { geocodeCity } from "../lib/CitySearchService";
import { formatTag, displayConfidence } from "../lib/formatters";
import { searchPlaces } from "../lib/api";
import type { SearchResultType } from "../lib/api";
import StatusBadge from "./StatusBadge";
import PaginationBar from "./PaginationBar";
import Link from "next/link";
import {
    AlertCircle, MapPin, Tag, CheckCircle, XCircle,
    List, Map as MapIcon, SlidersHorizontal, X, ChevronDown,
    ChevronRight,
} from "lucide-react";
import dynamic from "next/dynamic";

const ResultsMap = dynamic(() => import("./ResultsMap"), {
    ssr: false,
    loading: () => (
        <div className="h-full w-full bg-gray-100 dark:bg-gray-800 animate-pulse rounded-2xl flex items-center justify-center text-gray-500 font-semibold">
            Loading Map...
        </div>
    ),
});

function toTitleCase(str: string): string {
    return str.replace(/\b\w/g, (c) => c.toUpperCase());
}

type SortKey = "confidence" | "name" | "status";
type StatusFilter = "all" | "open" | "closed" | "unknown";

const PAGE_SIZES = [25, 50, 100, 250];
const DEFAULT_LIMIT = 50;

// Category icons (emoji) for browse cards and badges
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

// Colors for parent category badges
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

function SkeletonCard() {
    return (
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 p-4 animate-pulse">
            <div className="flex justify-between items-start mb-2">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/5" />
                <div className="h-5 bg-gray-200 dark:bg-gray-700 rounded-full w-14" />
            </div>
            <div className="h-3 bg-gray-100 dark:bg-gray-800 rounded w-1/3 mb-3" />
            <div className="h-3 bg-gray-100 dark:bg-gray-800 rounded w-4/5 mb-2" />
            <div className="h-1.5 bg-gray-100 dark:bg-gray-800 rounded-full w-full mt-4" />
        </div>
    );
}

function SkeletonGrid() {
    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
    );
}

// --- Sidebar category item ---
interface SidebarCategoryProps {
    parent: string;
    count: number;
    isActive: boolean;
    isExpanded: boolean;
    children: string[];
    childCounts: Record<string, number>;
    onSelect: (p: string | null) => void;
    onToggleExpand: (p: string) => void;
}
function SidebarCategory({
    parent, count, isActive, isExpanded, children, childCounts, onSelect, onToggleExpand,
}: SidebarCategoryProps) {
    const icon = CATEGORY_ICONS[parent] || "📍";
    const hasChildren = children.length > 0;

    return (
        <div>
            <button
                onClick={() => onSelect(isActive ? null : parent)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold transition-all text-left ${
                    isActive
                        ? "bg-emerald-500 text-white"
                        : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
                }`}
            >
                <span className="text-base leading-none">{icon}</span>
                <span className="flex-1 truncate">{parent}</span>
                <span className={`text-xs font-bold px-1.5 py-0.5 rounded-full ${
                    isActive ? "bg-white/20 text-white" : "bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400"
                }`}>
                    {count.toLocaleString()}
                </span>
                {hasChildren && isActive && (
                    <span onClick={(e) => { e.stopPropagation(); onToggleExpand(parent); }}
                        className="ml-1 opacity-70 hover:opacity-100">
                        {isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                    </span>
                )}
            </button>
            {isActive && isExpanded && hasChildren && (
                <div className="ml-4 mt-1 space-y-0.5">
                    {children
                        .filter((c) => (childCounts[c] ?? 0) > 0)
                        .sort((a, b) => (childCounts[b] ?? 0) - (childCounts[a] ?? 0))
                        .slice(0, 12)
                        .map((child) => (
                            <div key={child}
                                className="flex items-center gap-2 px-2 py-1 text-xs text-gray-500 dark:text-gray-400">
                                <span className="w-1 h-1 rounded-full bg-emerald-400 shrink-0" />
                                <span className="flex-1 truncate">{formatTag(child)}</span>
                                <span className="text-gray-400 dark:text-gray-600">
                                    {(childCounts[child] ?? 0).toLocaleString()}
                                </span>
                            </div>
                        ))}
                </div>
            )}
        </div>
    );
}

export default function CitySearchResults({
    query,
    city,
    initialPage = 1,
    initialParentCategory,
}: {
    query: string;
    city: string;
    initialPage?: number;
    initialParentCategory?: string;
}) {
    const router = useRouter();
    const pathname = usePathname();

    const [results, setResults] = useState<SearchResultType[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [resolvedCity, setResolvedCity] = useState<string | null>(null);
    const [boundary, setBoundary] = useState<object | null>(null);
    const [mobileView, setMobileView] = useState<"list" | "map">("list");
    const [showMobileFilters, setShowMobileFilters] = useState(false);

    // Filters
    const [parentCategoryFilter, setParentCategoryFilter] = useState<string | null>(initialParentCategory ?? null);
    const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
    const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
    const [sortKey, setSortKey] = useState<SortKey>("confidence");

    // Category counts from API (for sidebar)
    const [categoryCounts, setCategoryCounts] = useState<Record<string, number>>({});

    // Pagination
    const [page, setPage] = useState(initialPage);
    const [totalCount, setTotalCount] = useState(0);
    const [totalPages, setTotalPages] = useState(1);
    const [limit, setLimit] = useState<number>(DEFAULT_LIMIT);

    const geocodeRef = useRef<{
        city: string;
        bbox: { min_lat: number; max_lat: number; min_lon: number; max_lon: number } | null;
        resolved: string;
        boundary: object | null;
    } | null>(null);

    // Load saved page size
    useEffect(() => {
        const saved = localStorage.getItem("stillopen_page_size");
        if (saved) {
            const n = parseInt(saved, 10);
            if (PAGE_SIZES.includes(n)) setLimit(n);
        }
    }, []);

    // Sync page from prop
    useEffect(() => {
        setPage(initialPage);
    }, [initialPage]);

    // Phase 1: geocode city
    useEffect(() => {
        let cancelled = false;
        setBoundary(null);
        setResolvedCity(null);
        setError(null);
        setResults([]);
        setTotalCount(0);
        setTotalPages(1);
        geocodeRef.current = null;

        async function geocode() {
            try {
                const geoResult = await geocodeCity(city);
                if (cancelled) return;

                if (!geoResult) {
                    geocodeRef.current = { city, bbox: null, resolved: city, boundary: null };
                    setResolvedCity(city);
                } else {
                    const resolved = geoResult.displayName.split(",")[0].trim();
                    geocodeRef.current = {
                        city, bbox: geoResult.bbox, resolved, boundary: geoResult.boundary ?? null,
                    };
                    setResolvedCity(resolved);
                    setBoundary(geoResult.boundary ?? null);
                }
            } catch (err: unknown) {
                if (!cancelled) {
                    if (err instanceof Error && err.message === "THROTTLED") {
                        setError("Service temporarily throttled. Please try again in 60 seconds.");
                    } else {
                        setError(err instanceof Error ? err.message : "Could not geocode city.");
                    }
                    setLoading(false);
                }
            }
        }

        geocode();
        return () => { cancelled = true; };
    }, [city]);

    // Phase 2: fetch places
    useEffect(() => {
        if (!geocodeRef.current) return;
        if (geocodeRef.current.city !== city) return;

        let cancelled = false;
        setLoading(true);
        setResults([]);

        const { bbox: activeBbox } = geocodeRef.current;

        searchPlaces(
            query,
            limit,
            activeBbox ?? undefined,
            (page - 1) * limit,
            activeBbox ? undefined : city,
            page,
            parentCategoryFilter ?? undefined,
        )
            .then((data) => {
                if (cancelled) return;
                setResults(data.results);
                setTotalCount(data.total_count);
                setTotalPages(data.total_pages);
                if (data.category_counts) setCategoryCounts(data.category_counts);
                setLoading(false);
            })
            .catch((err: unknown) => {
                if (!cancelled) {
                    setError(err instanceof Error ? err.message : "Search failed.");
                    setLoading(false);
                }
            });

        return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [resolvedCity, query, page, limit, parentCategoryFilter]);

    // Compute child category counts from current page results
    const childCounts = useMemo(() => {
        const counts: Record<string, number> = {};
        for (const r of results) {
            if (r.category) counts[r.category] = (counts[r.category] ?? 0) + 1;
        }
        return counts;
    }, [results]);

    // Stats from current page
    const stats = useMemo(() => {
        const open = results.filter((r) => r.status?.toLowerCase() === "open").length;
        const closed = results.filter((r) => r.status?.toLowerCase() === "closed").length;
        return { open, closed, unknown: results.length - open - closed, total: results.length };
    }, [results]);

    // Client-side status filter + sort
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

    const handlePageChange = useCallback((newPage: number) => {
        setPage(newPage);
        const params = new URLSearchParams();
        if (query) params.set("q", query);
        params.set("city", city);
        params.set("page", String(newPage));
        if (parentCategoryFilter) params.set("parent_category", parentCategoryFilter);
        router.push(`${pathname}?${params.toString()}`);
        window.scrollTo({ top: 0, behavior: "smooth" });
    }, [query, city, parentCategoryFilter, pathname, router]);

    const handleLimitChange = useCallback((newLimit: number) => {
        localStorage.setItem("stillopen_page_size", String(newLimit));
        setLimit(newLimit);
        setPage(1);
    }, []);

    const handleCategorySelect = useCallback((cat: string | null) => {
        setParentCategoryFilter(cat);
        setExpandedCategory(cat);
        setPage(1);
        setStatusFilter("all");
    }, []);

    const clearAllFilters = useCallback(() => {
        setParentCategoryFilter(null);
        setExpandedCategory(null);
        setStatusFilter("all");
        setPage(1);
    }, []);

    const displayCity = resolvedCity || city;
    const queryLabel = query ? toTitleCase(query) : "All Places";

    // Ordered list of parent categories for sidebar (by count desc)
    const orderedParents = useMemo(() => {
        const parents = Object.keys(categoryCounts);
        return parents.sort((a, b) => (categoryCounts[b] ?? 0) - (categoryCounts[a] ?? 0));
    }, [categoryCounts]);

    // Active filter chips
    const activeFilters: { label: string; onRemove: () => void }[] = [];
    if (parentCategoryFilter) {
        activeFilters.push({ label: parentCategoryFilter, onRemove: () => handleCategorySelect(null) });
    }
    if (statusFilter !== "all") {
        activeFilters.push({
            label: statusFilter.charAt(0).toUpperCase() + statusFilter.slice(1) + " only",
            onRemove: () => setStatusFilter("all"),
        });
    }

    // --- Loading skeleton ---
    if (loading && results.length === 0) {
        return (
            <div className="flex flex-col gap-6 w-full">
                <div className="h-9 w-72 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
                <div className="flex gap-2">
                    {Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="h-6 w-16 bg-gray-100 dark:bg-gray-800 rounded-full animate-pulse" />
                    ))}
                </div>
                <SkeletonGrid />
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center p-16 text-center gap-4">
                <AlertCircle className="w-12 h-12 text-rose-400" />
                <p className="text-gray-600 dark:text-gray-400 text-lg font-medium">{error}</p>
            </div>
        );
    }

    if (!loading && results.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center p-16 text-center gap-4">
                <MapPin className="w-12 h-12 text-gray-300 dark:text-gray-600" />
                <h2 className="text-xl font-bold text-gray-700 dark:text-gray-300">No results found</h2>
                <p className="text-gray-500 dark:text-gray-400 max-w-sm">
                    No &ldquo;{query || "places"}&rdquo; found in {displayCity}. Try a broader term like{" "}
                    &ldquo;food&rdquo; or &ldquo;shop&rdquo;, or check the city name.
                </p>
            </div>
        );
    }

    // Sidebar content (shared between desktop sidebar + mobile bottom sheet)
    const sidebarContent = (
        <div className="flex flex-col gap-1 h-full">
            {/* Status filters */}
            <div className="pb-3 mb-2 border-b border-gray-100 dark:border-gray-800">
                <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-2 px-3">
                    Status
                </p>
                <div className="flex flex-col gap-0.5">
                    {(["all", "open", "closed", "unknown"] as StatusFilter[]).map((s) => (
                        <button
                            key={s}
                            onClick={() => setStatusFilter(s)}
                            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-sm font-semibold transition-all text-left ${
                                statusFilter === s
                                    ? "bg-emerald-500 text-white"
                                    : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400"
                            }`}
                        >
                            {s === "open" && <CheckCircle className="w-3.5 h-3.5" />}
                            {s === "closed" && <XCircle className="w-3.5 h-3.5" />}
                            {s === "unknown" && <span className="w-3.5 h-3.5 text-center leading-none">?</span>}
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
            </div>

            {/* Category filters */}
            <div className="flex-1 overflow-y-auto">
                <div className="flex items-center justify-between px-3 mb-2">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 dark:text-gray-500">
                        Category
                    </p>
                    {parentCategoryFilter && (
                        <button onClick={() => handleCategorySelect(null)}
                            className="text-[10px] text-emerald-600 dark:text-emerald-400 font-bold hover:underline">
                            Clear
                        </button>
                    )}
                </div>
                <div className="flex flex-col gap-0.5">
                    {orderedParents.map((parent) => (
                        <SidebarCategory
                            key={parent}
                            parent={parent}
                            count={categoryCounts[parent] ?? 0}
                            isActive={parentCategoryFilter === parent}
                            isExpanded={expandedCategory === parent}
                            children={[]} // child category breakdown shown from page results
                            childCounts={childCounts}
                            onSelect={handleCategorySelect}
                            onToggleExpand={setExpandedCategory}
                        />
                    ))}
                </div>
            </div>
        </div>
    );

    return (
        <div className="flex flex-col gap-4 w-full">
            {/* Header */}
            <div className="flex flex-col gap-1">
                <h1 className="text-2xl font-black text-gray-900 dark:text-white tracking-tight">
                    {query ? (
                        <>
                            <span className="text-emerald-500">{queryLabel}</span>{" in "}{displayCity}
                        </>
                    ) : (
                        <>All Places in <span className="text-emerald-500">{displayCity}</span></>
                    )}
                </h1>
                <p className="text-gray-400 text-sm">
                    {totalCount.toLocaleString()} result{totalCount !== 1 ? "s" : ""}
                </p>
            </div>

            {/* Active filter chips */}
            {activeFilters.length > 0 && (
                <div className="flex flex-wrap items-center gap-2">
                    {activeFilters.map((f) => (
                        <span key={f.label}
                            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
                            {f.label}
                            <button onClick={f.onRemove} className="hover:text-emerald-900 dark:hover:text-emerald-200">
                                <X className="w-3 h-3" />
                            </button>
                        </span>
                    ))}
                    <button onClick={clearAllFilters}
                        className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 font-semibold underline">
                        Clear all
                    </button>
                </div>
            )}

            {/* Stats + controls bar */}
            <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-4 text-sm">
                    <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-bold">
                        <CheckCircle className="w-3.5 h-3.5" /> {stats.open} open
                    </span>
                    <span className="flex items-center gap-1.5 text-rose-500 dark:text-rose-400 font-bold">
                        <XCircle className="w-3.5 h-3.5" /> {stats.closed} closed
                    </span>
                    <span className="text-gray-400 text-sm">{stats.unknown} unknown</span>
                </div>

                <div className="ml-auto flex items-center gap-2 flex-wrap">
                    {/* Sort */}
                    <select
                        value={sortKey}
                        onChange={(e) => setSortKey(e.target.value as SortKey)}
                        className="text-xs font-semibold border border-gray-200 dark:border-gray-700 rounded-lg px-2.5 py-1.5 bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-300 cursor-pointer focus:outline-none focus:ring-1 focus:ring-emerald-400"
                    >
                        <option value="confidence">Confidence</option>
                        <option value="name">Name A–Z</option>
                        <option value="status">Open first</option>
                    </select>

                    {/* Page size */}
                    <div className="flex items-center gap-1">
                        {PAGE_SIZES.map((size) => (
                            <button
                                key={size}
                                onClick={() => handleLimitChange(size)}
                                className={`px-2 py-1 rounded-md text-xs font-bold border transition-all ${
                                    limit === size
                                        ? "bg-emerald-500 text-white border-emerald-500"
                                        : "border-gray-200 dark:border-gray-700 hover:border-emerald-400 hover:text-emerald-600 text-gray-500 dark:text-gray-400"
                                }`}
                            >
                                {size}
                            </button>
                        ))}
                    </div>

                    {/* Mobile filters button */}
                    <button
                        onClick={() => setShowMobileFilters(true)}
                        className="lg:hidden inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 text-xs font-semibold text-gray-600 dark:text-gray-400 hover:border-emerald-400 transition-all"
                    >
                        <SlidersHorizontal className="w-3.5 h-3.5" /> Filters
                        {activeFilters.length > 0 && (
                            <span className="ml-0.5 w-4 h-4 rounded-full bg-emerald-500 text-white text-[10px] flex items-center justify-center">
                                {activeFilters.length}
                            </span>
                        )}
                    </button>

                    {/* Mobile list/map toggle */}
                    <div className="flex lg:hidden items-center gap-1">
                        <button
                            onClick={() => setMobileView("list")}
                            className={`inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                                mobileView === "list"
                                    ? "bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800"
                                    : "text-gray-500 border-gray-200 dark:border-gray-700"
                            }`}
                        >
                            <List className="w-3.5 h-3.5" /> List
                        </button>
                        <button
                            onClick={() => setMobileView("map")}
                            className={`inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                                mobileView === "map"
                                    ? "bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800"
                                    : "text-gray-500 border-gray-200 dark:border-gray-700"
                            }`}
                        >
                            <MapIcon className="w-3.5 h-3.5" /> Map
                        </button>
                    </div>
                </div>
            </div>

            {/* Main 3-panel layout: Sidebar | Results | Map */}
            <div className="flex gap-5 w-full" style={{ minHeight: "70vh" }}>
                {/* LEFT SIDEBAR — desktop only */}
                <aside className="hidden lg:flex flex-col w-56 shrink-0 bg-white dark:bg-gray-900 rounded-2xl border border-gray-100 dark:border-gray-800 p-3 overflow-hidden">
                    <div className="flex items-center justify-between mb-3 px-1">
                        <span className="text-xs font-black uppercase tracking-widest text-gray-500 dark:text-gray-400">
                            Filters
                        </span>
                        {activeFilters.length > 0 && (
                            <button onClick={clearAllFilters}
                                className="text-[10px] font-bold text-emerald-600 hover:underline">
                                Clear all
                            </button>
                        )}
                    </div>
                    {sidebarContent}
                </aside>

                {/* RESULTS AREA */}
                <div className={`flex flex-col gap-3 flex-1 min-w-0 overflow-y-auto pb-4 ${mobileView === "map" ? "hidden lg:flex" : "flex"}`}>
                    {/* Showing range label */}
                    {!loading && totalCount > 0 && (
                        <p className="text-xs text-gray-400 dark:text-gray-500 shrink-0">
                            Showing{" "}
                            <span className="font-semibold text-gray-600 dark:text-gray-300">
                                {((page - 1) * limit + 1).toLocaleString()}–
                                {Math.min(page * limit, totalCount).toLocaleString()}
                            </span>{" "}
                            of{" "}
                            <span className="font-semibold text-gray-600 dark:text-gray-300">
                                {totalCount.toLocaleString()}
                            </span>{" "}
                            results
                        </p>
                    )}

                    {loading ? (
                        <SkeletonGrid />
                    ) : displayed.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-16 text-center gap-3">
                            <MapPin className="w-10 h-10 text-gray-300 dark:text-gray-600" />
                            <p className="text-gray-500 dark:text-gray-400 text-sm">
                                No results match the current filters.
                            </p>
                            <button onClick={clearAllFilters}
                                className="text-sm text-emerald-600 dark:text-emerald-400 font-semibold hover:underline">
                                Clear filters
                            </button>
                        </div>
                    ) : (
                        <>
                            {/* Results grid: 2 cols on sm+, 1 col mobile */}
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                {displayed.map((res) => (
                                    <ResultCard key={res.id} res={res} />
                                ))}
                            </div>

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

                {/* MAP PANEL — desktop right side */}
                <div className={`rounded-2xl shadow-xl overflow-hidden border border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900 ${
                    mobileView === "map" ? "flex-1" : "hidden lg:block"
                } lg:w-[38%] lg:flex-none shrink-0`} style={{ minHeight: "60vh" }}>
                    <ResultsMap results={displayed} boundary={boundary} />
                </div>
            </div>

            {/* MOBILE BOTTOM SHEET FILTERS */}
            {showMobileFilters && (
                <>
                    {/* Backdrop */}
                    <div
                        className="fixed inset-0 bg-black/40 z-40 lg:hidden"
                        onClick={() => setShowMobileFilters(false)}
                    />
                    {/* Sheet */}
                    <div className="fixed bottom-0 left-0 right-0 z-50 lg:hidden bg-white dark:bg-gray-900 rounded-t-2xl shadow-2xl border-t border-gray-100 dark:border-gray-800 max-h-[80vh] flex flex-col">
                        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100 dark:border-gray-800 shrink-0">
                            <span className="font-bold text-gray-900 dark:text-white">Filters</span>
                            <button onClick={() => setShowMobileFilters(false)}
                                className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                                <X className="w-5 h-5 text-gray-500" />
                            </button>
                        </div>
                        <div className="flex-1 overflow-y-auto p-4">
                            {sidebarContent}
                        </div>
                        <div className="px-4 py-3 border-t border-gray-100 dark:border-gray-800 shrink-0">
                            <button
                                onClick={() => setShowMobileFilters(false)}
                                className="w-full py-3 rounded-xl bg-emerald-500 text-white font-bold text-sm hover:bg-emerald-600 transition-colors"
                            >
                                Show {displayed.length.toLocaleString()} results
                            </button>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

// --- Individual result card ---
function ResultCard({ res }: { res: SearchResultType }) {
    const parentColor = CATEGORY_COLORS[res.parent_category ?? "Other"] ?? CATEGORY_COLORS["Other"];

    return (
        <Link
            href={`/place/${res.id}`}
            className="block bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 hover:shadow-lg hover:border-emerald-200 dark:hover:border-emerald-800 transition-all group overflow-hidden"
        >
            <div className="flex flex-col p-4 min-h-[130px]">
                {/* Name + status */}
                <div className="flex justify-between items-start gap-2 mb-2">
                    <h2 className="text-sm font-bold text-gray-900 dark:text-white group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors leading-tight line-clamp-2">
                        {res.name || "Unknown Place"}
                    </h2>
                    <div className="shrink-0">
                        <StatusBadge status={res.status} predictionType={res.prediction_type} />
                    </div>
                </div>

                {/* Category badges */}
                <div className="flex flex-wrap items-center gap-1.5 mb-2">
                    {res.parent_category && (
                        <span className={`inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wide px-2 py-0.5 rounded-full ${parentColor}`}>
                            {CATEGORY_ICONS[res.parent_category] ?? "📍"} {res.parent_category}
                        </span>
                    )}
                    {res.category && res.category !== res.parent_category && (
                        <span className="inline-flex items-center gap-1 text-[10px] text-gray-400 dark:text-gray-500 uppercase tracking-wide">
                            <Tag className="w-2.5 h-2.5" /> {formatTag(res.category)}
                        </span>
                    )}
                </div>

                {/* Address */}
                {res.address && (
                    <p className="flex items-start gap-1.5 text-xs text-gray-500 dark:text-gray-400 mt-auto">
                        <MapPin className="w-3.5 h-3.5 mt-0.5 shrink-0 text-gray-400" />
                        <span className="line-clamp-2">{res.address}</span>
                    </p>
                )}

                {/* Confidence bar */}
                {(res.status === "open" || res.status === "closed") && res.confidence != null && (() => {
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
        </Link>
    );
}
