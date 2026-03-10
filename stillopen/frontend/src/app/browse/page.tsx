"use client";
import { useEffect, useState } from "react";
import { getCategoryTree } from "../../lib/api";
import Link from "next/link";
import { Loader2 } from "lucide-react";

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

const CATEGORY_GRADIENTS: Record<string, string> = {
    "Food & Drink": "from-orange-400 to-red-400",
    "Shopping": "from-purple-400 to-pink-400",
    "Health & Wellness": "from-pink-400 to-rose-400",
    "Services": "from-blue-400 to-indigo-400",
    "Automotive": "from-gray-400 to-slate-500",
    "Entertainment": "from-yellow-400 to-orange-400",
    "Arts & Culture": "from-indigo-400 to-purple-400",
    "Education": "from-teal-400 to-emerald-400",
    "Travel & Lodging": "from-sky-400 to-blue-400",
    "Outdoors & Recreation": "from-emerald-400 to-teal-400",
    "Community": "from-red-400 to-pink-400",
    "Other": "from-gray-300 to-gray-400",
};

export default function BrowsePage() {
    const [categories, setCategories] = useState<{ parent: string; count: number }[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        getCategoryTree()
            .then((data) => {
                const parents = Object.keys(data.tree).filter((p) => p !== "Other");
                const sorted = parents
                    .map((p) => ({ parent: p, count: data.counts[p] ?? 0 }))
                    .sort((a, b) => b.count - a.count);
                // Add Other at the end
                if (data.counts["Other"] !== undefined) {
                    sorted.push({ parent: "Other", count: data.counts["Other"] ?? 0 });
                }
                setCategories(sorted);
            })
            .catch((err) => setError(err.message))
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[50vh] gap-4">
                <Loader2 className="w-8 h-8 animate-spin text-emerald-500" />
                <p className="text-gray-400 text-sm">Loading categories…</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[50vh] gap-4 text-center">
                <p className="text-rose-500 font-semibold">Failed to load categories</p>
                <p className="text-gray-400 text-sm">{error}</p>
            </div>
        );
    }

    return (
        <div className="max-w-6xl mx-auto px-4 py-10">
            {/* Header */}
            <div className="mb-10 text-center">
                <h1 className="text-4xl font-black text-gray-900 dark:text-white tracking-tight mb-3">
                    Browse by Category
                </h1>
                <p className="text-gray-500 dark:text-gray-400 text-lg max-w-xl mx-auto">
                    Explore 1.4M+ California businesses organized by type.
                    Click any category to see open/closed predictions.
                </p>
            </div>

            {/* Category grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                {categories.map(({ parent, count }) => (
                    <Link
                        key={parent}
                        href={`/search?parent_category=${encodeURIComponent(parent)}`}
                        className="group relative overflow-hidden rounded-2xl shadow-sm border border-white/10 hover:shadow-xl hover:-translate-y-0.5 transition-all duration-200"
                    >
                        {/* Gradient background */}
                        <div className={`absolute inset-0 bg-gradient-to-br ${CATEGORY_GRADIENTS[parent] ?? CATEGORY_GRADIENTS["Other"]} opacity-90 group-hover:opacity-100 transition-opacity`} />

                        {/* Content */}
                        <div className="relative z-10 p-5 flex flex-col gap-2 text-white min-h-[130px]">
                            <span className="text-3xl leading-none">{CATEGORY_ICONS[parent] ?? "📍"}</span>
                            <div className="mt-auto">
                                <p className="font-black text-sm leading-tight">{parent}</p>
                                {count > 0 && (
                                    <p className="text-white/70 text-xs mt-0.5 font-medium">
                                        {count.toLocaleString()} places
                                    </p>
                                )}
                            </div>
                        </div>
                    </Link>
                ))}
            </div>

            {/* Explore prompt */}
            <div className="mt-12 text-center">
                <p className="text-gray-400 dark:text-gray-500 text-sm mb-4">
                    Looking for something specific?
                </p>
                <Link
                    href="/search?q="
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-emerald-500 text-white font-bold hover:bg-emerald-600 transition-colors shadow-md hover:shadow-lg"
                >
                    Search all places
                </Link>
            </div>
        </div>
    );
}
