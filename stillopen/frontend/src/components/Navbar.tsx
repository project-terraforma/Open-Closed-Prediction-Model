"use client";
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import Logo from './Logo';
import SearchBar from './SearchBar';
import { MapPin } from 'lucide-react';

export default function Navbar() {
    const pathname = usePathname();
    const isHome = pathname === '/';
    const isCities = pathname === '/cities';

    return (
        <nav className="sticky top-0 z-40 w-full bg-white/80 backdrop-blur-md border-b border-gray-100">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
                <Logo />

                {/* Show compact search bar in navbar if we aren't on the homepage */}
                {!isHome && !isCities && (
                    <div className="flex-1 max-w-md ml-8 hidden sm:block">
                        <SearchBar compact={true} />
                    </div>
                )}

                <div className="flex items-center gap-2 ml-auto">
                    <Link
                        href="/cities"
                        className={`inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-semibold transition-all ${isCities
                                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                                : "text-gray-500 hover:text-emerald-600 hover:bg-emerald-50/50 border border-transparent"
                            }`}
                    >
                        <MapPin className="w-4 h-4" />
                        <span className="hidden sm:inline">Cities</span>
                    </Link>
                </div>
            </div>
        </nav>
    );
}
