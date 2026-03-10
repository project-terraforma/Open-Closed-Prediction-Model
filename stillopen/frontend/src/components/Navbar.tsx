"use client";

import { usePathname } from 'next/navigation';
import Logo from './Logo';
import SearchBar from './SearchBar';
import Link from 'next/link';
import { Moon, Sun, MapPin, LayoutGrid } from 'lucide-react';
import { useState, useEffect, useCallback } from 'react';
import { useAppContext } from '../lib/AppContext';
import Breadcrumbs from './Breadcrumbs';

export default function Navbar() {
    const pathname = usePathname();
    const isHome = pathname === '/';
    const [isDark, setIsDark] = useState(false);
    const [isLocating, setIsLocating] = useState(false);

    const { setUserLocation, userLocation } = useAppContext();

    // On mount, handle theme and keybinds
    useEffect(() => {
        // Theme init
        const saved = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const initial = saved ? saved === 'dark' : prefersDark;
        setIsDark(initial);
        document.documentElement.classList.toggle('dark', initial);

        // Global Cmd+K / Ctrl+K listener
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.querySelector('input[placeholder*="Places"]') as HTMLInputElement;
                if (searchInput) searchInput.focus();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const toggleTheme = () => {
        const next = !isDark;
        setIsDark(next);
        document.documentElement.classList.toggle('dark', next);
        localStorage.setItem('theme', next ? 'dark' : 'light');
    };

    const handleLocationSwitch = useCallback(() => {
        setIsLocating(true);
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const { latitude: lat, longitude: lon } = pos.coords;
                setUserLocation({ lat, lon, displayName: 'Current Location' });
                setIsLocating(false);
                alert(`Location updated to: ${lat.toFixed(4)}, ${lon.toFixed(4)}`);
            },
            (err) => {
                console.error(err);
                setIsLocating(false);
                alert("Could not get location. Please enable permissions.");
            }
        );
    }, [setUserLocation]);

    return (
        <>
            <nav className="sticky top-0 z-40 w-full bg-white/90 dark:bg-gray-950/90 backdrop-blur-md border-b-2 border-gray-100 dark:border-gray-900 shadow-sm">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center gap-4">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:gap-4 shrink-0">
                        <Logo />
                        <div className="hidden md:block">
                            <Breadcrumbs />
                        </div>
                    </div>

                    {/* Show search bar in navbar if we aren't on the homepage */}
                    {!isHome && (
                        <div className="flex-1 max-w-md hidden md:block">
                            <div className="relative group">
                                <SearchBar compact={true} />
                                <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none hidden lg:flex items-center gap-1 px-1.5 py-0.5 rounded border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-[10px] font-bold text-gray-400 group-focus-within:hidden">
                                    <span className="text-[8px]">⌘</span>K
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="flex items-center gap-1 ml-auto">
                        <Link
                            href="/browse"
                            className={`hidden sm:inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-sm font-semibold transition-all ${
                                pathname === '/browse'
                                    ? 'text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/20'
                                    : 'text-gray-500 dark:text-gray-400 hover:text-emerald-600 dark:hover:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-900/20'
                            }`}
                        >
                            <LayoutGrid className="w-4 h-4" /> Browse
                        </Link>
                        <div className="hidden sm:block w-px h-6 bg-gray-200 dark:bg-gray-800 mx-1" />
                        <button
                            onClick={handleLocationSwitch}
                            disabled={isLocating}
                            className={`p-2 rounded-xl text-gray-500 hover:bg-emerald-50 dark:hover:bg-emerald-900/20 hover:text-emerald-600 dark:hover:text-emerald-400 transition-all ${isLocating ? 'animate-pulse text-emerald-500' : ''} ${userLocation ? 'text-emerald-500 bg-emerald-50 dark:bg-emerald-900/30' : ''}`}
                            aria-label="Switch location"
                            tabIndex={0}
                        >
                            <MapPin className="w-5 h-5" />
                        </button>

                        <div className="w-px h-6 bg-gray-200 dark:bg-gray-800 mx-1" />

                        <button
                            onClick={toggleTheme}
                            className="p-2 rounded-xl text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                            aria-label="Toggle dark mode"
                            tabIndex={0}
                        >
                            {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                        </button>
                    </div>
                </div>

                {/* Mobile Breadcrumbs - visible on small screens only */}
                <div className="md:hidden px-4 pb-2 -mt-1">
                    <Breadcrumbs />
                </div>
            </nav>
        </>
    );
}
