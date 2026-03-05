"use client";

import { motion } from "framer-motion";
import SearchBar from "../components/SearchBar";

export default function Home() {
    return (
        <div className="w-full flex-1 flex flex-col items-center justify-center p-6 relative overflow-hidden">
            {/* Background Gradients */}
            <div className="absolute top-20 -left-64 w-96 h-96 bg-purple-200/40 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse-slow"></div>
            <div className="absolute top-40 -right-64 w-96 h-96 bg-yellow-200/40 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse-slow animation-delay-2000"></div>
            <div className="absolute -bottom-32 left-1/2 transform -translate-x-1/2 w-[500px] h-[500px] bg-emerald-100/40 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse-slow animation-delay-4000"></div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                className="w-full flex flex-col items-center space-y-16 z-10 max-w-7xl mx-auto my-auto"
            >
                <div className="text-center space-y-6">
                    <h1 className="text-6xl sm:text-7xl font-black tracking-tighter text-gray-900 drop-shadow-sm">
                        Still<span className="text-emerald-500">Open</span>
                    </h1>
                    <p className="text-xl sm:text-2xl text-gray-500 font-light max-w-xl mx-auto leading-relaxed">
                        Operational status prediction powered by <span className="font-semibold text-gray-800">metadata intelligence</span>
                    </p>
                </div>

                <div className="w-full flex justify-center">
                    <SearchBar />
                </div>

                <div className="pt-8 flex gap-4 text-sm font-semibold tracking-wide">
                    <a href="/cities" className="text-gray-500 hover:text-emerald-600 border border-gray-200 bg-white/50 backdrop-blur-sm px-6 py-2 rounded-full shadow-sm hover:shadow transition-all">
                        🗺️ Try Cities Mode
                    </a>
                </div>
            </motion.div>

            <footer className="absolute bottom-8 w-full text-center text-gray-400 text-xs font-semibold tracking-widest uppercase z-10 opacity-50 space-x-4">
                <span>StillOpen Intelligence</span>
                <span>•</span>
                <span>v1.0.0</span>
            </footer>
        </div>
    );
}
