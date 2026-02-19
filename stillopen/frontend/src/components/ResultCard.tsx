"use client";
import { motion } from "framer-motion";
import { CheckCircle, XCircle, AlertCircle, Info } from "lucide-react";

interface PlaceDetail {
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
    explanation: string[];
}

interface ResultProps {
    data: PlaceDetail;
}

export default function ResultCard({ data }: ResultProps) {
    const isReady = data.status.toLowerCase();
    const isOpen = isReady === "open";
    const isClosed = isReady === "closed";

    let colorClass = "text-gray-500";
    let bgClass = "bg-gray-50";
    let icon = <AlertCircle className="w-12 h-12" />;

    if (isOpen) {
        colorClass = "text-emerald-500";
        bgClass = "bg-emerald-50";
        icon = <CheckCircle className="w-12 h-12" />;
    } else if (isClosed) {
        colorClass = "text-rose-500";
        bgClass = "bg-rose-50";
        icon = <XCircle className="w-12 h-12" />;
    }

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-2xl bg-white rounded-2xl shadow-xl flex flex-col p-8 sm:p-12 border border-gray-100"
        >
            <div className="flex flex-col items-center text-center space-y-6 mb-12">
                <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring" }}
                    className={`p-5 rounded-full ${bgClass} ${colorClass} bg-opacity-30`}
                >
                    {icon}
                </motion.div>
                <div>
                    <h2 className="text-3xl sm:text-4xl font-black text-gray-900 tracking-tight">{data.name}</h2>
                    <p className="text-gray-500 font-medium text-lg mt-2">{data.address}</p>
                    {(data.category || data.source) && (
                        <div className="flex items-center justify-center space-x-3 mt-4">
                            {data.category && (
                                <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-xs font-bold uppercase tracking-wider">{data.category}</span>
                            )}
                            {data.source && (
                                <span className="px-3 py-1 bg-blue-50 text-blue-600 border border-blue-100 rounded-full text-xs font-bold uppercase tracking-wider">Source: {data.source}</span>
                            )}
                        </div>
                    )}
                </div>
                <div className="py-4 flex flex-col items-center">
                    <span className={`text-6xl font-black tracking-tighter uppercase ${colorClass}`}>
                        {data.status}
                    </span>
                    {(isOpen || isClosed) && (
                        <div className="mt-4 flex flex-col sm:flex-row items-center space-y-2 sm:space-y-0 sm:space-x-6 text-gray-400 text-sm font-semibold uppercase tracking-widest">
                            <span>Confidence: {(data.confidence * 100).toFixed(0)}%</span>
                        </div>
                    )}
                </div>
            </div>

            <div className="space-y-6 border-t border-gray-100 pt-10">
                <h3 className="text-gray-400 font-bold text-xs tracking-widest uppercase mb-6 flex items-center gap-2">
                    <Info className="w-4 h-4" /> Signal Analysis
                </h3>
                {data.explanation && data.explanation.length > 0 ? (
                    <ul className="space-y-4">
                        {data.explanation.map((item, idx) => (
                            <li key={idx} className="flex items-start text-gray-600 bg-gray-50 p-4 rounded-xl border border-gray-100">
                                <span className="w-2 h-2 mt-2 mr-3 bg-emerald-400 rounded-full flex-shrink-0"></span>
                                <span>{item}</span>
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="text-gray-500 italic text-sm">No detailed explanation available for this prediction.</p>
                )}
            </div>
        </motion.div>
    );
}
