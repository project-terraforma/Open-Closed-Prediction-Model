"use client";
import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import Link from "next/link";
import StatusBadge from "./StatusBadge";
import { MapPin, Clock, Globe, Phone } from "lucide-react";
import type { SearchResultType } from "./SearchResults";

// Fix Leaflet default pin icons (webpack asset path issue)
L.Icon.Default.mergeOptions({
    iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
    iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
    shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

function MapUpdater({ center, zoom }: { center: [number, number]; zoom: number }) {
    const map = useMap();
    useEffect(() => {
        map.setView(center, zoom);
    }, [center, zoom, map]);
    return null;
}

interface ResultsMapProps {
    results: SearchResultType[];
}

export default function ResultsMap({ results }: ResultsMapProps) {
    const defaultCenter: [number, number] = [36.7783, -119.4179]; // California fallback

    const validResults = results.filter((r) => r.lat && r.lon);

    if (!validResults.length) {
        return (
            <MapContainer center={defaultCenter} zoom={6} scrollWheelZoom className="w-full h-full rounded-2xl z-0">
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
                    url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
                />
            </MapContainer>
        );
    }

    const first = validResults[0];
    const center: [number, number] = [first.lat as number, first.lon as number];

    return (
        <MapContainer center={center} zoom={13} scrollWheelZoom className="w-full h-full rounded-2xl z-0">
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
                url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
            />
            <MapUpdater center={center} zoom={13} />

            {validResults.map((res) => (
                <Marker key={res.id} position={[res.lat as number, res.lon as number]}>
                    <Popup className="rounded-xl overflow-hidden p-0" minWidth={220}>
                        <div className="flex flex-col font-sans text-sm">
                            {/* Photo — only if real URL */}
                            {res.photo_url && (
                                // eslint-disable-next-line @next/next/no-img-element
                                <img
                                    src={res.photo_url}
                                    alt={res.name}
                                    className="w-full h-28 object-cover"
                                />
                            )}

                            <div className="p-3 space-y-2">
                                <div className="flex items-start justify-between gap-2">
                                    <h3 className="font-bold text-gray-900 leading-tight">{res.name}</h3>
                                    <StatusBadge status={res.status} />
                                </div>

                                {res.category && (
                                    <span className="inline-block text-[10px] uppercase font-bold tracking-wider text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full">
                                        {res.category}
                                    </span>
                                )}

                                {res.address && (
                                    <p className="flex items-start gap-1.5 text-gray-600">
                                        <MapPin className="w-3.5 h-3.5 mt-0.5 shrink-0 text-gray-400" />
                                        <span className="text-xs leading-snug">{res.address}</span>
                                    </p>
                                )}

                                {res.opening_hours && (
                                    <p className="flex items-center gap-1.5 text-gray-600">
                                        <Clock className="w-3.5 h-3.5 shrink-0 text-gray-400" />
                                        <span className="text-xs">{res.opening_hours}</span>
                                    </p>
                                )}

                                {res.phone && (
                                    <p className="flex items-center gap-1.5">
                                        <Phone className="w-3.5 h-3.5 shrink-0 text-gray-400" />
                                        <a href={`tel:${res.phone}`} className="text-xs text-blue-500 hover:underline">
                                            {res.phone}
                                        </a>
                                    </p>
                                )}

                                {res.website && (
                                    <p className="flex items-center gap-1.5">
                                        <Globe className="w-3.5 h-3.5 shrink-0 text-gray-400" />
                                        <a
                                            href={res.website}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="text-xs text-blue-500 hover:underline truncate max-w-[150px]"
                                        >
                                            {res.website.replace(/^https?:\/\//, "")}
                                        </a>
                                    </p>
                                )}

                                <div className="pt-1 border-t border-gray-100">
                                    <Link
                                        href={`/place/${res.id}`}
                                        className="text-xs font-bold text-emerald-600 hover:text-emerald-700"
                                    >
                                        View details &rarr;
                                    </Link>
                                </div>
                            </div>
                        </div>
                    </Popup>
                </Marker>
            ))}
        </MapContainer>
    );
}
