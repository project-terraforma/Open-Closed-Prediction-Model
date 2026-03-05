"use client";

import { useEffect, useMemo } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import Link from "next/link";
import type { SearchResultType } from "./SearchResults";

function MapUpdater({ center, zoom }: { center: [number, number]; zoom: number }) {
    const map = useMap();
    useEffect(() => {
        map.setView(center, zoom);
    }, [center, zoom, map]);
    return null;
}

interface CityMapProps {
    results: SearchResultType[];
    center?: [number, number];
}

export default function CityMap({ results, center }: CityMapProps) {
    const defaultCenter: [number, number] = center || [36.97, -122.03]; // Santa Cruz fallback

    const validResults = useMemo(() => results.filter((r) => r.lat && r.lon), [results]);

    const mapCenter: [number, number] = useMemo(() => {
        if (center) return center;
        if (validResults.length > 0) return [validResults[0].lat!, validResults[0].lon!];
        return defaultCenter;
    }, [center, validResults, defaultCenter]);

    const mapZoom = center ? 12 : validResults.length > 0 ? 12 : 6;

    return (
        <MapContainer
            center={mapCenter}
            zoom={mapZoom}
            scrollWheelZoom
            className="w-full h-full z-0"
            style={{ background: "#f8fafb" }}
        >
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
                url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
            />
            <MapUpdater center={mapCenter} zoom={mapZoom} />

            {validResults.map((res, i) => {
                const isOpen = res.status?.toLowerCase() === "open";
                const isClosed = res.status?.toLowerCase() === "closed";
                const color = isOpen ? "#10b981" : isClosed ? "#ef4444" : "#9ca3af";

                return (
                    <CircleMarker
                        key={`${res.id}-${i}`}
                        center={[res.lat!, res.lon!]}
                        radius={7}
                        pathOptions={{
                            fillColor: color,
                            fillOpacity: 0.85,
                            color: "#fff",
                            weight: 2,
                        }}
                    >
                        <Popup minWidth={200}>
                            <div className="font-sans text-sm space-y-1.5 p-1">
                                <h3 className="font-bold text-gray-900 leading-tight text-base">
                                    {res.name || "Unknown"}
                                </h3>
                                <p className={`text-xs font-bold uppercase tracking-wider ${isOpen ? "text-emerald-600" : isClosed ? "text-rose-600" : "text-gray-500"}`}>
                                    {res.status} &middot; {((res.confidence ?? 0) * 100).toFixed(0)}% confidence
                                </p>
                                {res.address && (
                                    <p className="text-xs text-gray-500">{res.address}</p>
                                )}
                                {res.category && (
                                    <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">{res.category}</p>
                                )}
                                <Link
                                    href={`/place/${res.id}`}
                                    className="inline-block mt-1 text-xs font-bold text-emerald-600 hover:text-emerald-700"
                                >
                                    View details &rarr;
                                </Link>
                            </div>
                        </Popup>
                    </CircleMarker>
                );
            })}
        </MapContainer>
    );
}
