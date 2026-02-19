"use client";
import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import Link from "next/link";
import StatusBadge from "./StatusBadge";

// Fix generic map icon missing issues
L.Icon.Default.mergeOptions({
    iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
    iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
    shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

function MapUpdater({ center, zoom }: { center: [number, number], zoom: number }) {
    const map = useMap();
    useEffect(() => {
        map.setView(center, zoom);
    }, [center, zoom, map]);
    return null;
}

interface SearchResultType {
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
    website?: string;
    opening_hours?: string;
    photo_url?: string;
}

interface ResultsMapProps {
    results: SearchResultType[];
}

export default function ResultsMap({ results }: ResultsMapProps) {
    const defaultCenter: [number, number] = [36.7783, -119.4179]; // California fallback

    if (!results || results.length === 0) return (
        <MapContainer center={defaultCenter} zoom={6} scrollWheelZoom={true} className="w-full h-full rounded-2xl z-0">
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
                url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
            />
        </MapContainer>
    );

    // Center map on the first result with coords
    const firstResultWithCoords = results.find(r => r.lat && r.lon);
    const center: [number, number] = firstResultWithCoords
        ? [firstResultWithCoords.lat as number, firstResultWithCoords.lon as number]
        : defaultCenter;

    return (
        <MapContainer center={center} zoom={13} scrollWheelZoom={true} className="w-full h-full rounded-2xl z-0">
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
                url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
            />
            <MapUpdater center={center} zoom={13} />
            {results.filter(r => r.lat && r.lon).map((res) => (
                <Marker key={res.id} position={[res.lat as number, res.lon as number]}>
                    <Popup className="rounded-xl overflow-hidden p-0">
                        <div className="flex flex-col gap-2 min-w-[200px] mb-2 font-sans">
                            {res.photo_url && (
                                // eslint-disable-next-line @next/next/no-img-element
                                <img src={res.photo_url} alt={res.name} className="w-full h-24 object-cover rounded-t-md mb-1" />
                            )}
                            <div className="px-2 pb-1">
                                <h3 className="font-bold text-gray-900 border-b pb-1 mb-1 leading-tight">{res.name}</h3>
                                <p className="text-gray-500 text-xs mb-2 leading-snug">{res.address}</p>
                                <div className="flex items-center gap-2">
                                    <StatusBadge status={res.status} />
                                    <Link href={`/place/${res.id}`} className="text-emerald-600 hover:text-emerald-700 font-bold text-xs">
                                        View &rarr;
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
