export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface BoundingBox {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
}

export async function geocodeCity(cityName: string): Promise<{ bbox: BoundingBox, displayName: string } | null> {
    try {
        const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(cityName)}&format=json&limit=1`;
        const res = await fetch(url, { headers: { 'User-Agent': 'StillOpenCitiesMode/1.0' } });
        if (!res.ok) return null;

        const data = await res.json();
        if (!data || data.length === 0) return null;

        const firstResult = data[0];
        // Nominatim returns boundingbox as [SouthLat, NorthLat, WestLon, EastLon] strings
        const bboxArr = firstResult.boundingbox;

        return {
            displayName: firstResult.display_name,
            bbox: {
                min_lat: parseFloat(bboxArr[0]),
                max_lat: parseFloat(bboxArr[1]),
                min_lon: parseFloat(bboxArr[2]),
                max_lon: parseFloat(bboxArr[3]),
            }
        };
    } catch (e) {
        console.error("Failed to geocode city:", e);
        return null;
    }
}

export async function searchPlacesByBbox(bbox: BoundingBox, limit = 50, offset = 0) {
    let url = `${API_BASE}/search?q=&limit=${limit}&offset=${offset}`;
    url += `&min_lat=${bbox.min_lat}&max_lat=${bbox.max_lat}&min_lon=${bbox.min_lon}&max_lon=${bbox.max_lon}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to search places by bounding box");
    return res.json();
}
