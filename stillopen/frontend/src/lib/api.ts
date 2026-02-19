export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function searchPlaces(
    query: string,
    limit: number = 20,
    bbox?: { min_lat: number; max_lat: number; min_lon: number; max_lon: number }
) {
    let url = `${API_BASE}/search?q=${encodeURIComponent(query)}&limit=${limit}`;
    if (bbox) {
        url += `&min_lat=${bbox.min_lat}&max_lat=${bbox.max_lat}&min_lon=${bbox.min_lon}&max_lon=${bbox.max_lon}`;
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to search places");
    return res.json();
}

export async function getPlaceDetails(id: string) {
    const res = await fetch(`${API_BASE}/place/${id}`);
    if (!res.ok) {
        if (res.status === 404) return null;
        throw new Error("Failed to get place details");
    }
    return res.json();
}
