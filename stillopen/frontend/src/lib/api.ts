export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface SearchResultType {
    id: string;
    name: string;
    address: string;
    category?: string;
    parent_category?: string;   // mapped parent group (e.g. "Food & Drink")
    city?: string;
    state?: string;
    lat?: number;
    lon?: number;
    source?: string;
    // metadata_json not included in list results — only on /place/{id}
    status: string;
    confidence?: number | null;
    prediction_type?: string;
    website?: string;
    phone?: string;
    website_status?: string;
    website_checked_at?: string;
    website_http_code?: number;
    // kept for compatibility with components that might check these
    opening_hours?: string;
    photo_url?: string;
}

export interface SearchResponse {
    results: SearchResultType[];
    total_count: number;
    page: number;
    total_pages: number;
    limit: number;
    offset: number;
    has_next: boolean;
    has_prev: boolean;
    category_counts?: Record<string, number>;  // parent_category → total count
}

export interface CategoryTreeResponse {
    tree: Record<string, string[]>;
    counts: Record<string, number>;   // parent_category → total DB count
}

export async function searchPlaces(
    query: string,
    limit: number = 50,
    bbox?: { min_lat: number; max_lat: number; min_lon: number; max_lon: number },
    offset: number = 0,
    city?: string,
    page?: number,
    parent_category?: string,
): Promise<SearchResponse> {
    let url = `${API_BASE}/search?q=${encodeURIComponent(query)}&limit=${limit}&offset=${offset}`;
    if (page !== undefined) url += `&page=${page}`;
    if (bbox) {
        url += `&min_lat=${bbox.min_lat}&max_lat=${bbox.max_lat}&min_lon=${bbox.min_lon}&max_lon=${bbox.max_lon}`;
    }
    if (city) {
        url += `&city=${encodeURIComponent(city)}`;
    }
    if (parent_category) {
        url += `&parent_category=${encodeURIComponent(parent_category)}`;
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

export async function getCategoryTree(): Promise<CategoryTreeResponse> {
    const res = await fetch(`${API_BASE}/categories`);
    if (!res.ok) throw new Error("Failed to fetch categories");
    return res.json();
}
