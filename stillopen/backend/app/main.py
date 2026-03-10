from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
import time
import requests
from datetime import datetime, timezone
from sqlalchemy import text

from .models import SearchResult, SearchResponse, PlaceDetail
from .search import search_places, get_place_record, load_data_to_db, ensure_indexes
from .database import engine, Base, IS_POSTGRES
from .categories import CATEGORY_TREE

# ---------------------------------------------------------------------------
# OSM Overpass enrichment helpers
# ---------------------------------------------------------------------------
_OSM_TAGS = [
    "opening_hours", "phone", "website", "cuisine",
    "wheelchair", "outdoor_seating", "takeaway",
    "delivery", "wifi", "parking",
]
_OSM_ENRICHMENT_TTL_DAYS = 30
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _needs_osm_enrichment(meta: dict) -> bool:
    """True if osm_enriched_at is missing or older than 30 days."""
    ts = meta.get("osm_enriched_at")
    if not ts:
        return True
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days >= _OSM_ENRICHMENT_TTL_DAYS
    except Exception:
        return True


def _fetch_osm_tags(lat: float, lon: float, name: str) -> dict:
    """
    Query Overpass within 50 m of lat/lon, return dict of matched OSM tags.
    Returns {} on any failure (timeout, HTTP error, no results).
    """
    query = f"[out:json][timeout:3];node(around:50,{lat},{lon});out tags;"
    try:
        resp = requests.post(_OVERPASS_URL, data={"data": query}, timeout=3)
        if resp.status_code != 200:
            return {}
        elements = resp.json().get("elements", [])
        if not elements:
            return {}

        # Pick element whose name best matches the place name
        name_lower = (name or "").lower()

        def _score(el):
            n = (el.get("tags", {}).get("name") or "").lower()
            if not n or not name_lower:
                return 0
            if n == name_lower:
                return 2
            if name_lower in n or n in name_lower:
                return 1
            return 0

        best = max(elements, key=_score)
        tags = best.get("tags", {})
        return {t: tags[t] for t in _OSM_TAGS if t in tags}
    except Exception:
        return {}


def _write_osm_to_db(place_id: str, osm_patch: dict) -> None:
    """Persist osm_patch into metadata_json for the given place."""
    if IS_POSTGRES:
        patch_json = json.dumps(osm_patch)
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE places SET metadata_json = metadata_json || :patch::jsonb WHERE place_id = :pid"),
                {"patch": patch_json, "pid": place_id},
            )
    else:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT metadata_json FROM places WHERE place_id = :pid"),
                {"pid": place_id},
            ).fetchone()
        meta = {}
        if row and row.metadata_json:
            try:
                meta = json.loads(row.metadata_json)
            except Exception:
                pass
        meta.update(osm_patch)
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE places SET metadata_json = :m WHERE place_id = :pid"),
                {"m": json.dumps(meta), "pid": place_id},
            )

app = FastAPI(title="StillOpen API")

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_data_to_db()
    ensure_indexes()

@app.get("/geocode/city")
def geocode_city_endpoint(city: str):
    """
    Geocode a city name using a local cache and Nominatim fallback.
    """
    print(f"DEBUG: Received geocode request for city: {city}")
    with engine.connect() as conn:
        try:
            # 1. Check Cache
            query = text("SELECT display_name, bbox, boundary FROM city_cache WHERE LOWER(name) = LOWER(:name)")
            row = conn.execute(query, {"name": city}).fetchone()
            if row:
                print(f"DEBUG: Cache hit for city: {city}")
                return {
                    "displayName": row.display_name,
                    "bbox": row.bbox,
                    "boundary": row.boundary
                }

            # 2. Fetch from Nominatim
            print(f"DEBUG: Cache miss for {city} - attempting live fetch.")
            res = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": city,
                    "format": "json",
                    "limit": 5,
                    "polygon_geojson": 1,
                    "countrycodes": "us"
                },
                headers={"User-Agent": "StillOpenBackend/1.0"},
                timeout=10
            )
            
            if res.status_code == 429:
                print(f"ERROR: Nominatim rate limit hit (429) for city: {city}")
                raise HTTPException(status_code=429, detail="Geocoding service throttled. Please try again later.")
            
            if res.status_code != 200:
                print(f"ERROR: Nominatim returned status {res.status_code} for city: {city}")
                raise HTTPException(status_code=res.status_code, detail="Geocoding service error")

            data = res.json()
            if not data:
                print(f"DEBUG: No results found for city: {city}")
                return None

            # Logic to pick best result (same as frontend)
            boundaries = [r for r in data if r.get('class') == 'boundary']
            boundaries.sort(key=lambda x: float(x.get('importance') or 0), reverse=True)
            first = boundaries[0] if boundaries else data[0]
            
            bbox_arr = first.get('boundingbox', [])
            result = {
                "displayName": first.get('display_name'),
                "boundary": first.get('geojson'),
                "bbox": {
                    "min_lat": float(bbox_arr[0]),
                    "max_lat": float(bbox_arr[1]),
                    "min_lon": float(bbox_arr[2]),
                    "max_lon": float(bbox_arr[3]),
                }
            }

            # 3. Save to Cache
            with engine.begin() as save_conn:
                save_conn.execute(
                    text("INSERT INTO city_cache (name, display_name, bbox, boundary) VALUES (:name, :dname, :bbox, :boundary) ON CONFLICT (name) DO NOTHING"),
                    {
                        "name": city,
                        "dname": result["displayName"],
                        "bbox": json.dumps(result["bbox"]),
                        "boundary": json.dumps(result["boundary"])
                    }
                )

            print(f"DEBUG: Successfully fetched and cached city: {city}")
            return result
        except HTTPException:
            raise
        except Exception as e:
            print(f"ERROR: Geocoding exception for {city}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}


# Category tree cache (5-minute TTL)
_category_cache: dict = {}
_category_cache_ts: float = 0.0
_CATEGORY_CACHE_TTL = 300.0  # 5 minutes


@app.get("/categories")
def get_categories():
    """
    Return the full CATEGORY_TREE as JSON.
    Optionally includes per-parent place counts when the DB is reachable.
    Response is cached for 5 minutes.
    """
    global _category_cache, _category_cache_ts
    now = time.monotonic() if hasattr(time, "monotonic") else __import__("time").monotonic()
    if _category_cache and (now - _category_cache_ts) < _CATEGORY_CACHE_TTL:
        return _category_cache

    # Fetch raw category counts from DB
    counts: dict[str, int] = {}
    if IS_POSTGRES:
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT category, COUNT(*) AS n FROM places GROUP BY category")
                ).fetchall()
            from .categories import get_parent_category
            from collections import defaultdict
            parent_counts: dict[str, int] = defaultdict(int)
            for row in rows:
                parent_counts[get_parent_category(row.category or "")] += row.n
            counts = dict(parent_counts)
        except Exception as e:
            print(f"Category count query failed: {e}")

    result = {
        "tree": CATEGORY_TREE,
        "counts": counts,
    }
    _category_cache = result
    _category_cache_ts = now
    return result

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = "",
    city: str = None,
    limit: int = 50,
    offset: int = 0,
    page: int = None,
    min_lat: float = None,
    max_lat: float = None,
    min_lon: float = None,
    max_lon: float = None,
    parent_category: str = None,
    status: str = None,
):
    """
    Search places.

    ?status=open    → only predicted-open records (uses pre-computed predicted_status column)
    ?status=closed  → only predicted-closed records
    ?status=unknown → only records without a pre-computed prediction
    Omitting ?status returns all records (existing behaviour).
    """
    limit = max(1, min(limit, 1000))

    # ?page=N takes priority over raw offset
    if page is not None and page >= 1:
        offset = (page - 1) * limit
        actual_page = page
    else:
        actual_page = max(1, (offset // limit) + 1) if limit > 0 else 1

    # Normalise status filter
    status_filter = status.lower().strip() if status else None
    if status_filter not in (None, "open", "closed", "unknown"):
        status_filter = None

    print(
        f"DEBUG: Search q='{q}' city='{city}' parent_category='{parent_category}' "
        f"status='{status_filter}' limit={limit} page={actual_page} offset={offset} "
        f"bbox=[{min_lat},{max_lat},{min_lon},{max_lon}]"
    )
    return search_places(
        query=q,
        city=city,
        limit=limit,
        offset=offset,
        page=actual_page,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        parent_category=parent_category,
        status_filter=status_filter,
    )

@app.get("/place/{place_id}", response_model=PlaceDetail)
def get_place_details(place_id: str):
    record = get_place_record(place_id)
    if not record:
        raise HTTPException(status_code=404, detail="Place not found")

    from .predict import predict_status
    meta = record.get("metadata_json", {}) or {}
    raw = meta.get("raw", meta) if isinstance(meta, dict) else {}
    # Merge top-level record fields so the scorer can check name/category/address
    place_input = {
        "name": record.get("name", ""),
        "category": record.get("category", ""),
        "address": record.get("address", ""),
        **raw,
    }
    prediction = predict_status(place_input)

    # --- Step 1: OSM Overpass enrichment ---
    lat, lon = record.get("lat"), record.get("lon")
    name = record.get("name", "")
    # Check both the outer metadata and raw sub-dict for the enrichment timestamp
    enrichment_check = {**meta, **raw} if isinstance(meta, dict) else raw

    osm_fields: dict = {}
    if lat and lon:
        if _needs_osm_enrichment(enrichment_check):
            fetched = _fetch_osm_tags(lat, lon, name)
            if fetched:
                now_ts = datetime.now(timezone.utc).isoformat()
                osm_patch = {**fetched, "osm_enriched_at": now_ts}
                try:
                    _write_osm_to_db(place_id, osm_patch)
                except Exception as e:
                    print(f"OSM DB write failed for {place_id}: {e}")
                osm_fields = osm_patch
        else:
            # Surface already-stored OSM fields
            for tag in _OSM_TAGS + ["osm_enriched_at"]:
                val = enrichment_check.get(tag)
                if val is not None:
                    osm_fields[tag] = val

    # --- Step 2: Surface Overture data already in DB ---
    brand_name: str | None = None
    brand_data = raw.get("brand") or {}
    if isinstance(brand_data, dict):
        # Overture brand schema: brand.names.common[0].value
        names = brand_data.get("names", {})
        if isinstance(names, dict):
            common = names.get("common") or []
            if isinstance(common, list) and common:
                brand_name = common[0].get("value") if isinstance(common[0], dict) else None
        if not brand_name:
            brand_name = brand_data.get("name") or brand_data.get("wikidata")

    sources_raw = raw.get("sources") or []
    sources: list[str] | None = None
    if isinstance(sources_raw, list) and sources_raw:
        sources = [
            s.get("dataset") for s in sources_raw
            if isinstance(s, dict) and s.get("dataset")
        ] or None

    overture_confidence = raw.get("confidence") if isinstance(raw.get("confidence"), (int, float)) else None

    # Prefer OSM enrichment for phone/website/opening_hours when available
    return {
        **record,
        "explanation": prediction.get("explanation", []),
        "phone": osm_fields.get("phone") or record.get("phone"),
        "website": osm_fields.get("website") or record.get("website"),
        "opening_hours": osm_fields.get("opening_hours") or record.get("opening_hours"),
        # OSM amenity fields
        "cuisine": osm_fields.get("cuisine"),
        "wheelchair": osm_fields.get("wheelchair"),
        "outdoor_seating": osm_fields.get("outdoor_seating"),
        "takeaway": osm_fields.get("takeaway"),
        "delivery": osm_fields.get("delivery"),
        "wifi": osm_fields.get("wifi"),
        "parking": osm_fields.get("parking"),
        "osm_enriched_at": osm_fields.get("osm_enriched_at"),
        # Overture fields
        "brand": brand_name,
        "sources": sources,
        "overture_confidence": overture_confidence,
    }
