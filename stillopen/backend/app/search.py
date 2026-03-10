import json
import os
import re
import time
from collections import defaultdict
import pandas as pd
import numpy as np
from sqlalchemy import text, insert

from .database import engine, Base, IS_POSTGRES
from .models import Place
from .predict import predict_place, predict_status, predict_batch
from .utils import reverse_geocode
from utils.canonical_metadata import build_canonical_metadata
from .categories import get_parent_category, get_children, all_mapped_categories

# ---------------------------------------------------------------------------
# Simple in-memory query cache (TTL = 60 s, max 256 entries)
# ---------------------------------------------------------------------------
_CACHE_TTL = 60.0
_CACHE_MAX = 256
_cache: dict = {}  # key -> (timestamp, result)


def _cache_get(key):
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key, value):
    if len(_cache) >= _CACHE_MAX:
        # Evict oldest entry
        oldest = min(_cache, key=lambda k: _cache[k][0])
        del _cache[oldest]
    _cache[key] = (time.monotonic(), value)


def _extract_place_info(row, metadata: dict, address_override: str = None, pred: dict = None) -> dict:
    """
    Returns a fully-populated dict from a DB row + metadata_json.
    Pass `pred` to skip running the ML model (used by batch search path).
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    if "canonical" in metadata and "raw" in metadata:
        canonical = metadata.get("canonical") or {}
        raw = metadata.get("raw") or {}
    else:
        raw = metadata
        canonical = build_canonical_metadata(raw, lat=getattr(row, "lat", None), lon=getattr(row, "lon", None))

    if pred is None:
        try:
            pred = predict_status(raw)
        except Exception:
            pred = {}
    status = pred.get("status", "unknown")
    confidence = pred.get("confidence", 0.0)

    address = address_override or getattr(row, "address", None) or canonical.get("formatted_address") or ""
    website = canonical.get("website")
    phone = canonical.get("international_phone_number")

    opening_hours = None
    oh = canonical.get("opening_hours")
    if isinstance(oh, dict):
        wt = oh.get("weekday_text")
        if isinstance(wt, list) and wt:
            opening_hours = " | ".join(str(x) for x in wt if x is not None)

    photo_url = None
    photos = canonical.get("photos")
    if isinstance(photos, list) and photos:
        first = photos[0]
        if isinstance(first, dict):
            photo_url = first.get("photo_reference")

    return {
        "id": str(row.place_id),
        "name": canonical.get("name") or row.name,
        "category": row.category,
        "source": row.source,
        "lat": row.lat,
        "lon": row.lon,
        "metadata_json": raw,
        "address": address,
        "status": status,
        "confidence": confidence,
        "prediction_type": pred.get("prediction_type"),
        "website": website,
        "phone": phone,
        "opening_hours": opening_hours,
        "photo_url": photo_url,
        "website_status": raw.get("website_status"),
        "website_checked_at": raw.get("website_checked_at"),
        "website_http_code": raw.get("website_http_code"),
    }


# =============================================================================
# DATA SEEDING
# =============================================================================

def _records_from_overture_parquet(df, id_prefix="overture"):
    """Convert an Overture-format parquet DataFrame into DB record dicts."""
    records = []
    for idx, row in df.iterrows():
        names = row.get('names')
        name = 'Unknown'
        if isinstance(names, dict):
            name = names.get('primary', 'Unknown') or 'Unknown'

        cats = row.get('categories')
        category = 'unknown'
        if isinstance(cats, dict):
            category = cats.get('primary', 'unknown') or 'unknown'

        addrs = row.get('addresses')
        address_str = ''
        if isinstance(addrs, (list, np.ndarray)) and len(addrs) > 0:
            addr = addrs[0]
            if isinstance(addr, dict):
                parts = []
                if addr.get('freeform'):
                    parts.append(addr['freeform'])
                if addr.get('locality'):
                    parts.append(addr['locality'])
                if addr.get('region'):
                    parts.append(addr['region'])
                address_str = ', '.join(parts)

        # Extract lat/lon from bbox field (xmin/xmax=lon, ymin/ymax=lat)
        lat = None
        lon = None
        bbox_data = row.get('bbox')
        if isinstance(bbox_data, dict):
            try:
                ymin = bbox_data.get('ymin')
                ymax = bbox_data.get('ymax')
                xmin = bbox_data.get('xmin')
                xmax = bbox_data.get('xmax')
                if ymin is not None and ymax is not None:
                    lat = (float(ymin) + float(ymax)) / 2.0
                if xmin is not None and xmax is not None:
                    lon = (float(xmin) + float(xmax)) / 2.0
            except (TypeError, ValueError):
                pass

        metadata = {}
        for col in ['websites', 'socials', 'emails', 'phones', 'brand', 'addresses', 'sources', 'categories']:
            val = row.get(col)
            if val is not None:
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                metadata[col] = val

        metadata['confidence'] = float(row.get('confidence', 0))
        metadata['open'] = int(row.get('open', 1)) if 'open' in row.index else 1

        if isinstance(addrs, (list, np.ndarray)) and len(addrs) > 0:
            addr = addrs[0]
            if isinstance(addr, dict):
                metadata['city'] = addr.get('locality', '')
                metadata['state'] = addr.get('region', '')

        records.append({
            'place_id': str(row.get('id', f'{id_prefix}_{idx}')),
            'name': name,
            'category': category,
            'address': address_str,
            'lat': lat,
            'lon': lon,
            'source': 'overture',
            'metadata_json': json.dumps(metadata, default=str),
        })
    return records


def _records_from_osm_json(data):
    """Convert OSM JSON records into DB record dicts."""
    records = []
    for item in data:
        meta = item.get('metadata', {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        metadata = {
            'websites': meta.get('websites', []),
            'socials': meta.get('socials', []),
            'phones': meta.get('phones', []),
            'confidence': float(meta.get('confidence', 0.5)),
            'open': int(item.get('open', 1)),
            'city': '',
            'state': '',
        }

        addr = item.get('address', '')
        parts = [p.strip() for p in addr.split(',')]
        if len(parts) >= 2:
            metadata['city'] = parts[-2] if len(parts) > 1 else ''
            metadata['state'] = parts[-1] if len(parts) > 0 else ''

        records.append({
            'place_id': str(item.get('id', f"osm_{item.get('name', '')}_{len(records)}")),
            'name': item.get('name', 'Unknown'),
            'category': item.get('category', 'unknown'),
            'address': addr,
            'lat': item.get('lat'),
            'lon': item.get('lon'),
            'source': 'osm',
            'metadata_json': json.dumps(metadata, default=str),
        })
    return records


def load_data_to_db():
    """
    Seeds the database on startup.
    - PostgreSQL: No-op
    - SQLite: Seeds from project_c_samples.parquet, then supplements with
      scripts/data/overture_places_us.parquet and scripts/data/osm_places.json
      if present, giving a much richer dataset including closed places.
    """
    if IS_POSTGRES:
        return

    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM places")).fetchone()[0]
        if count > 0:
            has_coords = conn.execute(
                text("SELECT COUNT(*) FROM places WHERE lat IS NOT NULL")
            ).fetchone()[0]
            if has_coords > 0:
                return
            print("Detected old seed with no coordinates — re-seeding...")

    # --- Source 1: original project parquet ---
    backend_root = os.path.dirname(os.path.dirname(__file__))
    project_root = os.path.normpath(os.path.join(backend_root, "..", ".."))

    data_path = os.path.normpath(os.path.join(project_root, "data", "project_c_samples.parquet"))
    if not os.path.exists(data_path):
        data_path = os.path.normpath(os.path.join(backend_root, "..", "data", "project_c_samples.parquet"))

    records = []
    seen_ids = set()

    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        orig_records = _records_from_overture_parquet(df, id_prefix="orig")
        for r in orig_records:
            seen_ids.add(r['place_id'])
            records.append(r)
        print(f"  Loaded {len(orig_records)} places from project_c_samples.parquet")

    # --- Source 2: Overture US parquet (large dataset) ---
    overture_us_path = os.path.normpath(os.path.join(project_root, "scripts", "data", "overture_places_us.parquet"))
    if os.path.exists(overture_us_path):
        df_us = pd.read_parquet(overture_us_path)
        overture_records = _records_from_overture_parquet(df_us, id_prefix="overture_us")
        added = 0
        for r in overture_records:
            if r['place_id'] not in seen_ids:
                seen_ids.add(r['place_id'])
                records.append(r)
                added += 1
        print(f"  Loaded {added} additional places from overture_places_us.parquet")

    # --- Source 3: OSM places (includes closed businesses) ---
    osm_path = os.path.normpath(os.path.join(project_root, "scripts", "data", "osm_places.json"))
    if os.path.exists(osm_path):
        with open(osm_path) as f:
            osm_data = json.load(f)
        osm_records = _records_from_osm_json(osm_data)
        added = 0
        for r in osm_records:
            if r['place_id'] not in seen_ids:
                seen_ids.add(r['place_id'])
                records.append(r)
                added += 1
        closed_count = sum(1 for item in osm_data if item.get('open') == 0)
        print(f"  Loaded {added} additional places from osm_places.json ({closed_count} marked closed)")

    if not records:
        print("No data sources found — DB will be empty.")
        return

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM places"))
        conn.execute(insert(Place), records)
    print(f"Seeded {len(records)} total places into DB.")


# =============================================================================
# INDEX MANAGEMENT
# =============================================================================

_indexes_ensured = False


def ensure_indexes():
    """Create search/spatial indexes if they don't already exist. Safe to call repeatedly."""
    global _indexes_ensured
    if _indexes_ensured:
        return
    if not IS_POSTGRES:
        _indexes_ensured = True
        return
    print("Ensuring DB indexes...")
    ddl_statements = [
        """CREATE INDEX IF NOT EXISTS places_geom_idx
           ON places USING GIST(geom)""",
        """CREATE INDEX IF NOT EXISTS places_fts_idx
           ON places USING GIN(
               to_tsvector('english', coalesce(name,'') || ' ' || coalesce(category,''))
           )""",
        """CREATE INDEX IF NOT EXISTS places_category_idx
           ON places(category)""",
        """CREATE INDEX IF NOT EXISTS places_city_meta_idx
           ON places((metadata_json->>'city'))""",
        """CREATE INDEX IF NOT EXISTS places_parent_cat_idx
           ON places(category)""",
    ]
    try:
        with engine.begin() as conn:
            for ddl in ddl_statements:
                conn.execute(text(ddl))
        print("Indexes ready.")
    except Exception as e:
        print(f"Index creation warning (non-fatal): {e}")
    _indexes_ensured = True


# =============================================================================
# POSTGRES SEARCH — HELPERS
# =============================================================================

def _make_tsquery(q: str):
    """Convert a user query string into a prefix-matching tsquery string.

    "rest" → "rest:*"
    "santa cruz coffee" → "santa:* & cruz:* & coffee:*"
    Returns None if no valid words found.
    """
    words = re.findall(r"[a-zA-Z0-9]+", q)
    if not words:
        return None
    return " & ".join(f"{w.lower()}:*" for w in words)


# =============================================================================
# POSTGRES SEARCH
# =============================================================================

def _search_postgres(
    query: str,
    city: str = None,
    limit: int = 50,
    offset: int = 0,
    min_lat: float = None,
    max_lat: float = None,
    min_lon: float = None,
    max_lon: float = None,
    parent_category: str = None,
):
    """Returns (results_list, total_count, category_counts)."""
    has_bbox = all(v is not None for v in [min_lat, max_lat, min_lon, max_lon])
    has_query = bool(query and query.strip())
    has_city = bool(city and len(city.strip()) >= 2)

    with engine.connect() as conn:
        try:
            params: dict = {"limit": limit, "offset": offset}
            where_parts: list = []
            order_parts: list = []

            # --- Text search via FTS + ILIKE fallback ---
            if has_query:
                tsq = _make_tsquery(query)
                if tsq:
                    where_parts.append(
                        "(to_tsvector('english', coalesce(name,'') || ' ' || coalesce(category,''))"
                        " @@ to_tsquery('english', :tsq)"
                        " OR name ILIKE :ilike_q)"
                    )
                    params["tsq"] = tsq
                    order_parts.append(
                        "ts_rank("
                        "  to_tsvector('english', coalesce(name,'') || ' ' || coalesce(category,'')),"
                        "  to_tsquery('english', :tsq)"
                        ") DESC"
                    )
                else:
                    where_parts.append("name ILIKE :ilike_q")
                params["ilike_q"] = f"%{query.strip()}%"

            # --- City filter (metadata city field + address) ---
            if has_city:
                where_parts.append(
                    "(address ILIKE :ilike_city"
                    " OR metadata_json->>'city' ILIKE :ilike_city"
                    " OR metadata_json->>'addr:city' ILIKE :ilike_city)"
                )
                params["ilike_city"] = f"%{city.strip()}%"

            # --- Bounding-box spatial filter ---
            if has_bbox:
                where_parts.append(
                    "geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)"
                )
                params.update({
                    "min_lon": min_lon, "min_lat": min_lat,
                    "max_lon": max_lon, "max_lat": max_lat,
                })

            # --- Parent category filter ---
            if parent_category and parent_category != "Other":
                children = get_children(parent_category)
                if children:
                    cat_params = {f"_cat_{i}": c for i, c in enumerate(children)}
                    placeholders = ", ".join(f":_cat_{i}" for i in range(len(children)))
                    where_parts.append(f"LOWER(category) IN ({placeholders})")
                    params.update(cat_params)
            elif parent_category == "Other":
                mapped = list(all_mapped_categories())
                if mapped:
                    cat_params = {f"_cat_{i}": c for i, c in enumerate(mapped)}
                    placeholders = ", ".join(f":_cat_{i}" for i in range(len(mapped)))
                    where_parts.append(f"(category IS NULL OR LOWER(category) NOT IN ({placeholders}))")
                    params.update(cat_params)

            where_clause = " AND ".join(where_parts) if where_parts else "TRUE"

            # Fallback ordering when no text query
            if not order_parts:
                order_parts.append("(metadata_json->>'confidence')::numeric DESC NULLS LAST")
                order_parts.append("name ASC")
            order_clause = ", ".join(order_parts)

            # --- Category counts for sidebar (aggregation, separate from pagination) ---
            cat_count_sql = text(
                f"SELECT category, COUNT(*) AS n FROM places WHERE {where_clause} GROUP BY category"
            )
            cat_rows = conn.execute(cat_count_sql, params).fetchall()
            parent_counts: dict[str, int] = defaultdict(int)
            for cr in cat_rows:
                parent = get_parent_category(cr.category or "")
                parent_counts[parent] += cr.n
            category_counts = dict(parent_counts)

            # --- COUNT for pagination ---
            count_sql = text(f"SELECT COUNT(*) FROM places WHERE {where_clause}")
            total_count = conn.execute(count_sql, params).scalar() or 0

            # --- Lean SELECT — fetch metadata_json for ML batch prediction ---
            data_sql = text(f"""
                SELECT
                    place_id,
                    name,
                    category,
                    address,
                    source,
                    ST_X(geom::geometry) AS lon,
                    ST_Y(geom::geometry) AS lat,
                    COALESCE(metadata_json->>'city', metadata_json->>'addr:city', '') AS city,
                    COALESCE(metadata_json->>'state', metadata_json->>'addr:state', '') AS state,
                    metadata_json->>'website_status'    AS website_status,
                    metadata_json->>'website_checked_at' AS website_checked_at,
                    (metadata_json->>'website_http_code')::int AS website_http_code,
                    metadata_json
                FROM places
                WHERE {where_clause}
                ORDER BY {order_clause}
                LIMIT :limit OFFSET :offset
            """)

            rows = conn.execute(data_sql, params).fetchall()

            # --- Batch ML prediction ---
            raw_metadatas = []
            for row in rows:
                meta = row.metadata_json or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                raw_metadatas.append(meta)

            preds = predict_batch(raw_metadatas)

            results = []
            for row, meta, pred in zip(rows, raw_metadatas, preds):
                def _first_value(arr):
                    if not isinstance(arr, list) or not arr:
                        return None
                    first = arr[0]
                    if isinstance(first, dict):
                        return first.get("value") or first.get("phone") or first.get("url")
                    return str(first) if first else None

                phone = meta.get("phone") or _first_value(meta.get("phones"))
                website = meta.get("website") or _first_value(meta.get("websites"))
                results.append({
                    "id": str(row.place_id),
                    "name": row.name or "",
                    "category": row.category,
                    "parent_category": get_parent_category(row.category or ""),
                    "address": row.address or "",
                    "city": row.city or "",
                    "state": row.state or "",
                    "source": row.source,
                    "lat": row.lat,
                    "lon": row.lon,
                    "status": pred.get("status", "unknown"),
                    "confidence": pred.get("confidence"),
                    "prediction_type": pred.get("prediction_type"),
                    "phone": phone,
                    "website": website,
                    "website_status": row.website_status,
                    "website_checked_at": row.website_checked_at,
                    "website_http_code": row.website_http_code,
                })

            return results, total_count, category_counts

        except Exception as e:
            print(f"PostgreSQL search error: {e}")
            import traceback; traceback.print_exc()
            return [], 0, {}


# =============================================================================
# SQLITE SEARCH
# =============================================================================

def _search_sqlite(
    query: str,
    limit: int = 50,
    offset: int = 0,
    min_lat: float = None,
    max_lat: float = None,
    min_lon: float = None,
    max_lon: float = None,
    parent_category: str = None,
):
    """Returns (results_list, total_count, category_counts)."""
    has_bbox = all(v is not None for v in [min_lat, max_lat, min_lon, max_lon])
    has_query = bool(query and query.strip())

    with engine.connect() as conn:
        try:
            where_parts = []
            params: dict = {"limit": limit, "offset": offset}

            if has_query:
                where_parts.append("(name LIKE :ilike_query OR category LIKE :ilike_query)")
                params["ilike_query"] = f"%{query}%"
                params["exact_query"] = query
                params["start_query"] = f"{query}%"

            if has_bbox:
                where_parts.append(
                    "lat BETWEEN :min_lat AND :max_lat AND lon BETWEEN :min_lon AND :max_lon"
                )
                params.update({
                    "min_lat": min_lat, "max_lat": max_lat,
                    "min_lon": min_lon, "max_lon": max_lon,
                })

            # parent_category filter (SQLite — simple LIKE approach)
            if parent_category and parent_category != "Other":
                children = get_children(parent_category)
                if children:
                    # SQLite doesn't support :list params, build literal list
                    placeholders = ", ".join(f"''{c}''" for c in children)
                    where_parts.append(f"LOWER(category) IN ({placeholders})")

            where_clause = " AND ".join(where_parts) if where_parts else "1=1"

            if has_query:
                order_clause = (
                    "CASE WHEN LOWER(name) = LOWER(:exact_query) THEN 0"
                    "     WHEN LOWER(name) LIKE LOWER(:start_query) THEN 1"
                    "     ELSE 2 END, name"
                )
            else:
                order_clause = "name"

            count_sql = text(f"SELECT COUNT(*) FROM places WHERE {where_clause}")
            total_count = conn.execute(count_sql, params).scalar() or 0

            sql = text(f"""
                SELECT place_id, name, address, category, source, lat, lon, metadata_json
                FROM places
                WHERE {where_clause}
                ORDER BY {order_clause}
                LIMIT :limit OFFSET :offset
            """)

            rows = conn.execute(sql, params).fetchall()

            metadatas = []
            for row in rows:
                meta = {}
                if row.metadata_json:
                    try:
                        meta = json.loads(row.metadata_json)
                    except Exception:
                        pass
                metadatas.append(meta)

            preds = predict_batch(metadatas)

            out = []
            cat_counts: dict[str, int] = defaultdict(int)
            for row, meta, pred in zip(rows, metadatas, preds):
                city = meta.get("city", "")
                state = meta.get("state", "")
                raw_cat = getattr(row, "category", None)
                parent = get_parent_category(raw_cat or "")
                cat_counts[parent] += 1
                out.append({
                    "id": str(row.place_id),
                    "name": row.name or "",
                    "address": row.address or "",
                    "city": city,
                    "state": state,
                    "category": raw_cat,
                    "parent_category": parent,
                    "source": getattr(row, "source", None),
                    "lat": getattr(row, "lat", None),
                    "lon": getattr(row, "lon", None),
                    "status": pred.get("status", "unknown"),
                    "confidence": pred.get("confidence"),
                    "prediction_type": pred.get("prediction_type"),
                    "phone": meta.get("phone"),
                    "website": meta.get("website"),
                    "website_status": meta.get("website_status"),
                    "website_checked_at": meta.get("website_checked_at"),
                    "website_http_code": meta.get("website_http_code"),
                })

            return out, total_count, dict(cat_counts)
        except Exception as e:
            print(f"SQLite search error: {e}")
            return [], 0, {}


def search_places(
    query: str,
    city: str = None,
    limit: int = 50,
    offset: int = 0,
    page: int = 1,
    min_lat: float = None,
    max_lat: float = None,
    min_lon: float = None,
    max_lon: float = None,
    parent_category: str = None,
) -> dict:
    """Return a SearchResponse-compatible dict with results + pagination metadata."""
    limit = max(1, min(limit, 1000))

    cache_key = (query, city, limit, offset, min_lat, max_lat, min_lon, max_lon, parent_category)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if IS_POSTGRES:
        results, total_count, category_counts = _search_postgres(
            query,
            city=city,
            limit=limit,
            offset=offset,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            parent_category=parent_category,
        )
    else:
        results, total_count, category_counts = _search_sqlite(
            query, limit, offset,
            min_lat=min_lat, max_lat=max_lat,
            min_lon=min_lon, max_lon=max_lon,
            parent_category=parent_category,
        )

    total_pages = max(1, (total_count + limit - 1) // limit)

    response = {
        "results": results,
        "total_count": total_count,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "offset": offset,
        "has_next": page < total_pages,
        "has_prev": page > 1,
        "category_counts": category_counts,
    }
    _cache_set(cache_key, response)
    return response


# =============================================================================
# PLACE DETAIL LOOKUP
# =============================================================================

def _get_metadata_postgres(place_id: str):
    with engine.connect() as conn:
        try:
            sql = text("SELECT metadata_json, name FROM places WHERE place_id = :place_id")
            row = conn.execute(sql, {"place_id": place_id}).fetchone()
            if row and row.metadata_json:
                metadata = row.metadata_json
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                metadata['name'] = row.name
                return metadata
            return None
        except Exception as e:
            print(f"PostgreSQL query error: {e}")
            return None


def _get_metadata_sqlite(place_id: str):
    with engine.connect() as conn:
        try:
            sql = text("SELECT metadata_json, name, address FROM places WHERE place_id = :place_id")
            row = conn.execute(sql, {"place_id": place_id}).fetchone()
            if row and row.metadata_json:
                metadata = json.loads(row.metadata_json)
                metadata['name'] = row.name
                if row.address:
                    metadata['address_display'] = row.address
                return metadata
            return None
        except Exception as e:
            print(f"SQLite query error: {e}")
            return None


def get_place_metadata(place_id: str):
    if IS_POSTGRES:
        return _get_metadata_postgres(place_id)
    else:
        return _get_metadata_sqlite(place_id)


def get_place_record(place_id: str):
    """
    Return a fully-populated place dict (with prediction, canonical fields)
    for the /place/{place_id} detail route, or None if not found.
    """
    if IS_POSTGRES:
        return _get_place_record_postgres(place_id)
    else:
        return _get_place_record_sqlite(place_id)


def _get_place_record_postgres(place_id: str):
    with engine.connect() as conn:
        try:
            sql = text("""
                SELECT
                    place_id, name, category, address, source,
                    ST_X(geom::geometry) AS lon,
                    ST_Y(geom::geometry) AS lat,
                    metadata_json
                FROM places
                WHERE place_id = :place_id
            """)
            row = conn.execute(sql, {"place_id": place_id}).fetchone()
            if not row:
                return None
            metadata = row.metadata_json or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            # --- JIT Reverse Geocoding ---
            # If address is missing, try to fetch it and update DB
            current_address = getattr(row, "address", None)
            if not current_address and row.lat and row.lon:
                try:
                    new_address = reverse_geocode(row.lat, row.lon)
                    if new_address:
                        # Update database
                        with engine.begin() as update_conn:
                            update_conn.execute(
                                text("UPDATE places SET address = :addr, metadata_json = metadata_json || jsonb_build_object('address', :addr) WHERE place_id = :pid"),
                                {"addr": new_address, "pid": place_id}
                            )
                        # Re-fetch row or just manually update current_address
                        current_address = new_address
                except Exception as e:
                    print(f"JIT Geocoding failed: {e}")

            # Return info (manually inject current_address into row if we re-fetch, 
            # or just rely on _extract_place_info using row.address)
            return _extract_place_info(row, metadata, address_override=current_address)
        except Exception as e:
            print(f"PostgreSQL get_place_record error: {e}")
            return None


def _get_place_record_sqlite(place_id: str):
    with engine.connect() as conn:
        try:
            sql = text("""
                SELECT place_id, name, address, category, source, lat, lon, metadata_json
                FROM places
                WHERE place_id = :place_id
            """)
            row = conn.execute(sql, {"place_id": place_id}).fetchone()
            if not row:
                return None

            metadata = {}
            if row.metadata_json:
                try:
                    metadata = json.loads(row.metadata_json)
                except Exception:
                    pass

            try:
                pred = predict_place(metadata)
                status = pred.get('status', 'unknown')
                confidence = pred.get('confidence')
                prediction_type = pred.get('prediction_type')
            except Exception:
                status = 'unknown'
                confidence = None
                prediction_type = None

            return {
                "id": str(row.place_id),
                "name": row.name,
                "address": row.address or "",
                "category": getattr(row, 'category', None),
                "source": getattr(row, 'source', None),
                "lat": getattr(row, 'lat', None),
                "lon": getattr(row, 'lon', None),
                "metadata_json": metadata,
                "status": status,
                "confidence": confidence,
                "prediction_type": prediction_type,
                "website_status": metadata.get("website_status"),
                "website_checked_at": metadata.get("website_checked_at"),
                "website_http_code": metadata.get("website_http_code"),
            }
        except Exception as e:
            print(f"SQLite get_place_record error: {e}")
            return None