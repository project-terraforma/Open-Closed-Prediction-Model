from sqlalchemy import text
from .database import engine
from .predict import predict_status
from utils.canonical_metadata import build_canonical_metadata


def _extract_place_info(row, metadata: dict) -> dict:
    """
    Returns a fully-populated dict from a DB row + metadata_json.
    All fields that are absent default to None – no placeholders generated.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    if "canonical" in metadata and "raw" in metadata:
        canonical = metadata.get("canonical") or {}
        raw = metadata.get("raw") or {}
    else:
        # Backward-compat: older rows may store a flat metadata dict
        raw = metadata
        canonical = build_canonical_metadata(raw, lat=getattr(row, "lat", None), lon=getattr(row, "lon", None))

    try:
        pred = predict_status(raw)
        status = pred.get("status", "unknown")
        confidence = pred.get("confidence", 0.0)
    except Exception:
        status = "unknown"
        confidence = 0.0

    address = canonical.get("formatted_address") or ""
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
        "website": website,
        "phone": phone,
        "opening_hours": opening_hours,
        "photo_url": photo_url,
    }


def load_data_to_db():
    pass  # Data is ingested offline via osm2pgsql / Overture scripts


def search_places(
    query: str,
    limit: int = 20,
    offset: int = 0,
    min_lat: float = None,
    max_lat: float = None,
    min_lon: float = None,
    max_lon: float = None,
):
    if not query or len(query.strip()) < 2:
        return []

    with engine.connect() as conn:
        try:
            bbox_sql = ""
            params: dict = {
                "ilike_query": f"%{query}%",
                "query_str": query,
                "limit": limit,
                "offset": offset,
            }

            if all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
                bbox_sql = "AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)"
                params.update(
                    {
                        "min_lon": min_lon,
                        "min_lat": min_lat,
                        "max_lon": max_lon,
                        "max_lat": max_lat,
                    }
                )

            sql = text(
                f"""
                SELECT
                    place_id,
                    name,
                    category,
                    source,
                    ST_X(geom::geometry) AS lon,
                    ST_Y(geom::geometry) AS lat,
                    metadata_json,
                    similarity(name, :query_str) AS name_sim
                FROM places
                WHERE
                    (name ILIKE :ilike_query OR similarity(name, :query_str) > 0.15)
                    {bbox_sql}
                ORDER BY name_sim DESC
                LIMIT :limit OFFSET :offset;
                """
            )

            results = conn.execute(sql, params).fetchall()
            return [_extract_place_info(row, row.metadata_json or {}) for row in results]

        except Exception as e:
            print(f"Error on pg_trgm search: {e}")
            return []


def get_place_record(place_id: str):
    """Full POI record for the detail view."""
    with engine.connect() as conn:
        try:
            sql = text(
                """
                SELECT
                    place_id,
                    name,
                    category,
                    source,
                    ST_X(geom::geometry) AS lon,
                    ST_Y(geom::geometry) AS lat,
                    metadata_json
                FROM places
                WHERE place_id = :place_id
                """
            )
            row = conn.execute(sql, {"place_id": place_id}).fetchone()
            if row:
                return _extract_place_info(row, row.metadata_json or {})
            return None
        except Exception as e:
            print(f"Postgres Query Error: {e}")
            return None
