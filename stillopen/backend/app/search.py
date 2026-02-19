from sqlalchemy import text
from .database import engine
from .predict import predict_status


def _build_address(metadata: dict) -> str:
    """
    Compose a human-readable address from OSM addr:* fields stored in metadata_json.
    Falls back to city/state, then lat/lon coords if nothing else is found.
    """
    parts = []

    # Street-level: number + street
    house = metadata.get("addr:housenumber", "")
    street = metadata.get("addr:street", "")
    if house and street:
        parts.append(f"{house} {street}")
    elif street:
        parts.append(street)

    # City / municipality
    city = (
        metadata.get("addr:city")
        or metadata.get("addr:municipality")
        or metadata.get("city")
    )
    if city:
        parts.append(city)

    # State / province
    state = metadata.get("addr:state") or metadata.get("state")
    if state:
        parts.append(state)

    # Postcode
    postcode = metadata.get("addr:postcode") or metadata.get("postcode")
    if postcode:
        parts.append(str(postcode))

    return ", ".join(parts) if parts else ""


def _extract_place_info(row, metadata: dict) -> dict:
    """
    Returns a fully-populated dict from a DB row + metadata_json.
    All fields that are absent default to None – no placeholders generated.
    """
    try:
        pred = predict_status(metadata)
        status = pred.get("status", "unknown")
        confidence = pred.get("confidence", 0.0)
    except Exception:
        status = "unknown"
        confidence = 0.0

    address = _build_address(metadata)
    if not address and row.lat and row.lon:
        address = f"{row.lat:.5f}°N, {row.lon:.5f}°E"

    # Pull every useful contact/info field from metadata
    website = metadata.get("website") or metadata.get("contact:website")
    phone = (
        metadata.get("phone")
        or metadata.get("contact:phone")
        or metadata.get("telephone")
    )
    opening_hours = metadata.get("opening_hours")
    photo_url = metadata.get("photo_url") or None  # explicit None if absent

    return {
        "id": str(row.place_id),
        "name": row.name,
        "category": row.category,
        "source": row.source,
        "lat": row.lat,
        "lon": row.lon,
        "metadata_json": metadata,
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
