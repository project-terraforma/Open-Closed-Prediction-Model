import json
from sqlalchemy import text
from .database import get_db, engine
from .predict import predict_status

def load_data_to_db():
    pass

def search_places(query: str, limit: int = 20, offset: int = 0, 
                  min_lat: float = None, max_lat: float = None, 
                  min_lon: float = None, max_lon: float = None):
    if not query or len(query.strip()) < 2:
        return []

    with engine.connect() as conn:
        try:
            bbox_sql = ""
            params = {
                "ilike_query": f"%{query}%",
                "query_str": query,
                "limit": limit,
                "offset": offset
            }
            
            # PostGIS ST_MakeEnvelope spatial indexing
            if all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
                bbox_sql = "AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)"
                params.update({
                    "min_lon": min_lon,
                    "min_lat": min_lat,
                    "max_lon": max_lon,
                    "max_lat": max_lat
                })

            sql = text(f"""
                SELECT 
                    place_id, 
                    name, 
                    category,
                    source,
                    ST_X(geom::geometry) AS lon, 
                    ST_Y(geom::geometry) AS lat, 
                    metadata_json,
                    similarity(name, :query_str) as name_sim
                FROM places 
                WHERE 
                    (name ILIKE :ilike_query OR similarity(name, :query_str) > 0.15)
                    {bbox_sql}
                ORDER BY name_sim DESC 
                LIMIT :limit OFFSET :offset;
            """)
            
            results = conn.execute(sql, params).fetchall()
            
            out = []
            for row in results:
                metadata = row.metadata_json or {}
                
                try:
                    pred = predict_status(metadata)
                    status = pred.get('status', 'unknown')
                    confidence = pred.get('confidence', 0.0)
                except Exception:
                    status = 'unknown'
                    confidence = 0.0

                location_str = f"Lon: {row.lon:.5f}, Lat: {row.lat:.5f}" if row.lon else "Unknown Location"
                if 'city' in metadata and 'state' in metadata:
                    location_str = f"{metadata.get('city', '')}, {metadata.get('state', '')}".strip(', ')

                out.append({
                    "id": str(row.place_id),
                    "name": row.name,
                    "category": row.category,
                    "source": row.source,
                    "lat": row.lat,
                    "lon": row.lon,
                    "metadata_json": metadata,
                    "address": location_str,
                    "status": status,
                    "confidence": confidence
                })
            
            return out
        except Exception as e:
            print(f"Error on pg_trgm search: {e}")
            return []

def get_place_record(place_id: str):
    """
    Look up strict POI data for the detailed ML view via Indexed PK or place_id.
    """
    with engine.connect() as conn:
        try:
            sql = text("""
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
            """)
            row = conn.execute(sql, {"place_id": place_id}).fetchone()
            if row:
                return {
                    "id": str(row.place_id),
                    "name": row.name,
                    "category": row.category,
                    "source": row.source,
                    "lat": row.lat,
                    "lon": row.lon,
                    "metadata_json": row.metadata_json or {}
                }
            return None
        except Exception as e:
            print(f"Postgres Query Error: {e}")
            return None
