from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, BigInteger
from sqlalchemy.sql import func
from .database import Base, IS_POSTGRES

# --- Pydantic API Schemas ---
class PlacePredictRequest(BaseModel):
    place_id: str

class SearchResult(BaseModel):
    id: str
    name: str
    address: str
    category: Optional[str] = None
    parent_category: Optional[str] = None    # mapped parent group from CATEGORY_TREE
    city: Optional[str] = None
    state: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    source: Optional[str] = None
    # metadata_json intentionally omitted from list results — see /place/{id} for full detail
    status: str
    confidence: Optional[float] = None        # null when prediction_type == "likely_open"
    prediction_type: Optional[str] = None     # "open"|"likely_open"|"closed"
    website: Optional[str] = None
    phone: Optional[str] = None
    website_status: Optional[str] = None       # "active"|"likely_closed"|"inconclusive"|"unchecked"
    website_checked_at: Optional[str] = None   # ISO-8601 timestamp of last verification
    website_http_code: Optional[int] = None    # HTTP status code from last check


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    page: int
    total_pages: int
    limit: int
    offset: int
    has_next: bool
    has_prev: bool
    category_counts: Optional[dict] = None   # parent_category → total count for sidebar

class PlaceDetail(BaseModel):
    id: str
    name: str
    address: str
    category: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    source: Optional[str] = None
    metadata_json: Optional[dict] = None
    status: str
    confidence: Optional[float] = None        # null when prediction_type == "likely_open"
    prediction_type: Optional[str] = None     # "open"|"likely_open"|"closed"
    explanation: List[str]
    website: Optional[str] = None
    phone: Optional[str] = None
    opening_hours: Optional[str] = None
    photo_url: Optional[str] = None
    website_status: Optional[str] = None
    website_checked_at: Optional[str] = None
    website_http_code: Optional[int] = None
    # OSM Overpass enrichment
    cuisine: Optional[str] = None
    wheelchair: Optional[str] = None
    outdoor_seating: Optional[str] = None
    takeaway: Optional[str] = None
    delivery: Optional[str] = None
    wifi: Optional[str] = None
    parking: Optional[str] = None
    osm_enriched_at: Optional[str] = None
    # Overture data already in DB
    brand: Optional[str] = None
    sources: Optional[List[str]] = None
    overture_confidence: Optional[float] = None


# --- SQLAlchemy Database Models ---
# Conditional imports for PostgreSQL-specific types
if IS_POSTGRES:
    from sqlalchemy.dialects.postgresql import JSONB
    from geoalchemy2 import Geometry

    class Place(Base):
        """
        Production POI table with PostGIS geometry and JSONB metadata.
        Used when DATABASE_URL points to PostgreSQL.
        """
        __tablename__ = "places"

        id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
        place_id = Column(String, unique=True, index=True, nullable=False)
        name = Column(String, index=True, nullable=False)
        category = Column(String, index=True, nullable=True)
        address = Column(String, nullable=True)
        source = Column(String, nullable=False, default="osm")
        geom = Column(
            Geometry(geometry_type='POINT', srid=4326, spatial_index=True),
            nullable=True
        )
        metadata_json = Column(JSONB, nullable=True)
        last_updated = Column(DateTime, server_default=func.now())
else:
    class Place(Base):
        """
        Local development POI table using standard SQLite-compatible types.
        Used when no DATABASE_URL is set.
        """
        __tablename__ = "places"

        id = Column(Integer, primary_key=True, index=True, autoincrement=True)
        place_id = Column(String, unique=True, index=True, nullable=False)
        name = Column(String, index=True, nullable=False)
        category = Column(String, index=True, nullable=True)
        address = Column(String, nullable=True)
        lat = Column(Float, nullable=True)
        lon = Column(Float, nullable=True)
        source = Column(String, nullable=False, default="overture")
        metadata_json = Column(Text, nullable=True)  # JSON string for SQLite
        last_updated = Column(DateTime, server_default=func.now())
