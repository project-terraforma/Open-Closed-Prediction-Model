from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import Column, BigInteger, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from geoalchemy2 import Geometry, Geography
from .database import Base

# --- Pydantic API Schemas ---
class PlacePredictRequest(BaseModel):
    place_id: str

class SearchResult(BaseModel):
    id: str
    name: str
    address: str
    category: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    source: Optional[str] = None
    metadata_json: Optional[dict] = None
    status: str
    confidence: float
    website: Optional[str] = None
    opening_hours: Optional[str] = None
    photo_url: Optional[str] = None

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
    confidence: float
    explanation: List[str]
    website: Optional[str] = None
    opening_hours: Optional[str] = None
    photo_url: Optional[str] = None

# --- SQLAlchemy Database Models ---
class Place(Base):
    """
    Normalized core POI table capable of holding OSM and Overture records.
    Requires PostGIS (Geometry) and JSONB indexing.
    """
    __tablename__ = "places"
    
    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    place_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    category = Column(String, index=True, nullable=True)
    
    # Store lat/lon using Geography type for realistic real-world distance calculation (e.g. ST_DWithin)
    source = Column(String, nullable=False, default="osm")
    geom = Column(Geography(geometry_type='POINT', srid=4326, spatial_index=True), nullable=True)
    
    # Keeping raw metadata intact for Scikit-Learn feature feature engineering compatibility
    metadata_json = Column(JSONB, nullable=True)
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())
