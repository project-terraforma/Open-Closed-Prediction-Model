from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from .models import SearchResult, PlaceDetail
from .search import search_places, get_place_record, load_data_to_db
from .predict import predict_status

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search", response_model=List[SearchResult])
def search(q: str, limit: int = 20, offset: int = 0,
           min_lat: float = None, max_lat: float = None,
           min_lon: float = None, max_lon: float = None):
    if not q:
        return []
    return search_places(query=q, limit=limit, offset=offset,
                         min_lat=min_lat, max_lat=max_lat,
                         min_lon=min_lon, max_lon=max_lon)

@app.get("/place/{place_id}", response_model=PlaceDetail)
def get_place_details(place_id: str):
    record = get_place_record(place_id)
    if not record:
        raise HTTPException(status_code=404, detail="Place not found")
    
    metadata = record.get('metadata_json', {})
    
    # Run prediction to get status/confidence
    prediction_result = predict_status(metadata)
    
    location_str = f"Lon: {record['lon']:.5f}, Lat: {record['lat']:.5f}" if record.get('lon') else "Unknown Location"
    if 'city' in metadata and 'state' in metadata:
        location_str = f"{metadata.get('city', '')}, {metadata.get('state', '')}".strip(", ")
    
    return {
        "id": place_id,
        "name": record.get('name', 'Unknown'),
        "category": record.get('category'),
        "source": record.get('source'),
        "lat": record.get('lat'),
        "lon": record.get('lon'),
        "metadata_json": metadata,
        "address": location_str,
        "status": prediction_result['status'],
        "confidence": prediction_result['confidence'],
        "explanation": prediction_result['explanation']
    }
