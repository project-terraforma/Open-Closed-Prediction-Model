from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from .models import SearchResult, PlaceDetail
from .search import search_places, get_place_record, load_data_to_db

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
def search(q: str = "", limit: int = 20, offset: int = 0,
           min_lat: float = None, max_lat: float = None,
           min_lon: float = None, max_lon: float = None):
    return search_places(query=q, limit=limit, offset=offset,
                         min_lat=min_lat, max_lat=max_lat,
                         min_lon=min_lon, max_lon=max_lon)

@app.get("/place/{place_id}", response_model=PlaceDetail)
def get_place_details(place_id: str):
    record = get_place_record(place_id)
    if not record:
        raise HTTPException(status_code=404, detail="Place not found")

    # get_place_record already runs predict_status inside _extract_place_info.
    # We only need to attach the explanation list for the detail view.
    from .predict import predict_status
    meta = record.get("metadata_json", {})
    raw = meta.get("raw", meta) if isinstance(meta, dict) else {}
    prediction = predict_status(raw)

    return {
        **record,
        "explanation": prediction.get("explanation", []),
    }
