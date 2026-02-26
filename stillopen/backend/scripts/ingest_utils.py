"""
Shared utilities for StillOpen ingestion pipeline.

Handles DB connections, category normalization, address field parsing,
and JSON merge helpers reusable across all ingestor scripts.
"""

import os
import sys
import re
import json
import logging
from typing import Optional
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

# Add backend dir to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.utils import build_address

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ingest.utils")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    """Return a raw psycopg2 connection using DATABASE_URL or defaults."""
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://localhost:5432/stillopen",
    )
    # Strip SQLAlchemy driver prefix if present
    url = url.replace("postgresql+psycopg2://", "postgresql://")
    return psycopg2.connect(url)


# ---------------------------------------------------------------------------
# OSM category → normalized category string
# ---------------------------------------------------------------------------

# Ordered lookup: OSM tag key → value prefix → normalized label
_OSM_CATEGORY_MAP = [
    ("amenity",     "restaurant",    "restaurant"),
    ("amenity",     "cafe",          "cafe"),
    ("amenity",     "bar",           "bar"),
    ("amenity",     "pub",           "pub"),
    ("amenity",     "fast_food",     "fast food"),
    ("amenity",     "food_court",    "food court"),
    ("amenity",     "bank",          "bank"),
    ("amenity",     "pharmacy",      "pharmacy"),
    ("amenity",     "hospital",      "hospital"),
    ("amenity",     "clinic",        "clinic"),
    ("amenity",     "school",        "school"),
    ("amenity",     "university",    "university"),
    ("amenity",     "library",       "library"),
    ("amenity",     "fuel",          "gas station"),
    ("amenity",     "parking",       "parking"),
    ("amenity",     "post_office",   "post office"),
    ("amenity",     "police",        "police"),
    ("amenity",     "fire_station",  "fire station"),
    ("amenity",     "gym",           "gym"),
    ("amenity",     "hotel",         "hotel"),
    ("amenity",     "supermarket",   "supermarket"),
    ("shop",        "supermarket",   "supermarket"),
    ("shop",        "grocery",       "grocery"),
    ("shop",        "bakery",        "bakery"),
    ("shop",        "clothes",       "clothing"),
    ("shop",        "electronics",   "electronics"),
    ("shop",        "hardware",      "hardware"),
    ("shop",        "books",         "bookstore"),
    ("shop",        "beauty",        "beauty salon"),
    ("shop",        "hairdresser",   "hair salon"),
    ("shop",        "convenience",   "convenience store"),
    ("shop",        "car_repair",    "auto repair"),
    ("shop",        "laundry",       "laundry"),
    ("leisure",     "gym",           "gym"),
    ("leisure",     "fitness_centre","gym"),
    ("leisure",     "park",          "park"),
    ("office",      None,            "office"),
    ("tourism",     "hotel",         "hotel"),
    ("tourism",     "motel",         "motel"),
    ("healthcare",  None,            "healthcare"),
]


def normalize_category(tags: dict) -> Optional[str]:
    """
    Map OSM tags dict to a human-readable category string.
    Returns None if nothing matches.
    """
    for key, val_prefix, label in _OSM_CATEGORY_MAP:
        tag_val = tags.get(key, "")
        if not tag_val:
            continue
        if val_prefix is None or tag_val.startswith(val_prefix):
            return label

    # Fallback: return raw amenity / shop / office value with underscores replaced
    for key in ("amenity", "shop", "office", "tourism", "healthcare", "leisure"):
        val = tags.get(key)
        if val:
            return val.replace("_", " ")

    return None


# ---------------------------------------------------------------------------
# Metadata merge (non-destructive: never overwrite a filled field)
# ---------------------------------------------------------------------------

def merge_metadata(existing: dict, incoming: dict) -> dict:
    """
    Merge `incoming` fields into `existing` only for keys that are absent or blank.
    Returns the merged dict (does NOT mutate inputs).
    """
    merged = dict(existing)
    for key, val in incoming.items():
        if val is None or val == "":
            continue
        if not merged.get(key):  # only fill if currently empty/absent
            merged[key] = val
    return merged


def normalize_phone(raw: str) -> Optional[str]:
    """
    Coerce phone numbers to E.164-ish format.
    Strips junk, keeps +, digits, spaces, dashes, parens.
    """
    if not raw:
        return None
    cleaned = re.sub(r"[^\d\+\-\(\) ]", "", raw).strip()
    return cleaned if cleaned else None


def normalize_url(raw: str) -> Optional[str]:
    """Ensure URL starts with http/https."""
    if not raw:
        return None
    raw = raw.strip()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    return raw


# ---------------------------------------------------------------------------
# Batch upsert helper
# ---------------------------------------------------------------------------

UPSERT_SQL = """
INSERT INTO places (place_id, name, category, source, geom, metadata_json)
VALUES (%(place_id)s, %(name)s, %(category)s, %(source)s,
        ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography,
        %(metadata_json)s::jsonb)
ON CONFLICT (place_id) DO UPDATE SET
    name          = COALESCE(EXCLUDED.name, places.name),
    category      = COALESCE(EXCLUDED.category, places.category),
    metadata_json = places.metadata_json || EXCLUDED.metadata_json,
    last_updated  = now()
WHERE places.place_id = EXCLUDED.place_id;
"""


def batch_upsert(conn, rows: list[dict], batch_size: int = 500):
    """
    Execute upserts in batches. Each row dict must have:
    place_id, name, category, source, lat, lon, metadata_json (dict).
    """
    total = 0
    with conn.cursor() as cur:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            for row in batch:
                row["metadata_json"] = json.dumps(row["metadata_json"])
            psycopg2.extras.execute_batch(cur, UPSERT_SQL, batch)
            conn.commit()
            total += len(batch)
            logger.info("  Upserted %d rows (total so far: %d)", len(batch), total)
    return total
