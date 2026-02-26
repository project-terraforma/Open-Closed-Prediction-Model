"""
ingest_places.py — Ingest Overture Maps POI data using DuckDB and Pandas.

This script queries Overture Maps GeoParquet files directly over HTTPS using DuckDB,
processes and normalizes the data with Pandas, and upserts it into the PostGIS 
'places' table.

Usage:
    python overture_ingest/ingest_places.py [--release 2026-02-18.0] [--limit 1000]

Requirements:
    duckdb, pandas, pyarrow, sqlalchemy, psycopg2-binary
"""

import os
import sys
import json
import logging
import argparse
import duckdb
import pandas as pd
from dotenv import load_dotenv

# Add backend dir to path for internal imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_utils import (
    get_conn,
    batch_upsert,
    logger as base_logger
)
from utils.canonical_metadata import build_canonical_metadata, validate_canonical_metadata

logger = logging.getLogger("ingest.overture")

# ---------------------------------------------------------------------------
# Configuration Defaults
# ---------------------------------------------------------------------------
DEFAULT_RELEASE = '2026-02-18.0'
DEFAULT_THEME = 'places'
DEFAULT_TYPE = 'place'
# Overture uses Azure Blob Storage or AWS S3. S3 is generally more reliable for DuckDB globbing.
BASE_URL = 's3://overturemaps-us-west-2/release'

def fetch_overture_data(release: str, theme: str, p_type: str, limit: int = 1000, bbox: dict = None) -> pd.DataFrame:
    """
    Query Overture Maps Parquet files directly via DuckDB over S3.
    Supports basic spatial filtering if bbox is provided.
    """
    url = f"{BASE_URL}/{release}/theme={theme}/type={p_type}/*"
    
    logger.info(f"Connecting to DuckDB and querying Overture S3: {url}")
    
    # spatial filtering clause
    where_clause = ""
    if bbox:
        where_clause = f"""
            WHERE bbox.xmin >= {bbox['min_lon']}
              AND bbox.xmax <= {bbox['max_lon']}
              AND bbox.ymin >= {bbox['min_lat']}
              AND bbox.ymax <= {bbox['max_lat']}
        """

    query = f"""
        SELECT 
            id,
            names,
            categories,
            addresses,
            websites,
            phones,
            bbox,
            sources,
            socials
        FROM read_parquet('{url}', hive_partitioning=1)
        {where_clause}
        LIMIT {limit}
    """
    
    # DuckDB's spatial extension is required for geospatial types
    # httpfs is required for S3/HTTPS access
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    df = con.query(query).to_df()
    logger.info(f"Fetched {len(df)} rows from Overture.")
    return df

import numpy as np

def to_list(val):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val

def is_null(val):
    """Safely check if a value is null/NA, handling lists/arrays without ambiguity."""
    if val is None:
        return True
    if isinstance(val, (list, dict, tuple, np.ndarray)):
        return False
    try:
        res = pd.isna(val)
        if hasattr(res, 'any'):
            return res.any()
        return bool(res)
    except:
        return False

def _parse_overture_address(addr_list) -> dict:
    """Extract OSM-style address tags from Overture's nested structure."""
    addr_list = to_list(addr_list)
    if is_null(addr_list) or not isinstance(addr_list, (list, tuple)) or len(addr_list) == 0:
        return {}
    
    # Overture stores addresses as a list of structs. We take the first one.
    addr = addr_list[0]
    if not isinstance(addr, dict):
        return {}
        
    return {
        "addr:housenumber": addr.get("house_number"),
        "addr:street": addr.get("street"),
        "addr:city": addr.get("locality"),
        "addr:state": addr.get("region"),
        "addr:postcode": addr.get("postcode"),
        "addr:country": addr.get("country")
    }

def process_overture_row(row) -> dict:
    """Convert a single Overture row into our DB schema + canonical metadata."""
    
    # Extract coordinates from bbox
    bbox = row['bbox']
    if is_null(bbox):
        lon, lat = 0.0, 0.0
    else:
        try:
            # DuckDB may return bbox keys as xmin/xmax/ymin/ymax; older tests used minx/maxx/miny/maxy.
            xmin = bbox.get('xmin', bbox.get('minx', 0))
            xmax = bbox.get('xmax', bbox.get('maxx', 0))
            ymin = bbox.get('ymin', bbox.get('miny', 0))
            ymax = bbox.get('ymax', bbox.get('maxy', 0))
            lon = (xmin + xmax) / 2
            lat = (ymin + ymax) / 2
        except:
            lon, lat = 0.0, 0.0

    raw_metadata = {
        "id": str(row.get("id")) if not is_null(row.get("id")) else None,
        "names": to_list(row.get("names"))[0] if isinstance(row.get("names"), (list, tuple)) else row.get("names"),
        "categories": row.get("categories"),
        "addresses": row.get("addresses"),
        "websites": to_list(row.get("websites")) if not is_null(row.get("websites")) else [],
        "phones": to_list(row.get("phones")) if not is_null(row.get("phones")) else [],
        "bbox": row.get("bbox"),
        "sources": to_list(row.get("sources")) if not is_null(row.get("sources")) else [],
        "socials": to_list(row.get("socials")) if not is_null(row.get("socials")) else [],
    }

    canonical = build_canonical_metadata(raw_metadata, lat=float(lat), lon=float(lon))
    try:
        validate_canonical_metadata(canonical)
    except Exception as e:
        logger.warning("Canonical metadata validation failed for overture id=%s: %s", raw_metadata.get("id"), e)

    name = canonical.get("name") or "Unknown"

    categories = raw_metadata.get("categories")
    primary_cat = None
    if isinstance(categories, dict):
        primary_cat = categories.get("primary")
    if primary_cat is None and isinstance(row.get("category"), str):
        primary_cat = row.get("category")
    normalized_cat = str(primary_cat).replace("_", " ") if primary_cat else "unknown"

    return {
        "place_id": f"overture_{row['id']}",
        "name": str(name),
        "category": str(normalized_cat),
        "source": "overture",
        "lat": float(lat),
        "lon": float(lon),
        "metadata_json": {"canonical": canonical, "raw": raw_metadata},
    }

def main():
    parser = argparse.ArgumentParser(description="Ingest Overture Maps POI data")
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="Overture release version")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to fetch")
    parser.add_argument("--region", choices=["us-west", "all"], default="all", help="Region filter")
    args = parser.parse_args()

    # Regional bounding box (US West roughly)
    bbox = None
    if args.region == "us-west":
        bbox = {
            "min_lon": -125.0,
            "max_lon": -110.0,
            "min_lat": 32.0,
            "max_lat": 49.0
        }

    try:
        df = fetch_overture_data(args.release, DEFAULT_THEME, DEFAULT_TYPE, args.limit, bbox)
        
        if df.empty:
            logger.warning("No data found for the given parameters.")
            return

        logger.info("Processing data...")
        processed_rows = [process_overture_row(row) for _, row in df.iterrows()]
        
        logger.info(f"Upserting {len(processed_rows)} rows to PostGIS...")
        conn = get_conn()
        batch_upsert(conn, processed_rows)
        conn.close()
        
        logger.info("Overture ingestion complete.")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
