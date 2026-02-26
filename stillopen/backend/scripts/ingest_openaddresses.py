"""
ingest_openaddresses.py — Ingest OpenAddresses CSV files → staging table,
then enrich `places.metadata_json` by matching on geospatial proximity.

OpenAddresses (https://openaddresses.io) provides free city/county address
CSVs with standardized columns:
    LON, LAT, NUMBER, STREET, UNIT, CITY, DISTRICT, REGION, POSTCODE, ID, HASH

Usage:
    # Download a file: https://batch.openaddresses.io/data  (choose your region)
    python scripts/ingest_openaddresses.py path/to/addresses.csv [--radius 25]

Arguments:
    csv_path   Path to the OpenAddresses CSV file.
    --radius   Match radius in meters (default: 25). Smaller = more precise.
    --chunk    Rows to process per commit (default: 2000).
    --drop     Drop the staging table after enrichment (default: False).

How it works:
    1. Load the CSV into a temporary Postgres staging table.
    2. For each row in staging, find the nearest `places` record within --radius metres
       (using PostGIS ST_DWithin on a spatial index).
    3. Merge address fields into `metadata_json` non-destructively.
"""

import argparse
import csv
import logging
import sys
import os
import json
import psycopg2.extras

sys.path.insert(0, os.path.dirname(__file__))
from ingest_utils import get_conn, logger as _
from utils.canonical_metadata import build_canonical_metadata, validate_canonical_metadata

logger = logging.getLogger("ingest.openaddresses")

# ---------------------------------------------------------------------------
# Staging table DDL
# ---------------------------------------------------------------------------
STAGING_DDL = """
CREATE UNLOGGED TABLE IF NOT EXISTS _staging_openaddresses (
    id          BIGSERIAL PRIMARY KEY,
    lon         DOUBLE PRECISION,
    lat         DOUBLE PRECISION,
    number      TEXT,
    street      TEXT,
    unit        TEXT,
    city        TEXT,
    district    TEXT,
    region      TEXT,
    postcode    TEXT,
    source_id   TEXT,
    geom        GEOMETRY(Point, 4326)
);
CREATE INDEX IF NOT EXISTS idx_staging_oa_geom ON _staging_openaddresses USING GIST(geom);
"""

STAGING_INSERT = """
INSERT INTO _staging_openaddresses
    (lon, lat, number, street, unit, city, district, region, postcode, source_id, geom)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
"""

MATCH_SQL = """
SELECT
    p.place_id,
    p.metadata_json,
    ST_X(p.geom::geometry) AS lon,
    ST_Y(p.geom::geometry) AS lat,
    oa.number,
    oa.street,
    oa.unit,
    oa.city,
    oa.district,
    oa.region,
    oa.postcode,
    oa.source_id
FROM places p
CROSS JOIN LATERAL (
    SELECT
        number, street, unit, city, district, region, postcode, source_id, geom
    FROM _staging_openaddresses oa
    WHERE ST_DWithin(p.geom::geometry, oa.geom, %(radius)s)
    ORDER BY ST_Distance(p.geom::geometry, oa.geom)
    LIMIT 1
) oa
"""


def _load_csv_to_staging(conn, csv_path: str, chunk_size: int = 2000):
    """Stream load a large CSV into the staging table."""
    total = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        with conn.cursor() as cur:
            for row in reader:
                try:
                    lon = float(row.get("LON") or row.get("lon") or 0)
                    lat = float(row.get("LAT") or row.get("lat") or 0)
                except ValueError:
                    continue
                if lon == 0 or lat == 0:
                    continue

                batch.append((
                    lon, lat,
                    (row.get("NUMBER") or row.get("number") or "").strip() or None,
                    (row.get("STREET") or row.get("street") or "").strip() or None,
                    (row.get("UNIT") or row.get("unit") or "").strip() or None,
                    (row.get("CITY") or row.get("city") or "").strip() or None,
                    (row.get("DISTRICT") or row.get("district") or "").strip() or None,
                    (row.get("REGION") or row.get("region") or "").strip() or None,
                    (row.get("POSTCODE") or row.get("postcode") or "").strip() or None,
                    (row.get("ID") or row.get("id") or row.get("HASH") or "").strip() or None,
                    lon, lat,  # repeated for ST_MakePoint
                ))

                if len(batch) >= chunk_size:
                    cur.executemany(STAGING_INSERT, batch)
                    conn.commit()
                    total += len(batch)
                    logger.info("  Loaded %d rows into staging …", total)
                    batch = []

            if batch:
                cur.executemany(STAGING_INSERT, batch)
                conn.commit()
                total += len(batch)

    logger.info("Staging load complete: %d rows", total)
    return total


def _enrich_places(conn, radius_m: int = 25):
    """Match staging addresses to places and rebuild canonical metadata_json."""
    logger.info("Enriching places within %d m of staging addresses …", radius_m)
    updated = 0

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(MATCH_SQL, {"radius": radius_m})
        rows = cur.fetchall()

    logger.info("  Found %d candidate place matches", len(rows))

    with conn.cursor() as cur:
        for r in rows:
            place_id = r["place_id"]
            lat = r.get("lat")
            lon = r.get("lon")

            meta = r.get("metadata_json") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            if isinstance(meta, dict) and "canonical" in meta and "raw" in meta:
                raw = meta.get("raw") or {}
            else:
                raw = meta if isinstance(meta, dict) else {}

            # Skip if we already have a street (keep enrichment non-destructive)
            if raw.get("addr:street") or raw.get("address"):
                continue

            oa_raw = {
                "number": r.get("number"),
                "street": r.get("street"),
                "unit": r.get("unit"),
                "city": r.get("city"),
                "district": r.get("district"),
                "region": r.get("region"),
                "postcode": r.get("postcode"),
                "source_id": r.get("source_id"),
            }

            # Preserve the OpenAddresses row as-is
            raw = dict(raw)
            raw.setdefault("openaddresses", oa_raw)

            # Also project components into common addr:* keys (no formatted address string built here)
            if not raw.get("addr:housenumber") and oa_raw.get("number"):
                raw["addr:housenumber"] = oa_raw["number"]
            if not raw.get("addr:street") and oa_raw.get("street"):
                raw["addr:street"] = oa_raw["street"]
            if not raw.get("addr:unit") and oa_raw.get("unit"):
                raw["addr:unit"] = oa_raw["unit"]
            if not raw.get("addr:city") and oa_raw.get("city"):
                raw["addr:city"] = oa_raw["city"]
            if not raw.get("addr:district") and oa_raw.get("district"):
                raw["addr:district"] = oa_raw["district"]
            if not raw.get("addr:state") and oa_raw.get("region"):
                raw["addr:state"] = oa_raw["region"]
            if not raw.get("addr:postcode") and oa_raw.get("postcode"):
                raw["addr:postcode"] = oa_raw["postcode"]

            canonical = build_canonical_metadata(raw, lat=lat, lon=lon)
            try:
                validate_canonical_metadata(canonical)
            except Exception as e:
                logger.warning("Canonical metadata validation failed for place_id=%s: %s", place_id, e)

            new_meta = {"canonical": canonical, "raw": raw}
            cur.execute(
                "UPDATE places SET metadata_json = %s::jsonb WHERE place_id = %s",
                (json.dumps(new_meta), place_id),
            )
            updated += 1

        conn.commit()

    logger.info("  ✓ Enriched %d places records", updated)
    return updated


def main():
    parser = argparse.ArgumentParser(description="Ingest OpenAddresses CSV → enrich places")
    parser.add_argument("csv_path", help="Path to OpenAddresses CSV file")
    parser.add_argument("--radius", type=int, default=25, help="Match radius in metres (default: 25)")
    parser.add_argument("--chunk", type=int, default=2000, help="CSV rows per DB commit")
    parser.add_argument("--drop", action="store_true", help="Drop staging table after completion")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        logger.error("File not found: %s", args.csv_path)
        sys.exit(1)

    conn = get_conn()

    # Create staging table + spatial index
    logger.info("Creating staging table …")
    with conn.cursor() as cur:
        cur.execute(STAGING_DDL)
        conn.commit()

    # Load CSV → staging
    loaded = _load_csv_to_staging(conn, args.csv_path, args.chunk)

    if loaded == 0:
        logger.warning("No rows loaded — check CSV format. Expected columns: LON,LAT,NUMBER,STREET,CITY,REGION,POSTCODE")
        conn.close()
        sys.exit(0)

    # Enrich places
    _enrich_places(conn, args.radius)

    if args.drop:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS _staging_openaddresses;")
            conn.commit()
        logger.info("Staging table dropped.")

    conn.close()
    logger.info("OpenAddresses ingestion complete.")


if __name__ == "__main__":
    main()
