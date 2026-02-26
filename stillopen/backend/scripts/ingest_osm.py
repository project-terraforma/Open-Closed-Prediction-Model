"""
ingest_osm.py — Transfer records from osm2pgsql tables → places table.

Reads from:  planet_osm_point  (amenities, shops, etc.)
             planet_osm_polygon (ditto, centroid as point)
Writes to:   places (via batch upsert)

Usage:
    python scripts/ingest_osm.py [--limit N] [--source-table {point,polygon,both}]

Dependencies:
    psycopg2, python-dotenv (installed via requirements.txt)
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ingest_utils import (
    get_conn,
    normalize_category,
    batch_upsert,
    logger as base_logger,
)
from utils.canonical_metadata import build_canonical_metadata, validate_canonical_metadata

logger = logging.getLogger("ingest.osm")

# ---------------------------------------------------------------------------
# Which OSM tags we pull out of hstore `tags` column
# ---------------------------------------------------------------------------
WANTED_TAGS = [
    "amenity", "shop", "office", "tourism", "healthcare", "leisure",
    "addr:housenumber", "addr:street", "addr:city", "addr:state",
    "addr:postcode", "addr:country",
    "phone", "contact:phone", "telephone",
    "website", "contact:website", "contact:email",
    "opening_hours",
    "brand", "operator",
    "cuisine", "diet:vegan", "diet:vegetarian",
    "wheelchair", "outdoor_seating",
    "description",
]

_TAG_SELECT = ", ".join(
    f"tags->'{t}' AS \"{t}\"" for t in WANTED_TAGS
)

# Minimum set of tags a POI must have to be worth importing
_REQUIRED_TAGS = {"amenity", "shop", "office", "tourism", "healthcare", "leisure"}


def _build_query(table: str, limit_clause: str, extra_cols: str = "") -> str:
    """
    Build a parameterized SELECT for planet_osm_point or planet_osm_polygon.
    For polygons we use ST_Centroid to get a representative point.
    """
    if "polygon" in table:
        geom_expr = "ST_X(ST_Centroid(way::geometry)) AS lon, ST_Y(ST_Centroid(way::geometry)) AS lat"
    else:
        geom_expr = "ST_X(way::geometry) AS lon, ST_Y(way::geometry) AS lat"

    return f"""
    SELECT
        osm_id,
        name,
        {geom_expr},
        {_TAG_SELECT}
        {extra_cols}
    FROM {table}
    WHERE
        name IS NOT NULL
        AND name != ''
        AND (
            {" OR ".join(f"tags ? '{t}'" for t in _REQUIRED_TAGS)}
        )
    {limit_clause};
    """


def _row_to_place(row, source: str = "osm") -> dict | None:
    """Convert a DB row to a places-table-compatible dict. Returns None to skip."""
    lon, lat = row["lon"], row["lat"]
    if lon is None or lat is None:
        return None

    # Build tags dict from row (only non-None values)
    tags: dict = {"name": row.get("name")}
    for t in WANTED_TAGS:
        val = row.get(t)
        if val:
            tags[t] = val

    category = normalize_category(tags)
    raw_metadata: dict = dict(tags)
    canonical = build_canonical_metadata(raw_metadata, lat=lat, lon=lon)
    try:
        validate_canonical_metadata(canonical)
    except Exception as e:
        logger.warning("Canonical metadata validation failed for osm_id=%s: %s", row.get("osm_id"), e)

    return {
        "place_id": f"osm_{source}_{row['osm_id']}",
        "name": row["name"],
        "category": category,
        "source": "osm",
        "lat": lat,
        "lon": lon,
        "metadata_json": {"canonical": canonical, "raw": raw_metadata},
    }


def ingest_table(conn, table: str, limit: int | None = None):
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = _build_query(table, limit_clause)
    table_short = "point" if "point" in table else "polygon"

    logger.info("Querying %s …", table)
    with conn.cursor(cursor_factory=__import__("psycopg2").extras.RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()

    logger.info("  Fetched %d candidate rows from %s", len(rows), table)

    places = []
    skipped = 0
    for row in rows:
        p = _row_to_place(dict(row), source=table_short)
        if p:
            places.append(p)
        else:
            skipped += 1

    logger.info("  Converted %d rows (%d skipped — no geom)", len(places), skipped)
    total = batch_upsert(conn, places)
    logger.info("  ✓ Upserted %d records from %s", total, table)
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest OSM data → places table")
    parser.add_argument("--limit", type=int, default=None, help="Max rows per table (for testing)")
    parser.add_argument(
        "--source-table",
        choices=["point", "polygon", "both"],
        default="both",
        help="Which osm2pgsql table(s) to read",
    )
    args = parser.parse_args()

    conn = get_conn()
    total = 0

    # Check planet_osm_point / polygon exist
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('planet_osm_point', 'planet_osm_polygon');
        """)
        available = {r[0] for r in cur.fetchall()}

    if not available:
        logger.warning(
            "No planet_osm_* tables found. Run osm2pgsql first — see MIGRATION.md."
        )
        conn.close()
        sys.exit(0)

    if args.source_table in ("point", "both") and "planet_osm_point" in available:
        total += ingest_table(conn, "planet_osm_point", args.limit)

    if args.source_table in ("polygon", "both") and "planet_osm_polygon" in available:
        total += ingest_table(conn, "planet_osm_polygon", args.limit)

    conn.close()
    logger.info("OSM ingestion complete. Total records upserted: %d", total)


if __name__ == "__main__":
    main()
