"""
fetch_wikidata_ca.py — Fetch California businesses from Wikidata SPARQL endpoint.

Wikidata is a free, open knowledge base with structured data on companies,
restaurants, hotels, museums, hospitals, and other businesses — including
coordinates, addresses, websites, and official names.

This is especially good for:
  - National/regional chains with official Wikidata entries
  - Museums, universities, hospitals, landmarks
  - Well-known independent businesses
  - Places that have closed (end_time field in Wikidata)

No API key required. Uses the public SPARQL endpoint.

Usage:
    python scripts/fetch_wikidata_ca.py --local
    python scripts/fetch_wikidata_ca.py --local --dry-run
    python scripts/fetch_wikidata_ca.py --db postgresql://user:pass@host/db

Output:
    Upserts to places table with source='wikidata'
    After completion prints counts and sample records
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from collections import defaultdict
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
BATCH_SIZE = 500
SLEEP_BETWEEN_QUERIES = 2  # Wikidata asks for polite delays

# ── SPARQL queries ─────────────────────────────────────────────────────────────
# We split into multiple targeted queries to avoid the 60s timeout.
# Each query targets a specific business type in California (Q99).

QUERIES = {
    "restaurants_cafes": """
        SELECT DISTINCT ?place ?placeLabel ?coord ?website ?phone ?instanceLabel ?endTime WHERE {
          ?place wdt:P31/wdt:P279* wd:Q11707 .       # instance of: restaurant (or subclass)
          ?place wdt:P131* wd:Q99 .                    # located in California
          ?place wdt:P625 ?coord .                     # has coordinates
          OPTIONAL { ?place wdt:P856 ?website }
          OPTIONAL { ?place wdt:P1329 ?phone }
          OPTIONAL { ?place wdt:P31 ?instance }
          OPTIONAL { ?place wdt:P576 ?endTime }        # dissolution date (= closed)
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 3000
    """,
    "hotels_lodging": """
        SELECT DISTINCT ?place ?placeLabel ?coord ?website ?phone ?endTime WHERE {
          ?place wdt:P31/wdt:P279* wd:Q27686 .        # hotel or subclass
          ?place wdt:P131* wd:Q99 .
          ?place wdt:P625 ?coord .
          OPTIONAL { ?place wdt:P856 ?website }
          OPTIONAL { ?place wdt:P1329 ?phone }
          OPTIONAL { ?place wdt:P576 ?endTime }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 2000
    """,
    "museums_galleries": """
        SELECT DISTINCT ?place ?placeLabel ?coord ?website ?phone ?endTime WHERE {
          VALUES ?type { wd:Q33506 wd:Q207694 wd:Q1030034 wd:Q856584 }
          ?place wdt:P31/wdt:P279* ?type .             # museum / art gallery / science museum
          ?place wdt:P131* wd:Q99 .
          ?place wdt:P625 ?coord .
          OPTIONAL { ?place wdt:P856 ?website }
          OPTIONAL { ?place wdt:P1329 ?phone }
          OPTIONAL { ?place wdt:P576 ?endTime }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 2000
    """,
    "hospitals_clinics": """
        SELECT DISTINCT ?place ?placeLabel ?coord ?website ?phone ?endTime WHERE {
          VALUES ?type { wd:Q16917 wd:Q180320 wd:Q1774898 }
          ?place wdt:P31/wdt:P279* ?type .             # hospital / medical center
          ?place wdt:P131* wd:Q99 .
          ?place wdt:P625 ?coord .
          OPTIONAL { ?place wdt:P856 ?website }
          OPTIONAL { ?place wdt:P1329 ?phone }
          OPTIONAL { ?place wdt:P576 ?endTime }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 2000
    """,
    "universities_colleges": """
        SELECT DISTINCT ?place ?placeLabel ?coord ?website ?phone WHERE {
          VALUES ?type { wd:Q3918 wd:Q189004 wd:Q38723 wd:Q15936437 }
          ?place wdt:P31/wdt:P279* ?type .             # university / community college
          ?place wdt:P131* wd:Q99 .
          ?place wdt:P625 ?coord .
          OPTIONAL { ?place wdt:P856 ?website }
          OPTIONAL { ?place wdt:P1329 ?phone }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 1000
    """,
    "retail_chains": """
        SELECT DISTINCT ?place ?placeLabel ?coord ?website ?endTime WHERE {
          VALUES ?type { wd:Q878070 wd:Q1311670 wd:Q1047641 wd:Q2567502 }
          ?place wdt:P31/wdt:P279* ?type .             # retail store / supermarket / chain
          ?place wdt:P131* wd:Q99 .
          ?place wdt:P625 ?coord .
          OPTIONAL { ?place wdt:P856 ?website }
          OPTIONAL { ?place wdt:P576 ?endTime }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 3000
    """,
    "banks_financial": """
        SELECT DISTINCT ?place ?placeLabel ?coord ?website ?endTime WHERE {
          VALUES ?type { wd:Q22687 wd:Q786820 wd:Q2085381 }
          ?place wdt:P31/wdt:P279* ?type .             # bank / credit union
          ?place wdt:P131* wd:Q99 .
          ?place wdt:P625 ?coord .
          OPTIONAL { ?place wdt:P856 ?website }
          OPTIONAL { ?place wdt:P576 ?endTime }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 2000
    """,
    "companies_offices": """
        SELECT DISTINCT ?place ?placeLabel ?coord ?website ?endTime WHERE {
          ?place wdt:P31/wdt:P279* wd:Q4830453 .      # business / company
          ?place wdt:P131* wd:Q99 .
          ?place wdt:P625 ?coord .
          OPTIONAL { ?place wdt:P856 ?website }
          OPTIONAL { ?place wdt:P576 ?endTime }
          FILTER NOT EXISTS { ?place wdt:P31/wdt:P279* wd:Q11707 }  # not already a restaurant
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 3000
    """,
}

# Wikidata → our category mapping
CATEGORY_MAP = {
    "restaurants_cafes":      "restaurant",
    "hotels_lodging":         "hotel",
    "museums_galleries":      "museum",
    "hospitals_clinics":      "hospital",
    "universities_colleges":  "university",
    "retail_chains":          "retail",
    "banks_financial":        "bank",
    "companies_offices":      "office",
}


# ── SPARQL fetch ───────────────────────────────────────────────────────────────

def _run_sparql(query: str, description: str) -> list[dict]:
    """Run a SPARQL query and return list of result bindings."""
    params = urllib.parse.urlencode({
        "query": query,
        "format": "json",
    })
    url = f"{SPARQL_ENDPOINT}?{params}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "StillOpenProject/2.0 (research@example.com)")
    req.add_header("Accept", "application/sparql-results+json")

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", {}).get("bindings", [])
            print(f"  {description}: {len(results):,} results")
            return results
    except Exception as e:
        print(f"  {description}: FAILED — {e}")
        return []


def _parse_coord(coord_str: str) -> tuple[float, float] | None:
    """Parse 'Point(lon lat)' WKT from Wikidata into (lat, lon)."""
    if not coord_str:
        return None
    try:
        # Wikidata returns "Point(lon lat)"
        inner = coord_str.replace("Point(", "").replace(")", "").strip()
        parts = inner.split()
        lon, lat = float(parts[0]), float(parts[1])
        # Sanity-check California bounds
        if not (32.0 <= lat <= 42.5 and -125.0 <= lon <= -113.0):
            return None
        return lat, lon
    except Exception:
        return None


def _parse_wikidata_results(bindings: list[dict], category: str) -> list[dict]:
    """Convert SPARQL result bindings into place dicts."""
    seen_ids: set[str] = set()
    records = []

    for b in bindings:
        # Wikidata entity URI → place_id
        place_uri = b.get("place", {}).get("value", "")
        qid = place_uri.split("/")[-1]
        if not qid.startswith("Q"):
            continue
        place_id = f"wikidata_{qid}"
        if place_id in seen_ids:
            continue
        seen_ids.add(place_id)

        # Name
        name = b.get("placeLabel", {}).get("value", "").strip()
        if not name or name == qid:  # Wikidata returns QID when no label
            continue

        # Coordinates
        coord_str = b.get("coord", {}).get("value", "")
        coords = _parse_coord(coord_str)
        if not coords:
            continue
        lat, lon = coords

        # Optional fields
        website = b.get("website", {}).get("value", "") or ""
        phone   = b.get("phone",   {}).get("value", "") or ""
        end_time = b.get("endTime", {}).get("value", "")  # dissolution date

        # If Wikidata has an end_time, the entity is closed/dissolved
        is_closed = bool(end_time)
        open_flag = 0 if is_closed else 1
        operating_status = "closed" if is_closed else None

        websites = [website] if website else []
        phones   = [phone]   if phone   else []

        # Confidence: Wikidata is authoritative for what it knows
        confidence = 0.80
        if website:
            confidence = min(0.92, confidence + 0.07)
        if phone:
            confidence = min(0.92, confidence + 0.03)

        metadata: dict = {
            "websites":   websites,
            "phones":     phones,
            "socials":    [],
            "confidence": round(confidence, 2),
            "open":       open_flag,
            "sources": [{
                "dataset":    "wikidata",
                "record_id":  qid,
                "confidence": round(confidence, 2),
            }],
            "wikidata_qid": qid,
        }
        if operating_status:
            metadata["operating_status"] = operating_status
        if end_time:
            metadata["end_date"] = end_time[:10]  # store as YYYY-MM-DD

        records.append({
            "place_id": place_id,
            "name":     name,
            "category": category,
            "address":  None,   # Wikidata rarely has street-level addresses
            "source":   "wikidata",
            "lat":      lat,
            "lon":      lon,
            "metadata": metadata,
        })

    return records


# ── DB upsert ─────────────────────────────────────────────────────────────────

def _upsert_batch(cur, records: list) -> int:
    if not records:
        return 0
    rows = []
    for r in records:
        geom = f"SRID=4326;POINT({r['lon']} {r['lat']})"
        rows.append((
            r["place_id"],
            r["name"],
            r["category"],
            r["address"],
            r["source"],
            geom,
            json.dumps(r["metadata"]),
        ))

    from psycopg2.extras import execute_values
    execute_values(
        cur,
        """
        INSERT INTO places (place_id, name, category, address, source, geom, metadata_json)
        VALUES %s
        ON CONFLICT (place_id) DO UPDATE SET
            name          = EXCLUDED.name,
            metadata_json = places.metadata_json || EXCLUDED.metadata_json
        """,
        rows,
        template="(%s, %s, %s, %s, %s, ST_GeomFromEWKT(%s), %s::jsonb)",
    )
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch California businesses from Wikidata SPARQL and upsert to DB."
    )
    parser.add_argument("--local", action="store_true",
                        help="Use local Postgres")
    parser.add_argument("--db", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and parse but do not write to DB")
    parser.add_argument("--query", default=None,
                        help=f"Only run one query type: {list(QUERIES)}")
    args = parser.parse_args()

    # ── DB setup ──────────────────────────────────────────────────────────────
    conn = None
    if not args.dry_run:
        try:
            import psycopg2
        except ImportError:
            sys.exit("psycopg2 not installed.")

        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, "stillopen", "backend", ".env"))

        if args.db:
            db_url = args.db
        elif args.local:
            db_url = "postgresql://postgres:postgres123@localhost:5432/stillopen"
        else:
            db_url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/stillopen")
        db_url = db_url.replace("postgresql+psycopg2://", "postgresql://")
        print(f"DB: {db_url.split('@')[-1] if '@' in db_url else db_url}")
        conn = psycopg2.connect(db_url)
        conn.autocommit = False
    else:
        print("DRY RUN — no DB writes")

    queries_to_run = {k: v for k, v in QUERIES.items()
                      if args.query is None or args.query == k}

    print(f"\nFetching {len(queries_to_run)} query type(s) from Wikidata...\n")

    total_fetched  = 0
    total_upserted = 0
    category_totals: dict[str, int] = defaultdict(int)
    closed_count = 0

    for query_name, sparql in queries_to_run.items():
        category = CATEGORY_MAP.get(query_name, query_name)
        print(f"[{query_name}]")
        bindings = _run_sparql(sparql, "  SPARQL")

        records = _parse_wikidata_results(bindings, category)
        closed_here = sum(1 for r in records if r["metadata"]["open"] == 0)
        print(f"  Parsed: {len(records):,}  closed: {closed_here:,}")

        total_fetched += len(records)
        closed_count  += closed_here
        category_totals[category] += len(records)

        if not args.dry_run and records and conn:
            cur = conn.cursor()
            upserted = 0
            for i in range(0, len(records), BATCH_SIZE):
                chunk = records[i:i + BATCH_SIZE]
                upserted += _upsert_batch(cur, chunk)
            conn.commit()
            cur.close()
            total_upserted += upserted
            print(f"  Upserted: {upserted:,}")

        time.sleep(SLEEP_BETWEEN_QUERIES)

    if conn:
        conn.close()

    sep = "=" * 60
    print(f"\n{sep}")
    print("WIKIDATA FETCH — COMPLETE")
    print(sep)
    print(f"  Query types run    : {len(queries_to_run)}")
    print(f"  Total places parsed: {total_fetched:,}")
    print(f"  Closed (end_time)  : {closed_count:,}")
    if not args.dry_run:
        print(f"  DB upserted        : {total_upserted:,}")

    print(f"\n  Results by category:")
    for cat, count in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cat:<30s}  {count:>6,}")

    print(f"\n{sep}")
    if args.dry_run:
        print("Dry run — re-run without --dry-run to write to DB.")
    else:
        print("Done. Run apply_predictions_v2.py --local to score new records.")
    print(sep)


if __name__ == "__main__":
    main()
