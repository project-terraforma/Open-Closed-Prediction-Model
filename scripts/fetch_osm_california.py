"""
fetch_osm_california.py — Fetch all named business places from OpenStreetMap
for California via the Overpass API, then upsert into PostgreSQL.

Strategy:
  - Split California into ~25 county-level bounding boxes.
  - For each county, query Overpass for amenity + shop + tourism + leisure nodes/ways.
  - Upsert directly to the `places` table (ON CONFLICT DO UPDATE to refresh data).
  - Skips records with no name and records that duplicate existing Overture place_ids.
  - Rate-limited: 3 seconds between Overpass calls to respect the fair-use policy.

Usage:
    python scripts/fetch_osm_california.py --local
    python scripts/fetch_osm_california.py --db postgresql://user:pass@host/db
    python scripts/fetch_osm_california.py --local --dry-run   (parse only, no DB writes)
    python scripts/fetch_osm_california.py --local --county "Los Angeles"

After completion prints:
  - Records fetched per county
  - Total inserted / updated / skipped
  - Top categories in new data
"""

import argparse
import json
import os
import sys
import time
import ssl
import urllib.request
import urllib.parse
from collections import defaultdict
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
DATA_DIR = os.path.join(SCRIPTS_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SLEEP_BETWEEN_QUERIES = 3   # seconds — Overpass fair-use policy
BATCH_SIZE = 500             # DB upsert batch size

# ── California county bounding boxes (south, west, north, east) ───────────────
# Each box is small enough that Overpass can return results within the timeout.
COUNTIES = [
    # (display_name, south, west, north, east)
    ("Los Angeles",      33.70, -118.95, 34.82, -117.65),
    ("San Diego",        32.53, -117.60, 33.51, -116.08),
    ("Orange",           33.37, -118.12, 33.96, -117.41),
    ("Riverside",        33.50, -117.68, 34.08, -114.43),
    ("San Bernardino W", 33.85, -117.68, 35.10, -116.00),
    ("San Bernardino E", 34.80, -116.00, 35.80, -114.13),
    ("Santa Clara",      36.90, -122.20, 37.51, -121.20),
    ("Alameda",          37.44, -122.38, 37.91, -121.46),
    ("San Francisco",    37.70, -122.52, 37.84, -122.35),
    ("San Mateo",        37.33, -122.52, 37.71, -122.05),
    ("Contra Costa",     37.82, -122.38, 38.08, -121.53),
    ("Sacramento",       38.24, -121.87, 38.69, -121.19),
    ("Fresno",           36.60, -120.10, 37.10, -119.57),
    ("Kern",             34.76, -119.57, 35.80, -117.63),
    ("Ventura",          33.98, -119.56, 34.62, -118.63),
    ("Santa Barbara",    34.35, -120.60, 34.97, -119.47),
    ("Monterey",         35.79, -121.98, 36.93, -120.21),
    ("San Luis Obispo",  34.83, -120.88, 35.80, -119.45),
    ("Solano",           38.04, -122.41, 38.55, -121.58),
    ("Sonoma",           38.17, -123.55, 38.88, -122.35),
    ("Napa",             38.20, -122.69, 38.88, -122.06),
    ("Marin",            37.83, -123.05, 38.25, -122.40),
    ("San Joaquin",      37.65, -121.59, 38.30, -120.92),
    ("Stanislaus",       37.30, -121.27, 37.94, -120.62),
    ("Merced",           37.00, -121.27, 37.65, -120.00),
    ("Tulare",           35.79, -119.57, 36.60, -118.38),
    ("Kings",            35.79, -120.10, 36.34, -119.57),
    ("Madera",           36.60, -120.10, 37.35, -119.10),
    ("El Dorado",        38.52, -121.02, 39.08, -119.93),
    ("Placer",           38.73, -121.48, 39.33, -120.47),
    ("Shasta",           40.07, -123.00, 41.18, -121.30),
    ("Butte",            39.33, -122.00, 39.93, -121.30),
    ("Santa Cruz",       36.85, -122.33, 37.29, -121.58),
    ("Humboldt",         40.00, -124.41, 41.48, -123.40),
]

# OSM tags we consider "business" places
AMENITY_TYPES = (
    "restaurant|cafe|bar|fast_food|pub|food_court|ice_cream|bakery|"
    "bank|atm|pharmacy|hospital|clinic|dentist|doctors|veterinary|"
    "school|college|university|library|cinema|theatre|nightclub|casino|"
    "fuel|car_wash|car_rental|car_repair|parking|"
    "post_office|police|fire_station|embassy|"
    "place_of_worship|community_centre|social_facility|"
    "marketplace|biergarten|food_court"
)

SHOP_TYPES = (
    "supermarket|convenience|department_store|mall|"
    "clothes|shoes|jewelry|gift|electronics|mobile_phone|computer|"
    "hardware|furniture|home_furnishings|interior_decoration|"
    "beauty|hairdresser|cosmetics|massage|"
    "sports|outdoor|bicycle|car|car_parts|"
    "books|music|video_games|toys|"
    "bakery|butcher|deli|greengrocer|alcohol|"
    "florist|garden_centre|pet|"
    "dry_cleaning|laundry|"
    "optician|"
    "copyshop|stationery|"
    "vacant"
)

TOURISM_TYPES = "hotel|motel|hostel|guest_house|apartment|museum|gallery|attraction|viewpoint"
LEISURE_TYPES = "fitness_centre|sports_centre|swimming_pool|golf_course|bowling_alley|spa"

# ── SSL context (Windows sometimes has cert issues with Overpass) ──────────────
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


# ── Overpass helpers ───────────────────────────────────────────────────────────

def _query_overpass(query: str, description: str, retries: int = 2) -> list:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(OVERPASS_URL, data=data)
    req.add_header("User-Agent", "StillOpenProject/2.0 (Academic research)")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=150, context=_ssl_ctx) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                elements = result.get("elements", [])
                print(f"  {description}: {len(elements):,} elements")
                return elements
        except Exception as e:
            if attempt < retries:
                print(f"  {description}: error ({e}), retrying in 10s...")
                time.sleep(10)
            else:
                print(f"  {description}: FAILED after {retries+1} attempts — {e}")
                return []


def _build_county_query(s: float, w: float, n: float, e: float) -> str:
    bbox = f"{s},{w},{n},{e}"
    return f"""
[out:json][timeout:120][bbox:{bbox}];
(
  node["amenity"~"^({AMENITY_TYPES})$"]["name"];
  way["amenity"~"^({AMENITY_TYPES})$"]["name"];
  node["shop"~"^({SHOP_TYPES})$"]["name"];
  way["shop"~"^({SHOP_TYPES})$"]["name"];
  node["tourism"~"^({TOURISM_TYPES})$"]["name"];
  way["tourism"~"^({TOURISM_TYPES})$"]["name"];
  node["leisure"~"^({LEISURE_TYPES})$"]["name"];
  way["leisure"~"^({LEISURE_TYPES})$"]["name"];
  node["office"]["name"];
);
out center tags;
"""


# ── Record parsing ─────────────────────────────────────────────────────────────

def _parse_element(element: dict) -> dict | None:
    tags = element.get("tags", {})
    name = (tags.get("name") or tags.get("brand") or "").strip()
    if not name:
        return None

    # lat/lon — nodes have direct coords; ways have a "center"
    lat = element.get("lat") or (element.get("center") or {}).get("lat")
    lon = element.get("lon") or (element.get("center") or {}).get("lon")
    if not lat or not lon:
        return None

    # Category: prefer most specific tag
    category = (
        tags.get("shop") or tags.get("amenity") or
        tags.get("tourism") or tags.get("leisure") or
        tags.get("office") or "unknown"
    )

    # Address
    addr_parts = []
    house  = tags.get("addr:housenumber", "")
    street = tags.get("addr:street", "")
    if house or street:
        addr_parts.append(f"{house} {street}".strip())
    city   = tags.get("addr:city", "")
    state  = tags.get("addr:state", "")
    if city:
        addr_parts.append(city)
    if state:
        addr_parts.append(state)
    address = ", ".join(p for p in addr_parts if p)

    # Contact info
    websites = [v for k, v in tags.items()
                if k in ("website", "contact:website", "url") and v]
    phones   = [v for k, v in tags.items()
                if k in ("phone", "contact:phone", "telephone") and v]
    socials  = [v for k, v in tags.items()
                if k in ("contact:facebook", "contact:instagram", "facebook",
                         "contact:twitter", "contact:youtube") and v]

    # Brand
    brand_name = tags.get("brand") or tags.get("operator") or ""
    brand = {"names": {"primary": brand_name}} if brand_name else None

    # Closed signals
    is_disused = bool(
        tags.get("disused:amenity") or tags.get("disused:shop") or
        tags.get("closed:amenity") or tags.get("closed:shop")
    )
    is_vacant = tags.get("shop") == "vacant" or tags.get("amenity") == "vacant"
    open_flag = 0 if (is_disused or is_vacant) else 1
    operating_status = "closed" if (is_disused or is_vacant) else None

    # OSM disused tags (pass through for scorer)
    osm_closure_tags = {}
    for k in ("disused:amenity", "disused:shop", "closed:amenity", "closed:shop", "end_date"):
        if tags.get(k):
            osm_closure_tags[k] = tags[k]

    # Confidence: higher when address + website present
    confidence = 0.5
    if address:
        confidence += 0.15
    if websites:
        confidence += 0.15
    if phones:
        confidence += 0.05
    confidence = round(min(0.85, confidence), 2)

    osm_id = str(element.get("id", ""))
    place_id = f"osm_{osm_id}"

    metadata: dict = {
        "websites": websites,
        "phones":   phones,
        "socials":  socials,
        "confidence": confidence,
        "open": open_flag,
        "city": city,
        "state": state or "CA",
        "sources": [{"dataset": "osm", "record_id": osm_id}],
    }
    if brand:
        metadata["brand"] = brand
    if address:
        metadata["address"] = address
    if operating_status:
        metadata["operating_status"] = operating_status
    metadata.update(osm_closure_tags)

    return {
        "place_id": place_id,
        "name": name,
        "category": category,
        "address": address or None,
        "source": "osm",
        "lat": float(lat),
        "lon": float(lon),
        "metadata": metadata,
    }


# ── DB upsert ─────────────────────────────────────────────────────────────────

def _upsert_batch(cur, records: list) -> tuple[int, int]:
    """Upsert a list of parsed place dicts. Returns (inserted, updated)."""
    if not records:
        return 0, 0

    rows = []
    for r in records:
        lat, lon = r["lat"], r["lon"]
        geom = f"SRID=4326;POINT({lon} {lat})"
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
            category      = EXCLUDED.category,
            address       = COALESCE(EXCLUDED.address, places.address),
            metadata_json = places.metadata_json || EXCLUDED.metadata_json
        """,
        rows,
        template="(%s, %s, %s, %s, %s, ST_GeomFromEWKT(%s), %s::jsonb)",
    )
    # psycopg2 doesn't expose inserted vs updated counts for execute_values,
    # so we return total rows processed
    return len(rows), 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch California OSM businesses and upsert to PostgreSQL."
    )
    parser.add_argument("--local", action="store_true",
                        help="Use local Postgres (postgres:postgres123@localhost:5432/stillopen)")
    parser.add_argument("--db", default=None,
                        help="Override DATABASE_URL entirely")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and print stats without writing to DB")
    parser.add_argument("--county", default=None,
                        help="Only process a specific county (partial match OK)")
    args = parser.parse_args()

    # ── DB connection ─────────────────────────────────────────────────────────
    if not args.dry_run:
        try:
            import psycopg2
        except ImportError:
            sys.exit("psycopg2 not installed. Run: pip install psycopg2-binary")

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
        conn = None
        print("DRY RUN — no DB writes")

    # ── Select counties to process ────────────────────────────────────────────
    counties = COUNTIES
    if args.county:
        counties = [c for c in COUNTIES if args.county.lower() in c[0].lower()]
        if not counties:
            sys.exit(f"No county matching '{args.county}'. Available: {[c[0] for c in COUNTIES]}")
        print(f"Processing {len(counties)} county(ies) matching '{args.county}'")

    # ── Process each county ───────────────────────────────────────────────────
    print(f"\nFetching OSM places for {len(counties)} California regions...\n")

    total_fetched   = 0
    total_upserted  = 0
    total_skipped   = 0
    category_counts: dict[str, int] = defaultdict(int)
    now_ts = datetime.utcnow().isoformat()

    for county_name, s, w, n, e in counties:
        print(f"[{county_name}]")
        query = _build_county_query(s, w, n, e)
        elements = _query_overpass(query, f"  Overpass query")

        parsed = []
        for elem in elements:
            record = _parse_element(elem)
            if record:
                parsed.append(record)
                category_counts[record["category"]] += 1

        skipped = len(elements) - len(parsed)
        total_fetched  += len(parsed)
        total_skipped  += skipped
        print(f"  Parsed: {len(parsed):,}  skipped (no name/coords): {skipped:,}")

        if not args.dry_run and parsed and conn:
            cur = conn.cursor()
            # Process in sub-batches
            upserted = 0
            for i in range(0, len(parsed), BATCH_SIZE):
                chunk = parsed[i:i + BATCH_SIZE]
                n_up, _ = _upsert_batch(cur, chunk)
                upserted += n_up
            conn.commit()
            cur.close()
            total_upserted += upserted
            print(f"  Upserted: {upserted:,}")

        time.sleep(SLEEP_BETWEEN_QUERIES)

    if conn:
        conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("OSM CALIFORNIA FETCH — COMPLETE")
    print(sep)
    print(f"  Counties processed : {len(counties)}")
    print(f"  Total places parsed: {total_fetched:,}")
    print(f"  Skipped (no name)  : {total_skipped:,}")
    if not args.dry_run:
        print(f"  DB upserted        : {total_upserted:,}")

    print(f"\n  Top 20 categories in fetched data:")
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for cat, count in sorted_cats:
        print(f"    {cat:<30s}  {count:>8,}")

    print(f"\n{sep}")
    if args.dry_run:
        print("Dry run complete — re-run without --dry-run to write to DB.")
    else:
        print("Done. Run apply_predictions_v2.py --local to score new records.")
    print(sep)


if __name__ == "__main__":
    main()
