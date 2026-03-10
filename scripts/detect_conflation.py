"""
detect_conflation.py — Find and resolve duplicate places across OSM and Overture data.

The Problem
-----------
Overture Maps Foundation (OMF) ingests OpenStreetMap as a primary source.
A place can end up in our DB twice:
  - Overture record  (source='overture', place_id like 'overture_abc...')
  - OSM record       (source='osm',      place_id like 'osm_1234567890')
  - Wikidata record  (source='wikidata', place_id like 'wikidata_Q123')

Detection Methods (in order of confidence)
-------------------------------------------
1. OSM ID cross-reference (EXACT, confidence=1.0)
   Overture stores source attribution in metadata_json.sources[].record_id
   e.g. {"dataset": "openstreetmap", "record_id": "node/1234567890"}
   Our OSM records have place_id = "osm_1234567890"
   → Direct match is possible with a JSONB query.

2. Spatial proximity + name similarity (FUZZY, confidence=variable)
   Places within 25m of each other from different sources, with name
   similarity > 0.6 (pg_trgm trigram similarity), are likely the same.
   → Catches re-mapped places, ID changes, and Wikidata/OSM overlap.

3. Address + name fingerprint (HEURISTIC, confidence=medium)
   Normalized name + normalized address hash. Catches cases where
   coordinates differ but address is identical.

Resolution Strategies
---------------------
--action=report    (default) Print stats only, no changes
--action=mark      Add a 'duplicate_of' key to the secondary record's metadata
--action=delete    Delete secondary records (OSM/Wikidata) where Overture exists
--action=merge     Merge: copy OSM closure signals into Overture record, then delete OSM copy

Usage:
    python scripts/detect_conflation.py --local --action=report
    python scripts/detect_conflation.py --local --action=mark
    python scripts/detect_conflation.py --local --action=merge
    python scripts/detect_conflation.py --local --method=spatial --threshold=25
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)


# ── Text normalization ─────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove punctuation/whitespace runs."""
    if not name:
        return ""
    n = unicodedata.normalize("NFD", name)
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")  # strip accents
    n = n.lower()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    # Common suffix normalizations
    for suffix in (" inc", " llc", " ltd", " corp", " co"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def _normalize_address(addr: str) -> str:
    """Lowercase, expand common abbreviations, strip unit numbers."""
    if not addr:
        return ""
    a = addr.lower()
    replacements = [
        (r"\bst\b", "street"), (r"\bave\b", "avenue"), (r"\bblvd\b", "boulevard"),
        (r"\bdr\b", "drive"),  (r"\brd\b",  "road"),   (r"\bln\b",  "lane"),
        (r"\bct\b", "court"),  (r"\bpl\b",  "place"),  (r"\bpkwy\b","parkway"),
        (r"\bsuite\s+\w+", ""), (r"\bunit\s+\w+", ""), (r"\bapt\s+\w+", ""),
        (r"\s+", " "),
    ]
    for pattern, repl in replacements:
        a = re.sub(pattern, repl, a)
    return a.strip()


# ── Detection methods ──────────────────────────────────────────────────────────

def _detect_by_osm_id(conn) -> list[dict]:
    """
    Method 1: Exact OSM ID cross-reference.

    Overture records store OSM attribution in:
      metadata_json->'sources' = [{"dataset": "openstreetmap", "record_id": "node/123"}]

    We extract those OSM node/way IDs and compare against our osm_* place_ids.
    Returns list of {overture_id, osm_id, name, confidence} dicts.
    """
    print("\n[Method 1] OSM ID cross-reference...")

    query = """
    WITH overture_osm_ids AS (
        -- Extract OSM record IDs from Overture source attribution
        SELECT
            p.place_id AS overture_place_id,
            p.name     AS overture_name,
            src->>'record_id' AS osm_ref
        FROM places p,
             jsonb_array_elements(
                 CASE
                     WHEN jsonb_typeof(p.metadata_json->'sources') = 'array'
                     THEN p.metadata_json->'sources'
                     ELSE '[]'::jsonb
                 END
             ) AS src
        WHERE p.source = 'overture'
          AND src->>'dataset' ILIKE '%openstreetmap%'
          AND src->>'record_id' IS NOT NULL
    ),
    parsed AS (
        -- Strip 'node/' or 'way/' prefix to get bare OSM ID
        SELECT
            overture_place_id,
            overture_name,
            regexp_replace(osm_ref, '^(node|way|relation)/', '') AS osm_num_id
        FROM overture_osm_ids
        WHERE osm_ref ~ '^(node|way|relation)/[0-9]+'
    )
    SELECT
        p.overture_place_id,
        p.overture_name,
        osm.place_id    AS osm_place_id,
        osm.name        AS osm_name,
        osm.category    AS category,
        osm.address     AS address,
        osm.metadata_json AS osm_meta
    FROM parsed p
    JOIN places osm ON osm.place_id = 'osm_' || p.osm_num_id
    ORDER BY p.overture_name;
    """

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]

    results = [dict(zip(cols, row)) for row in rows]
    print(f"  Found {len(results):,} exact OSM-ID matches")
    return results


def _detect_by_spatial_proximity(conn, threshold_meters: float = 25) -> list[dict]:
    """
    Method 2: Spatial proximity + name similarity.

    Finds pairs of places from different sources within `threshold_meters`
    that have trigram name similarity > 0.5.
    Requires pg_trgm extension.
    """
    print(f"\n[Method 2] Spatial proximity ({threshold_meters}m) + name similarity...")

    # Ensure pg_trgm is available
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
        if not cur.fetchone():
            print("  pg_trgm not installed — skipping spatial method")
            print("  Install with: CREATE EXTENSION pg_trgm;")
            return []

    query = f"""
    SELECT
        a.place_id  AS primary_id,
        a.name      AS primary_name,
        a.source    AS primary_source,
        b.place_id  AS duplicate_id,
        b.name      AS duplicate_name,
        b.source    AS duplicate_source,
        b.category  AS category,
        b.address   AS address,
        ROUND(ST_Distance(a.geom::geography, b.geom::geography)::numeric, 1) AS dist_m,
        ROUND(similarity(a.name, b.name)::numeric, 3) AS name_sim,
        b.metadata_json AS dup_meta
    FROM places a
    JOIN places b
      ON a.source IN ('overture')
     AND b.source IN ('osm', 'wikidata')
     AND a.place_id <> b.place_id
     AND ST_DWithin(a.geom::geography, b.geom::geography, {threshold_meters})
     AND similarity(a.name, b.name) > 0.5
    WHERE a.geom IS NOT NULL
      AND b.geom IS NOT NULL
    ORDER BY name_sim DESC, dist_m ASC
    LIMIT 50000;
    """

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]

    results = [dict(zip(cols, row)) for row in rows]

    # Deduplicate: keep only the best match per duplicate_id
    seen: dict[str, dict] = {}
    for r in results:
        dup_id = r["duplicate_id"]
        if dup_id not in seen or r["name_sim"] > seen[dup_id]["name_sim"]:
            seen[dup_id] = r
    deduped = list(seen.values())

    print(f"  Found {len(deduped):,} spatial near-duplicates (from {len(results):,} raw pairs)")
    return deduped


def _detect_by_name_address(conn) -> list[dict]:
    """
    Method 3: Normalized name + address fingerprint.

    Pulls all OSM/Wikidata records with an address, computes a fingerprint,
    and matches against Overture records with the same fingerprint.
    Done in Python (not SQL) to use _normalize_name/_normalize_address.
    """
    print("\n[Method 3] Name + address fingerprint matching...")

    with conn.cursor() as cur:
        cur.execute("""
            SELECT place_id, name, address, source, category, metadata_json
            FROM places
            WHERE address IS NOT NULL AND address != ''
              AND source IN ('overture', 'osm', 'wikidata')
        """)
        rows = cur.fetchall()

    overture_fp: dict[str, dict] = {}
    secondary: list[dict] = []

    for place_id, name, address, source, category, meta in rows:
        fp = _normalize_name(name) + "|" + _normalize_address(address)
        if len(fp) < 6:
            continue
        rec = {
            "place_id": place_id, "name": name, "address": address,
            "source": source, "category": category, "meta": meta, "fp": fp,
        }
        if source == "overture":
            overture_fp[fp] = rec
        else:
            secondary.append(rec)

    matches = []
    for rec in secondary:
        if rec["fp"] in overture_fp:
            overture_rec = overture_fp[rec["fp"]]
            matches.append({
                "primary_id":        overture_rec["place_id"],
                "primary_name":      overture_rec["name"],
                "primary_source":    "overture",
                "duplicate_id":      rec["place_id"],
                "duplicate_name":    rec["name"],
                "duplicate_source":  rec["source"],
                "category":          rec["category"],
                "address":           rec["address"],
                "dup_meta":          rec["meta"],
            })

    print(f"  Found {len(matches):,} name+address fingerprint matches")
    return matches


# ── Resolution actions ─────────────────────────────────────────────────────────

def _action_mark(conn, all_duplicates: list[dict]) -> int:
    """
    Add 'duplicate_of' key to each secondary record's metadata_json.
    Safe — does not delete anything. Allows the scorer to deprioritize them.
    """
    print("\n[Action: mark] Adding duplicate_of field to secondary records...")
    seen: set[str] = set()
    updates = []
    for dup in all_duplicates:
        dup_id = dup.get("duplicate_id") or dup.get("osm_place_id")
        primary_id = dup.get("primary_id") or dup.get("overture_place_id")
        if dup_id and dup_id not in seen:
            seen.add(dup_id)
            updates.append((primary_id, dup_id))

    with conn.cursor() as cur:
        for primary_id, dup_id in updates:
            cur.execute(
                """
                UPDATE places
                SET metadata_json = metadata_json || jsonb_build_object(
                    'duplicate_of', %s::text,
                    'conflation_method', 'detected'
                )
                WHERE place_id = %s
                """,
                (primary_id, dup_id),
            )
    conn.commit()
    print(f"  Marked {len(updates):,} secondary records")
    return len(updates)


def _action_merge(conn, all_duplicates: list[dict]) -> int:
    """
    For each duplicate pair:
      1. Copy OSM closure signals into the Overture record (disused:*, end_date,
         operating_status, website_status) if not already present.
      2. Delete the secondary (OSM/Wikidata) record.

    This preserves Overture's rich metadata while keeping OSM's closure signals.
    """
    print("\n[Action: merge] Merging closure signals → Overture, deleting secondaries...")

    CLOSURE_FIELDS = {
        "disused:amenity", "disused:shop", "closed:amenity", "closed:shop",
        "end_date", "operating_status", "website_status",
    }

    seen: set[str] = set()
    merged = 0
    deleted = 0

    with conn.cursor() as cur:
        for dup in all_duplicates:
            dup_id     = dup.get("duplicate_id") or dup.get("osm_place_id")
            primary_id = dup.get("primary_id")   or dup.get("overture_place_id")
            dup_meta   = dup.get("dup_meta")      or dup.get("osm_meta") or {}

            if not dup_id or dup_id in seen:
                continue
            seen.add(dup_id)

            # Extract closure signals from secondary record
            if isinstance(dup_meta, str):
                try:
                    dup_meta = json.loads(dup_meta)
                except Exception:
                    dup_meta = {}

            closure_patch = {
                k: v for k, v in dup_meta.items()
                if k in CLOSURE_FIELDS and v
            }

            # Merge into primary record (only adds fields, won't overwrite existing)
            if closure_patch and primary_id:
                patch_json = json.dumps(closure_patch)
                cur.execute(
                    """
                    UPDATE places
                    SET metadata_json = %s::jsonb || metadata_json
                    WHERE place_id = %s
                    """,
                    (patch_json, primary_id),
                )
                merged += 1

            # Delete the secondary record
            cur.execute("DELETE FROM places WHERE place_id = %s", (dup_id,))
            deleted += 1

    conn.commit()
    print(f"  Merged closure signals into {merged:,} Overture records")
    print(f"  Deleted {deleted:,} secondary records")
    return deleted


def _action_delete(conn, all_duplicates: list[dict]) -> int:
    """
    Simply delete all secondary (OSM/Wikidata) records that duplicate an Overture record.
    """
    print("\n[Action: delete] Deleting secondary duplicate records...")
    seen: set[str] = set()
    with conn.cursor() as cur:
        for dup in all_duplicates:
            dup_id = dup.get("duplicate_id") or dup.get("osm_place_id")
            if dup_id and dup_id not in seen:
                seen.add(dup_id)
                cur.execute("DELETE FROM places WHERE place_id = %s", (dup_id,))
    conn.commit()
    print(f"  Deleted {len(seen):,} duplicate records")
    return len(seen)


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_report(osm_id_matches, spatial_matches, fp_matches):
    sep = "=" * 65

    # Combine and deduplicate across methods
    all_dup_ids: set[str] = set()
    for m in osm_id_matches:
        all_dup_ids.add(m.get("osm_place_id", ""))
    for m in spatial_matches:
        all_dup_ids.add(m.get("duplicate_id", ""))
    for m in fp_matches:
        all_dup_ids.add(m.get("duplicate_id", ""))
    all_dup_ids.discard("")

    print(f"\n{sep}")
    print("CONFLATION DETECTION REPORT")
    print(sep)
    print(f"  Method 1 — Exact OSM ID match     : {len(osm_id_matches):>8,}")
    print(f"  Method 2 — Spatial + name sim      : {len(spatial_matches):>8,}")
    print(f"  Method 3 — Name + address fingerpt : {len(fp_matches):>8,}")
    print(f"  {'─'*40}")
    print(f"  Total unique duplicate place_ids   : {len(all_dup_ids):>8,}")

    # Source breakdown of duplicates
    source_counts: dict[str, int] = defaultdict(int)
    for m in spatial_matches + fp_matches:
        source_counts[m.get("duplicate_source", "unknown")] += 1
    for m in osm_id_matches:
        source_counts["osm"] += 1

    print(f"\n  Duplicates by source:")
    for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {src:<20s}  {count:>8,}")

    # Sample spatial matches with similarity scores
    if spatial_matches:
        print(f"\n  Sample spatial matches (top 15 by name similarity):")
        print(f"  {'Primary (Overture)':30s}  {'Secondary':30s}  {'Dist':>6}  {'Sim':>5}")
        print(f"  {'-'*30}  {'-'*30}  {'-'*6}  {'-'*5}")
        for m in sorted(spatial_matches, key=lambda x: x.get("name_sim", 0), reverse=True)[:15]:
            pn = (m.get("primary_name") or "")[:30]
            dn = (m.get("duplicate_name") or "")[:30]
            dist = m.get("dist_m", "?")
            sim  = m.get("name_sim", "?")
            print(f"  {pn:<30s}  {dn:<30s}  {dist:>5}m  {sim:>5}")

    # Sample OSM ID matches
    if osm_id_matches:
        print(f"\n  Sample exact OSM ID matches (first 10):")
        for m in osm_id_matches[:10]:
            ov = (m.get("overture_name") or "")[:45]
            os_ = (m.get("osm_name") or "")[:45]
            print(f"    Overture: {ov}")
            print(f"    OSM:      {os_}")
            print()

    print(f"\n{sep}")
    print("RECOMMENDATIONS")
    print(sep)
    print("""
  --action=merge   (RECOMMENDED)
    Copies OSM closure signals (disused:*, end_date, operating_status)
    into the Overture record, then deletes the OSM duplicate.
    Best of both worlds: Overture's rich metadata + OSM's closure signals.

  --action=mark
    Adds 'duplicate_of' to secondary records. Nothing deleted.
    Useful for auditing before committing to deletion.

  --action=delete
    Deletes secondary records outright.
    Use when you trust Overture's data completeness.
""")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect and resolve conflated places between OSM and Overture."
    )
    parser.add_argument("--local", action="store_true",
                        help="Use local Postgres")
    parser.add_argument("--db", default=None)
    parser.add_argument("--action", default="report",
                        choices=["report", "mark", "merge", "delete"],
                        help="What to do with detected duplicates (default: report)")
    parser.add_argument("--method", default="all",
                        choices=["all", "osm_id", "spatial", "fingerprint"],
                        help="Which detection method(s) to run (default: all)")
    parser.add_argument("--threshold", type=float, default=25.0,
                        help="Spatial proximity threshold in meters (default: 25)")
    args = parser.parse_args()

    # ── DB connection ─────────────────────────────────────────────────────────
    try:
        import psycopg2
        import psycopg2.extras
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

    # ── Source counts ─────────────────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute("SELECT source, COUNT(*) FROM places GROUP BY source ORDER BY COUNT(*) DESC")
        rows = cur.fetchall()
    print(f"\nCurrent DB source breakdown:")
    for source, count in rows:
        print(f"  {(source or 'unknown'):<20s}  {count:>10,}")

    # ── Run detection methods ─────────────────────────────────────────────────
    osm_id_matches: list[dict] = []
    spatial_matches: list[dict] = []
    fp_matches: list[dict] = []

    if args.method in ("all", "osm_id"):
        osm_id_matches = _detect_by_osm_id(conn)

    if args.method in ("all", "spatial"):
        spatial_matches = _detect_by_spatial_proximity(conn, args.threshold)

    if args.method in ("all", "fingerprint"):
        fp_matches = _detect_by_name_address(conn)

    # ── Build unified duplicate list ──────────────────────────────────────────
    # Normalise all matches to the same shape: {primary_id, duplicate_id, ...}
    all_duplicates: list[dict] = []

    for m in osm_id_matches:
        all_duplicates.append({
            "primary_id":       m["overture_place_id"],
            "primary_name":     m["overture_name"],
            "primary_source":   "overture",
            "duplicate_id":     m["osm_place_id"],
            "duplicate_name":   m["osm_name"],
            "duplicate_source": "osm",
            "category":         m["category"],
            "address":          m["address"],
            "dup_meta":         m["osm_meta"],
            "method":           "osm_id",
        })

    # Avoid double-counting with method-1 results
    already_marked = {d["duplicate_id"] for d in all_duplicates}

    for m in spatial_matches + fp_matches:
        dup_id = m.get("duplicate_id", "")
        if dup_id not in already_marked:
            all_duplicates.append({**m, "method": "spatial_or_fp"})
            already_marked.add(dup_id)

    # ── Print report ──────────────────────────────────────────────────────────
    _print_report(osm_id_matches, spatial_matches, fp_matches)

    print(f"  Total unique duplicates found across all methods: {len(all_duplicates):,}")

    if args.action == "report":
        print("\n  Run with --action=merge to resolve (recommended).")
        conn.close()
        return

    # ── Apply resolution ──────────────────────────────────────────────────────
    if args.action == "mark":
        _action_mark(conn, all_duplicates)
    elif args.action == "merge":
        _action_merge(conn, all_duplicates)
    elif args.action == "delete":
        _action_delete(conn, all_duplicates)

    # Final count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM places")
        total = cur.fetchone()[0]
    print(f"\nFinal place count: {total:,}")
    conn.close()


if __name__ == "__main__":
    main()
