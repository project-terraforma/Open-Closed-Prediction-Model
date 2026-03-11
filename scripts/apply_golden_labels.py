"""
apply_golden_labels.py — Bulk-match golden dataset against Postgres places
and write ground_truth columns using set-based SQL (not per-row queries).

Usage:
    python scripts/apply_golden_labels.py [--local] [--limit N] [--dry-run]

Flags:
    --local    Use local postgres (postgresql://postgres:postgres123@localhost:5432/stillopen)
    --limit N  Only load first N golden records (testing)
    --dry-run  Match and report without writing to DB

Strategy:
  Step 1  COPY golden CSV into a temp table with a GIST-indexed geometry column
  Step 2  Strategy-1: LATERAL spatial join — pure SQL, one GIST lookup per row,
          no Python loop. ~15-30s for 47k coord rows.
  Step 3  Strategy-2: exact name + city bulk scan for unmatched rows
  Step 4  Bulk UPDATE places with ground_truth columns from match results
  Step 5  Second bulk UPDATE: override predicted_status for high-conf closed
"""

import argparse
import csv
import io
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPTS_DIR.parent
GOLDEN_DIR  = SCRIPTS_DIR / "data" / "golden"
GOLDEN_CSV  = GOLDEN_DIR / "golden_dataset.csv"
RESULTS_CSV = GOLDEN_DIR / "apply_labels_results.csv"

LOCAL_DB_URL = "postgresql://postgres:postgres123@localhost:5432/stillopen"


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_db_url(use_local: bool) -> str:
    if use_local:
        return LOCAL_DB_URL
    env_path = PROJECT_DIR / "stillopen" / "backend" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("Could not find DATABASE_URL. Use --local.")


def get_connection(db_url: str):
    try:
        import psycopg2
        return psycopg2.connect(db_url)
    except ImportError:
        print("[ERROR] psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)


# ── Setup SQL ──────────────────────────────────────────────────────────────────

# These must run with autocommit=True — CREATE EXTENSION and CREATE INDEX
# acquire heavy locks and cannot be run inside a transaction block.
SETUP_AUTOCOMMIT_STMTS = [
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE EXTENSION IF NOT EXISTS postgis",
    "CREATE INDEX IF NOT EXISTS places_geom_idx  ON places USING GIST(geom)",
    "CREATE INDEX IF NOT EXISTS places_trgm_idx  ON places USING GIN(name gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS places_name_lower_idx ON places ((lower(name)))",
]

# Column additions can run in a normal transaction
ENSURE_COLUMNS_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='places' AND column_name='ground_truth_status'
    ) THEN
        ALTER TABLE places ADD COLUMN ground_truth_status     VARCHAR(10);
        ALTER TABLE places ADD COLUMN ground_truth_source     VARCHAR(30);
        ALTER TABLE places ADD COLUMN ground_truth_confidence VARCHAR(10);
        ALTER TABLE places ADD COLUMN ground_truth_updated_at TIMESTAMP WITH TIME ZONE;
        RAISE NOTICE 'Added ground_truth columns.';
    END IF;
END$$;
"""

CREATE_TEMP_TABLE_SQL = """
DROP TABLE IF EXISTS golden_temp;
CREATE TEMP TABLE golden_temp (
    row_id       SERIAL PRIMARY KEY,
    g_name       TEXT,
    g_city       TEXT,
    g_address    TEXT,
    g_lat        DOUBLE PRECISION,
    g_lng        DOUBLE PRECISION,
    g_truth      TEXT,
    g_confidence TEXT,
    g_source     TEXT,
    g_geom       geometry(Point, 4326)
);
"""

# Populate geometry column and build indexes.
# GIST on g_geom enables the LATERAL spatial join to use Index Scan per row.
# ANALYZE gives planner accurate row estimates so it picks Nested Loop + GIST.
POPULATE_GEOM_SQL = """
UPDATE golden_temp
SET g_geom = ST_SetSRID(ST_MakePoint(g_lng, g_lat), 4326)
WHERE g_lat IS NOT NULL AND g_lng IS NOT NULL;
"""

CREATE_TEMP_INDEX_SQL = """
CREATE INDEX golden_temp_geom_idx  ON golden_temp USING GIST(g_geom);
CREATE INDEX golden_temp_name_idx  ON golden_temp (lower(g_name));
CREATE INDEX golden_temp_city_idx  ON golden_temp (lower(g_city));
"""
ANALYZE_TEMP_SQL = "ANALYZE golden_temp;"

# ── Bulk match queries ─────────────────────────────────────────────────────────
#
# Two strategies only — Strategy 3 (fuzzy cross-join) is dropped because
# similarity() between two non-literal columns cannot use the GIN index and
# degrades to O(98k × 1.4M) = catastrophically slow.
#
# Anti-joins use LEFT JOIN ... WHERE IS NULL instead of NOT IN (subquery)
# to avoid re-scanning match_results on every row evaluation.

CREATE_MATCH_TABLE_SQL = """
DROP TABLE IF EXISTS match_results;
CREATE TEMP TABLE match_results (
    row_id    INT PRIMARY KEY,
    place_id  TEXT,
    p_name    TEXT,
    p_address TEXT,
    p_predicted_status      TEXT,
    p_prediction_confidence DOUBLE PRECISION,
    strategy  INT,
    name_sim  DOUBLE PRECISION
);
"""

# Strategy 1: LATERAL spatial join — runs entirely inside Postgres.
# golden_temp has a GIST-indexed g_geom column, so Postgres plans this as a
# Nested Loop: for each golden row it does one GIST Index Scan on places.geom
# (~0.5ms each) rather than a full seq scan.  No Python loop needed.
# SET enable_hashjoin/mergejoin = OFF forces Nested Loop so GIST fires.
MATCH_SPATIAL_SETUP_SQL = [
    "SET LOCAL enable_hashjoin  = OFF",
    "SET LOCAL enable_mergejoin = OFF",
]
MATCH_SPATIAL_BULK_SQL = """
INSERT INTO match_results
    (row_id, place_id, p_name, p_address,
     p_predicted_status, p_prediction_confidence, strategy, name_sim)
SELECT
    g.row_id,
    m.place_id,
    m.p_name,
    m.p_address,
    m.p_predicted_status,
    m.p_prediction_confidence,
    1,
    m.name_sim
FROM golden_temp g,
LATERAL (
    SELECT
        p.id::text            AS place_id,
        p.name                AS p_name,
        p.address             AS p_address,
        p.predicted_status    AS p_predicted_status,
        p.prediction_confidence AS p_prediction_confidence,
        similarity(lower(p.name), lower(g.g_name)) AS name_sim
    FROM places p
    WHERE ST_DWithin(p.geom, g.g_geom, 0.001)
      AND similarity(lower(p.name), lower(g.g_name)) > 0.70
    ORDER BY name_sim DESC
    LIMIT 1
) m
WHERE g.g_geom IS NOT NULL
ON CONFLICT (row_id) DO NOTHING
"""

# Strategy 2: one bulk query — exact lower(name) + city LIKE.
# Single full scan of places (~3-5 sec), handles all unmatched golden rows.
MATCH_EXACT_SQL = """
INSERT INTO match_results
    (row_id, place_id, p_name, p_address,
     p_predicted_status, p_prediction_confidence, strategy, name_sim)
SELECT DISTINCT ON (g.row_id)
    g.row_id,
    p.id::text,
    p.name,
    p.address,
    p.predicted_status,
    p.prediction_confidence,
    2,
    1.0
FROM golden_temp g
LEFT JOIN match_results mr ON mr.row_id = g.row_id
JOIN places p ON
    lower(p.name) = lower(g.g_name)
    AND lower(p.address) LIKE ('%' || lower(g.g_city) || '%')
WHERE mr.row_id IS NULL
  AND g.g_city <> ''
ORDER BY g.row_id
ON CONFLICT (row_id) DO NOTHING
"""

# ── Bulk UPDATE SQL ────────────────────────────────────────────────────────────

# Write ground_truth for ALL matched records
UPDATE_GROUND_TRUTH_SQL = """
UPDATE places p
SET
    ground_truth_status     = g.g_truth,
    ground_truth_source     = g.g_source,
    ground_truth_confidence = g.g_confidence,
    ground_truth_updated_at = NOW()
FROM match_results mr
JOIN golden_temp g ON g.row_id = mr.row_id
WHERE p.id::text = mr.place_id;
"""

# Override predicted_status for high-confidence CLOSED ground truth
UPDATE_OVERRIDE_PREDICTED_SQL = """
UPDATE places p
SET
    predicted_status        = 'closed',
    prediction_confidence   = 0.95,
    prediction_updated_at   = NOW()
FROM match_results mr
JOIN golden_temp g ON g.row_id = mr.row_id
WHERE p.id::text = mr.place_id
  AND g.g_truth      = 'closed'
  AND g.g_confidence = 'high';
"""

# ── Fetch results for reporting ────────────────────────────────────────────────

FETCH_RESULTS_SQL = """
SELECT
    g.row_id,
    g.g_source,
    g.g_name,
    g.g_address,
    g.g_city,
    g.g_truth,
    g.g_confidence,
    mr.place_id,
    mr.p_name,
    mr.p_address,
    mr.p_predicted_status,
    mr.p_prediction_confidence,
    mr.strategy
FROM golden_temp g
LEFT JOIN match_results mr ON mr.row_id = g.row_id
ORDER BY g.row_id;
"""


# ── Main ───────────────────────────────────────────────────────────────────────

RESULTS_FIELDS = [
    "golden_source", "golden_name", "golden_address", "golden_city",
    "golden_ground_truth", "golden_confidence",
    "matched", "match_strategy",
    "db_id", "db_name", "db_address",
    "db_predicted_status", "db_prediction_confidence",
    "action_taken",
]


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Bulk-apply golden labels to DB")
    parser.add_argument("--local",   action="store_true")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Match and report without writing to DB")
    args = parser.parse_args()

    db_url = get_db_url(args.local)
    print(f"\n[{_ts()}] Connecting to DB...")
    conn   = get_connection(db_url)
    cursor = conn.cursor()

    # ── Step 0: Extensions + indexes (autocommit) + columns ───────────────────
    # CREATE EXTENSION and CREATE INDEX must run outside a transaction block —
    # they acquire heavy locks that deadlock if inside BEGIN/COMMIT.
    print(f"[{_ts()}] Ensuring extensions and indexes (autocommit mode)...")
    conn.autocommit = True
    for stmt in SETUP_AUTOCOMMIT_STMTS:
        try:
            cursor.execute(stmt)
            label = stmt.split()[2] if len(stmt.split()) > 2 else stmt[:40]
            print(f"  OK: {label}")
        except Exception as e:
            print(f"  [WARN] {stmt[:50]}: {e}")

    # Now switch to transactional mode for everything else
    conn.autocommit = False

    try:
        cursor.execute(ENSURE_COLUMNS_SQL)
        conn.commit()
        print(f"  Columns verified.")
    except Exception as e:
        conn.rollback()
        print(f"  [WARN] Column check failed: {e}")

    # ── Step 1: Load golden CSV into TEMP table ────────────────────────────────
    if not GOLDEN_CSV.exists():
        print(f"[ERROR] Golden dataset not found: {GOLDEN_CSV}")
        print("  Run: python scripts/build_golden_dataset.py")
        sys.exit(1)

    print(f"[{_ts()}] Loading golden CSV into temp table...")

    # Read golden records, parse coords
    golden_rows: list[dict] = []
    with open(GOLDEN_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            golden_rows.append(r)
            if args.limit and len(golden_rows) >= args.limit:
                break

    total_golden = len(golden_rows)
    print(f"  Loaded {total_golden:,} golden records from CSV")
    if args.dry_run:
        print("  DRY RUN — no DB writes will be made")

    # Create temp table
    cursor.execute(CREATE_TEMP_TABLE_SQL)
    conn.commit()

    # COPY golden rows into temp table via StringIO buffer
    buf = io.StringIO()
    for r in golden_rows:
        lat_str = r.get("latitude", "").strip()
        lng_str = r.get("longitude", "").strip()
        try:
            lat = float(lat_str) if lat_str else None
            lng = float(lng_str) if lng_str else None
        except ValueError:
            lat = lng = None

        def _esc(s: str) -> str:
            """Escape for COPY text format."""
            return s.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ").replace("\r", " ")

        buf.write("\t".join([
            _esc(r.get("name", "")),
            _esc(r.get("city", "")),
            _esc(r.get("address", "")),
            str(lat) if lat is not None else r"\N",
            str(lng) if lng is not None else r"\N",
            _esc(r.get("ground_truth", "")),
            _esc(r.get("confidence", "")),
            _esc(r.get("source", "")),
        ]) + "\n")
    buf.seek(0)

    cursor.copy_from(
        buf,
        "golden_temp",
        columns=("g_name", "g_city", "g_address", "g_lat", "g_lng",
                 "g_truth", "g_confidence", "g_source"),
        null=r"\N",
    )
    conn.commit()
    print(f"  COPY done. Building temp indexes...")

    print(f"  Populating geometry column...")
    cursor.execute(POPULATE_GEOM_SQL)
    conn.commit()

    cursor.execute(CREATE_TEMP_INDEX_SQL)
    conn.commit()

    cursor.execute(ANALYZE_TEMP_SQL)
    conn.commit()
    print(f"  Temp table ready.")

    # ── Step 2: Create match_results table ────────────────────────────────────
    cursor.execute(CREATE_MATCH_TABLE_SQL)
    conn.commit()

    # ── Step 3a: Strategy 1 — LATERAL spatial join (pure SQL, no Python loop) ─
    # golden_temp.g_geom has a GIST index; Postgres plans Nested Loop + GIST
    # index scan per row. SET LOCAL disables Hash/Merge join for this tx.
    print(f"\n[{_ts()}] Strategy 1: LATERAL spatial bulk join...")
    for stmt in MATCH_SPATIAL_SETUP_SQL:
        cursor.execute(stmt)
    cursor.execute(MATCH_SPATIAL_BULK_SQL)
    s1_count = cursor.rowcount
    conn.commit()
    print(f"  Matched: {s1_count:,} records  [{_ts()}]")

    # ── Step 3b: Strategy 2 — bulk exact name + city (one seq scan) ───────────
    print(f"[{_ts()}] Strategy 2: exact name + city bulk query...")
    cursor.execute(MATCH_EXACT_SQL)
    s2_count = cursor.rowcount
    conn.commit()
    print(f"  Matched: {s2_count:,} additional records  [{_ts()}]")

    total_matched = s1_count + s2_count
    print(f"\n  Total matched: {total_matched:,} / {total_golden:,} ({total_matched/total_golden*100:.1f}%)")

    # ── Step 4: Bulk UPDATE ────────────────────────────────────────────────────
    if not args.dry_run:
        print(f"\n[{_ts()}] Writing ground_truth to places table...")
        cursor.execute(UPDATE_GROUND_TRUTH_SQL)
        updated_gt = cursor.rowcount
        conn.commit()
        print(f"  ground_truth updated on {updated_gt:,} rows")

        print(f"[{_ts()}] Overriding predicted_status for high-conf closed matches...")
        cursor.execute(UPDATE_OVERRIDE_PREDICTED_SQL)
        updated_pred = cursor.rowcount
        conn.commit()
        print(f"  predicted_status overridden on {updated_pred:,} rows")
    else:
        updated_gt = updated_pred = 0

    # ── Step 5: Fetch results for report + CSV ────────────────────────────────
    print(f"\n[{_ts()}] Fetching match results for report...")
    cursor.execute(FETCH_RESULTS_SQL)
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]

    conn.close()

    # Build result dicts
    results: list[dict] = []
    gt_open = gt_closed = 0
    by_source: defaultdict[str, int] = defaultdict(int)
    by_strategy: defaultdict[int, int] = defaultdict(int)

    for row in rows:
        r = dict(zip(cols, row))
        matched = r["place_id"] is not None
        gt = r["g_truth"] or ""
        src = r["g_source"] or ""
        strat = r["strategy"]

        if matched:
            by_source[src] += 1
            if strat:
                by_strategy[strat] += 1
            if gt == "open":   gt_open  += 1
            elif gt == "closed": gt_closed += 1

        action = "no_match"
        if matched:
            if gt == "closed" and (r["g_confidence"] or "") == "high":
                action = "gt+predicted_overridden" if not args.dry_run else "would_override"
            else:
                action = "gt_written" if not args.dry_run else "would_write"

        results.append({
            "golden_source":             src,
            "golden_name":               r["g_name"] or "",
            "golden_address":            r["g_address"] or "",
            "golden_city":               r["g_city"] or "",
            "golden_ground_truth":       gt,
            "golden_confidence":         r["g_confidence"] or "",
            "matched":                   "yes" if matched else "no",
            "match_strategy":            str(strat) if strat else "",
            "db_id":                     str(r["place_id"]) if matched else "",
            "db_name":                   str(r["p_name"] or "") if matched else "",
            "db_address":                str(r["p_address"] or "") if matched else "",
            "db_predicted_status":       str(r["p_predicted_status"] or "") if matched else "",
            "db_prediction_confidence":  str(r["p_prediction_confidence"] or "") if matched else "",
            "action_taken":              action,
        })

    # Save CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # ── Report ─────────────────────────────────────────────────────────────────
    no_match = total_golden - total_matched
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("APPLY GOLDEN LABELS REPORT")
    print(SEP)
    print(f"  Total golden records:           {total_golden:>8,}")
    print(f"  Matched to DB:                  {total_matched:>8,}  ({total_matched/total_golden*100:.1f}%)")
    print(f"    Strategy 1 (spatial+name):    {s1_count:>8,}")
    print(f"    Strategy 2 (exact name+city): {s2_count:>8,}")
    print(f"  Not matched:                    {no_match:>8,}  ({no_match/total_golden*100:.1f}%)")
    print()
    print(f"  Ground truth for matched records:")
    print(f"    Confirmed open:               {gt_open:>8,}")
    print(f"    Confirmed closed:             {gt_closed:>8,}")
    if not args.dry_run:
        print(f"    DB rows updated (ground_truth):{updated_gt:>7,}")
        print(f"    DB rows overridden (predicted): {updated_pred:>7,}")
    print()
    print(f"  Matches by source:")
    for src, cnt in sorted(by_source.items()):
        print(f"    {src:<25} {cnt:>7,}")

    _print_confusion(results)

    print(SEP)
    print(f"\n  Full log saved: {RESULTS_CSV}")
    if not args.dry_run:
        print("\nNext step:")
        print("  python scripts/apply_predictions.py --local --batch-size 50000")
    else:
        print("\nRe-run without --dry-run to write to DB.")


def _print_confusion(results: list[dict]) -> None:
    """Confusion matrix of golden ground truth vs pre-override predictions."""
    tp = tn = fp = fn = 0
    for r in results:
        if r["matched"] != "yes":
            continue
        gt   = r["golden_ground_truth"]
        pred = (r["db_predicted_status"] or "open").lower()
        if   gt == "open"   and pred == "open":    tp += 1
        elif gt == "closed" and pred == "closed":  tn += 1
        elif gt == "open"   and pred == "closed":  fp += 1
        elif gt == "closed" and pred == "open":    fn += 1

    matched = tp + tn + fp + fn
    if matched == 0:
        return

    print()
    print("  Predictions vs ground truth (state BEFORE override):")
    print(f"  {'':26} {'Pred Open':>12} {'Pred Closed':>12}")
    print(f"  {'Actually Open  (TP/FP)':26} {tp:>12,} {fp:>12,}")
    print(f"  {'Actually Closed(FN/TN)':26} {fn:>12,} {tn:>12,}")
    print()

    accuracy  = (tp + tn) / matched * 100
    precision = tn / (tn + fp) * 100 if (tn + fp) else 0
    recall    = tn / (tn + fn) * 100 if (tn + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print(f"  Accuracy:            {accuracy:.1f}%")
    print(f"  Precision (closed):  {precision:.1f}%")
    print(f"  Recall (closed):     {recall:.1f}%")
    print(f"  F1 (closed):         {f1:.1f}%")

    fn_records = [
        r for r in results
        if r["matched"] == "yes"
        and r["golden_ground_truth"] == "closed"
        and (r["db_predicted_status"] or "open").lower() == "open"
    ]
    if fn_records:
        print()
        print(f"  Sample confirmed-closed predicted as open "
              f"({min(20, len(fn_records))} of {len(fn_records):,}):")
        for r in fn_records[:20]:
            name = r["golden_name"][:38]
            city = r["golden_city"][:14]
            src  = r["golden_source"]
            conf = r["db_prediction_confidence"] or "?"
            print(f"    {name:<38} {city:<14} [{src}] conf={conf}")


if __name__ == "__main__":
    main()
