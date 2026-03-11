"""
validate_scorer.py — Match golden dataset against Postgres places table
and produce a scorer validation report.

Usage:
    python scripts/validate_scorer.py [--local] [--limit 5000] [--sample]

Flags:
    --local   Use local postgres (postgresql://postgres:postgres123@localhost:5432/stillopen)
              Default reads DATABASE_URL from .env

    --limit N  Only process first N golden records (for quick testing)

    --sample   Print sample false negatives and false positives to console

Input:  scripts/data/golden/golden_dataset.csv
Output: scripts/data/golden/validation_results.csv
        prints SCORER VALIDATION REPORT to stdout
"""

import argparse
import csv
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPTS_DIR.parent
GOLDEN_DIR  = SCRIPTS_DIR / "data" / "golden"
GOLDEN_CSV  = GOLDEN_DIR / "golden_dataset.csv"
RESULTS_CSV = GOLDEN_DIR / "validation_results.csv"

LOCAL_DB_URL = "postgresql://postgres:postgres123@localhost:5432/stillopen"


# ── DB connection ──────────────────────────────────────────────────────────────

def get_db_url(use_local: bool) -> str:
    if use_local:
        return LOCAL_DB_URL
    # Try to read from backend .env
    env_path = PROJECT_DIR / "stillopen" / "backend" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "Could not find DATABASE_URL. Use --local or set DATABASE_URL in stillopen/backend/.env"
    )


def get_connection(db_url: str):
    try:
        import psycopg2
        return psycopg2.connect(db_url)
    except ImportError:
        print("[ERROR] psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)


# ── Golden data loader ─────────────────────────────────────────────────────────

def load_golden(limit: int | None) -> list[dict]:
    if not GOLDEN_CSV.exists():
        print(f"[ERROR] Golden dataset not found: {GOLDEN_CSV}")
        print("  Run: python scripts/build_golden_dataset.py")
        sys.exit(1)

    rows: list[dict] = []
    with open(GOLDEN_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if limit and len(rows) >= limit:
                break

    print(f"  Loaded {len(rows):,} golden records from {GOLDEN_CSV.name}")
    return rows


# ── Matcher ────────────────────────────────────────────────────────────────────

MATCH_QUERY = """
WITH candidates AS (
    -- Strategy 1: coordinate-based spatial match (within 100 m) + name similarity
    SELECT
        id,
        name,
        address,
        predicted_status,
        prediction_confidence,
        category,
        websites,
        phones,
        confidence AS overture_confidence,
        sources,
        1 AS match_strategy,
        similarity(lower(name), lower(%(name)s)) AS name_sim
    FROM places
    WHERE
        %(has_coords)s = TRUE
        AND ST_DWithin(
            geom::geography,
            ST_SetSRID(ST_MakePoint(%(lng)s, %(lat)s), 4326)::geography,
            100
        )
        AND similarity(lower(name), lower(%(name)s)) > 0.5
    ORDER BY name_sim DESC
    LIMIT 3

    UNION ALL

    -- Strategy 2: fuzzy name match + city in address (no coordinates needed)
    SELECT
        id,
        name,
        address,
        predicted_status,
        prediction_confidence,
        category,
        websites,
        phones,
        confidence AS overture_confidence,
        sources,
        2 AS match_strategy,
        similarity(lower(name), lower(%(name)s)) AS name_sim
    FROM places
    WHERE
        similarity(lower(name), lower(%(name)s)) > 0.7
        AND lower(address) LIKE lower(%(city_pat)s)
    ORDER BY name_sim DESC
    LIMIT 3
)
SELECT *
FROM candidates
ORDER BY match_strategy ASC, name_sim DESC
LIMIT 1;
"""

# Simpler fallback when pg_trgm extension might not be available
MATCH_QUERY_EXACT = """
SELECT
    id, name, address, predicted_status, prediction_confidence,
    category, websites, phones, confidence AS overture_confidence, sources,
    1 AS match_strategy, 1.0 AS name_sim
FROM places
WHERE lower(name) = lower(%(name)s)
  AND lower(address) LIKE lower(%(city_pat)s)
LIMIT 1;
"""


def match_record(cursor, row: dict) -> dict | None:
    """Try to match a golden record to a DB row. Returns DB row dict or None."""
    name     = row["name"]
    city     = row.get("city", "")
    lat_str  = row.get("latitude", "")
    lng_str  = row.get("longitude", "")

    has_coords = bool(lat_str and lng_str)
    try:
        lat = float(lat_str) if has_coords else 0.0
        lng = float(lng_str) if has_coords else 0.0
    except ValueError:
        has_coords = False
        lat = lng = 0.0

    city_pat = f"%{city}%" if city else "%"

    params = {
        "name":       name,
        "lat":        lat,
        "lng":        lng,
        "has_coords": has_coords,
        "city_pat":   city_pat,
    }

    try:
        cursor.execute(MATCH_QUERY, params)
        result = cursor.fetchone()
    except Exception:
        # Fall back to exact match if pg_trgm not available
        try:
            cursor.connection.rollback()
            cursor.execute(MATCH_QUERY_EXACT, params)
            result = cursor.fetchone()
        except Exception as exc:
            print(f"  [WARN] Match query failed for '{name}': {exc}")
            cursor.connection.rollback()
            return None

    if result is None:
        return None

    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, result))


# ── Scorer signal extractor ────────────────────────────────────────────────────

def extract_fired_signals(db_row: dict) -> list[str]:
    """
    Re-run scorer on the DB row to get which signals fired.
    This requires the scorer to be importable.
    """
    try:
        sys.path.insert(0, str(PROJECT_DIR / "stillopen" / "backend"))
        from app.scorer import scorer as _scorer

        # Build a place dict from DB columns
        place = {
            "name":       db_row.get("name", ""),
            "category":   db_row.get("category", ""),
            "address":    db_row.get("address", ""),
            "confidence": db_row.get("overture_confidence"),
            "sources":    db_row.get("sources") or [],
            "websites":   db_row.get("websites") or [],
            "phones":     db_row.get("phones") or [],
        }
        result = _scorer.score(place)
        return [s.name for s in result.fired_signals]
    except Exception:
        return []


# ── Analysis helpers ───────────────────────────────────────────────────────────

def analyze_misses(false_negatives: list[dict]) -> dict[str, int]:
    """Count which signals were NOT fired on false negatives (missed closures)."""
    all_closed_signals = {
        "explicitly_marked_closed",
        "dead_website",
        "closure_in_name",
        "osm_disused_tag",
        "no_contact_info",
        "very_low_data_confidence",
        "single_unverified_source",
        "high_turnover_category",
        "missing_address",
        "placeholder_name",
    }
    missing_counts: dict[str, int] = defaultdict(int)
    for r in false_negatives:
        fired = set(r.get("fired_signals", "").split("|") if r.get("fired_signals") else [])
        for sig in all_closed_signals:
            if sig not in fired:
                missing_counts[sig] += 1
    return dict(sorted(missing_counts.items(), key=lambda x: -x[1]))


def analyze_false_positives(false_positives: list[dict]) -> dict[str, int]:
    """Count which closed signals fired on false positives (over-predicted closed)."""
    all_closed_signals = {
        "explicitly_marked_closed", "dead_website", "closure_in_name",
        "osm_disused_tag", "no_contact_info", "very_low_data_confidence",
        "single_unverified_source", "high_turnover_category",
        "missing_address", "placeholder_name",
    }
    misfiring: dict[str, int] = defaultdict(int)
    for r in false_positives:
        fired = set(r.get("fired_signals", "").split("|") if r.get("fired_signals") else [])
        for sig in all_closed_signals:
            if sig in fired:
                misfiring[sig] += 1
    return dict(sorted(misfiring.items(), key=lambda x: -x[1]))


# ── Validation report ──────────────────────────────────────────────────────────

RESULTS_FIELDS = [
    "golden_source", "golden_license_id", "golden_name", "golden_address",
    "golden_city", "golden_ground_truth", "golden_confidence",
    "matched", "match_strategy",
    "db_id", "db_name", "db_address", "db_predicted_status",
    "db_prediction_confidence", "db_overture_confidence",
    "correct", "error_type",   # TP/TN/FP/FN or "no_match"
    "fired_signals",
]


def main():
    parser = argparse.ArgumentParser(description="Validate scorer against golden dataset")
    parser.add_argument("--local",  action="store_true", help="Use local postgres")
    parser.add_argument("--limit",  type=int, default=None, help="Process only first N records")
    parser.add_argument("--sample", action="store_true", help="Print sample FP/FN to console")
    args = parser.parse_args()

    db_url = get_db_url(args.local)
    print(f"\nConnecting to DB...")
    conn   = get_connection(db_url)
    cursor = conn.cursor()

    # Ensure pg_trgm is available (non-fatal if not)
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        conn.commit()
    except Exception:
        conn.rollback()
        print("  [INFO] pg_trgm not available — falling back to exact name matching")

    golden = load_golden(args.limit)

    results: list[dict] = []
    tp = tn = fp = fn = no_match = 0
    total_with_coords = sum(
        1 for r in golden if r.get("latitude") and r.get("longitude")
    )

    print(f"\n  Matching {len(golden):,} golden records against DB...")
    print(f"  (Records with coordinates: {total_with_coords:,})\n")

    for i, gr in enumerate(golden, 1):
        if i % 500 == 0:
            print(f"  Progress: {i:,}/{len(golden):,}  matched={tp+tn+fp+fn}")

        db_row = match_record(cursor, gr)

        result_row = {
            "golden_source":      gr.get("source", ""),
            "golden_license_id":  gr.get("license_id", ""),
            "golden_name":        gr.get("name", ""),
            "golden_address":     gr.get("address", ""),
            "golden_city":        gr.get("city", ""),
            "golden_ground_truth": gr.get("ground_truth", ""),
            "golden_confidence":  gr.get("confidence", ""),
            "matched":            "no",
            "match_strategy":     "",
            "db_id":              "",
            "db_name":            "",
            "db_address":         "",
            "db_predicted_status": "",
            "db_prediction_confidence": "",
            "db_overture_confidence":   "",
            "correct":            "",
            "error_type":         "no_match",
            "fired_signals":      "",
        }

        if db_row is None:
            no_match += 1
            results.append(result_row)
            continue

        # Extract signals for analysis
        fired = extract_fired_signals(db_row)

        result_row.update({
            "matched":                   "yes",
            "match_strategy":            str(db_row.get("match_strategy", "")),
            "db_id":                     str(db_row.get("id", "")),
            "db_name":                   str(db_row.get("name", "")),
            "db_address":                str(db_row.get("address", "")),
            "db_predicted_status":       str(db_row.get("predicted_status", "")),
            "db_prediction_confidence":  str(db_row.get("prediction_confidence", "")),
            "db_overture_confidence":    str(db_row.get("overture_confidence", "")),
            "fired_signals":             "|".join(fired),
        })

        gt          = gr.get("ground_truth", "")
        pred_status = str(db_row.get("predicted_status") or "open").lower()

        if gt == "open"   and pred_status == "open":
            tp += 1
            result_row["correct"]    = "yes"
            result_row["error_type"] = "TP"
        elif gt == "closed" and pred_status == "closed":
            tn += 1
            result_row["correct"]    = "yes"
            result_row["error_type"] = "TN"
        elif gt == "open"   and pred_status == "closed":
            fp += 1
            result_row["correct"]    = "no"
            result_row["error_type"] = "FP"
        elif gt == "closed" and pred_status == "open":
            fn += 1
            result_row["correct"]    = "no"
            result_row["error_type"] = "FN"

        results.append(result_row)

    conn.close()

    # ── Save results CSV ────────────────────────────────────────────────────────
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # ── Compute metrics ─────────────────────────────────────────────────────────
    matched = tp + tn + fp + fn
    total   = len(golden)
    correct = tp + tn

    accuracy  = correct / matched * 100  if matched else 0
    precision = tn / (tn + fp) * 100     if (tn + fp) else 0   # of those called closed
    recall    = tn / (tn + fn) * 100     if (tn + fn) else 0   # of actual closed ones

    false_negatives = [r for r in results if r["error_type"] == "FN"]
    false_positives = [r for r in results if r["error_type"] == "FP"]

    missing_signal_counts = analyze_misses(false_negatives)
    misfiring_counts      = analyze_false_positives(false_positives)

    # ── Print report ────────────────────────────────────────────────────────────
    SEP = "=" * 64
    print(f"\n{SEP}")
    print("SCORER VALIDATION REPORT")
    print(SEP)
    print(f"  Total golden records:          {total:>8,}")
    print(f"  Matched to DB:                 {matched:>8,}  ({matched/total*100:.1f}%)")
    print(f"  Unmatched:                     {no_match:>8,}  ({no_match/total*100:.1f}%)")
    print()
    print(f"  For matched records:")
    print(f"    Correctly predicted open:    {tp:>8,}  ({tp/matched*100:.1f}%)  [TP]")
    print(f"    Correctly predicted closed:  {tn:>8,}  ({tn/matched*100:.1f}%)  [TN]")
    print(f"    False positives (said closed, actually open):  {fp:>6,}")
    print(f"    False negatives (said open, actually closed):  {fn:>6,}")
    print()
    print(f"  Overall accuracy:              {accuracy:>7.1f}%")
    print(f"  Precision (closed):            {precision:>7.1f}%")
    print(f"  Recall (closed):               {recall:>7.1f}%")

    print()
    print("  Top reasons for missed closures (signals NOT fired on FN):")
    for sig, cnt in list(missing_signal_counts.items())[:8]:
        print(f"    - {sig:<35} {cnt:>5} cases")

    print()
    print("  Top misfiring signals on false positives:")
    for sig, cnt in list(misfiring_counts.items())[:8]:
        print(f"    - {sig:<35} {cnt:>5} cases")

    if args.sample and false_negatives:
        print()
        print("  Sample false negatives (said OPEN, actually CLOSED):")
        for r in false_negatives[:10]:
            sigs = r.get("fired_signals", "") or "(none)"
            print(f"    {r['golden_name'][:35]:<35}  {r['golden_address'][:30]}")
            print(f"      DB predicted: {r['db_predicted_status']}  |  Signals: {sigs[:80]}")

    if args.sample and false_positives:
        print()
        print("  Sample false positives (said CLOSED, actually OPEN):")
        for r in false_positives[:10]:
            sigs = r.get("fired_signals", "") or "(none)"
            print(f"    {r['golden_name'][:35]:<35}  {r['golden_address'][:30]}")
            print(f"      DB predicted: {r['db_predicted_status']}  |  Signals: {sigs[:80]}")

    print()
    print("  TUNING RECOMMENDATIONS")
    print("  " + "-" * 60)
    _print_recommendations(
        false_negatives, false_positives,
        missing_signal_counts, misfiring_counts,
        fn, fp, tn, matched
    )

    print(SEP)
    print(f"\n  Full results saved → {RESULTS_CSV}\n")


def _print_recommendations(
    false_negatives: list[dict],
    false_positives: list[dict],
    missing_signals: dict[str, int],
    misfiring: dict[str, int],
    fn_count: int,
    fp_count: int,
    tn_count: int,
    matched: int,
) -> None:
    """Print human-readable weight tuning recommendations."""
    recs: list[str] = []

    if fn_count > fp_count * 2:
        recs.append(
            "  Recall is low — scorer is under-predicting closures. "
            "Consider INCREASING weights for signals that are currently NOT firing "
            "on missed closures (see 'Top reasons' above)."
        )
    if fp_count > fn_count * 2:
        recs.append(
            "  Precision is low — scorer is over-predicting closures. "
            "Consider DECREASING weights for misfiring closed signals "
            "(see 'Top misfiring' above)."
        )

    # Signal-specific advice
    if missing_signals.get("no_contact_info", 0) > fn_count * 0.3:
        recs.append(
            "  'no_contact_info' is missing on many false negatives. "
            "Consider raising its weight (currently 0.18 → try 0.25–0.30)."
        )
    if missing_signals.get("high_turnover_category", 0) > fn_count * 0.3:
        recs.append(
            "  'high_turnover_category' not firing on many missed closures. "
            "Check if the business categories in your DB match HIGH_TURNOVER_CATEGORIES set."
        )
    if misfiring.get("no_contact_info", 0) > fp_count * 0.4:
        recs.append(
            "  'no_contact_info' is causing many false positives — many open businesses "
            "in SF/LA have no website/phone in Overture. Consider lowering weight "
            "(currently 0.18 → try 0.10–0.12)."
        )
    if misfiring.get("single_unverified_source", 0) > fp_count * 0.4:
        recs.append(
            "  'single_unverified_source' over-firing. Many new businesses start with "
            "one source. Consider raising the confidence threshold or lowering weight "
            "(currently 0.10 → try 0.06)."
        )
    if misfiring.get("high_turnover_category", 0) > fp_count * 0.3:
        recs.append(
            "  'high_turnover_category' causing false positives. Category coverage may "
            "be too broad. Review HIGH_TURNOVER_CATEGORIES set for over-broad terms."
        )

    if not recs:
        recs.append("  No strong tuning signals. Scorer appears well-balanced for this dataset.")

    for rec in recs:
        for line in textwrap.wrap(rec, width=70, subsequent_indent="    "):
            print(f"  {line}")
        print()


if __name__ == "__main__":
    main()
