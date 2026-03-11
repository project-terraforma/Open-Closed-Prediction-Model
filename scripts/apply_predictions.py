"""
apply_predictions.py — Apply the signal-based scorer to all 1.4M places.

Replaces the previous XGBoost model approach.
No ML model required — pure Python signal logic via scorer.py.

Updates three columns on the places table:
  predicted_status       VARCHAR(20)  — 'open' | 'closed'
  prediction_confidence  FLOAT        — 50-99 (percentage)
  prediction_updated_at  TIMESTAMP    — when the prediction was last written

Usage:
    python scripts/apply_predictions_v2.py [--batch-size N] [--offset N] [--local] [--db-url URL]

After completion prints:
  - Total open vs closed counts and percentages
  - Average confidence for open and closed predictions
  - Which signals fired most frequently (top 10)
  - Top 10 categories with highest closed prediction rate
  - Chain verification (no known chains predicted closed)
  - 30 sample closed predictions with fired signals
  - Spot check pass/fail results
"""

import json
import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from stillopen.backend.app.scorer import PlaceScorer, KNOWN_CHAINS


def build_place_dict(row: dict) -> dict:
    """
    Combine DB row fields into a flat dict for the scorer.
    metadata_json is already a dict (psycopg2 parses JSONB automatically).
    """
    meta = row.get("metadata_json") or {}
    if not isinstance(meta, dict):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}

    return {
        "name":     row.get("name") or "",
        "category": row.get("category") or "",
        "address":  row.get("address") or "",
        **meta,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply signal-based scorer to all places in Postgres."
    )
    parser.add_argument("--batch-size", type=int, default=50_000,
                        help="Records per DB batch (default: 50000)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N records (resume from checkpoint)")
    parser.add_argument("--local", action="store_true",
                        help="Use local Postgres (postgres:postgres123@localhost:5432/stillopen)")
    parser.add_argument("--db-url", type=str, default=None,
                        help="Override DATABASE_URL entirely")
    args = parser.parse_args()

    # ── DB connection ─────────────────────────────────────────────────────────
    from dotenv import load_dotenv
    import psycopg2
    import psycopg2.extras

    load_dotenv(os.path.join(PROJECT_ROOT, "stillopen", "backend", ".env"))

    if args.db_url:
        db_url = args.db_url
    elif args.local:
        db_url = "postgresql://postgres:postgres123@localhost:5432/stillopen"
    else:
        db_url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/stillopen")
    db_url = db_url.replace("postgresql+psycopg2://", "postgresql://")

    print(f"StillOpen — Signal-Based Batch Predictor v2")
    print(f"  DB: {db_url.split('@')[-1] if '@' in db_url else db_url}")

    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    # ── Total record count ────────────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM places")
        total = cur.fetchone()[0]
    print(f"  Total records : {total:,}")
    print(f"  Batch size    : {args.batch_size:,}")
    print(f"  Starting at   : {args.offset:,}\n")

    scorer = PlaceScorer()

    batch_size = args.batch_size
    offset = args.offset
    processed = 0
    n_open = 0
    n_closed = 0
    now_ts = datetime.utcnow().isoformat()

    # Accumulators for post-run report
    open_conf_sum   = 0.0
    closed_conf_sum = 0.0
    signal_counts: dict[str, int] = defaultdict(int)
    category_totals: dict[str, int] = defaultdict(int)
    category_closed: dict[str, int] = defaultdict(int)

    print(f"Processing...")

    while offset < total:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT place_id, name, category, address, metadata_json
                FROM places
                ORDER BY id
                LIMIT %s OFFSET %s
                """,
                (batch_size, offset),
            )
            batch_rows = cur.fetchall()

        if not batch_rows:
            break

        updates = []
        for row in batch_rows:
            place = build_place_dict(row)
            result = scorer.score(place)

            cat = (row.get("category") or "unknown").lower()
            category_totals[cat] += 1

            if result.status == "closed":
                n_closed += 1
                closed_conf_sum += result.confidence
                category_closed[cat] += 1
            else:
                n_open += 1
                open_conf_sum += result.confidence

            for sig in result.fired_signals:
                signal_counts[sig.name] += 1

            updates.append((result.status, result.confidence, now_ts, row["place_id"]))

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                UPDATE places
                SET predicted_status      = %s,
                    prediction_confidence = %s,
                    prediction_updated_at = %s
                WHERE place_id = %s
                """,
                updates,
                page_size=1000,
            )
        conn.commit()

        processed += len(batch_rows)
        offset += batch_size

        if processed % 100_000 < batch_size:
            pct = processed / total * 100
            print(
                f"  {processed:>10,} / {total:,} ({pct:.1f}%)  "
                f"closed={n_closed:,}  open={n_open:,}"
            )

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print("APPLY PREDICTIONS V2 — COMPLETE")
    print(sep)
    print(f"  Total processed  : {processed:,}")
    print(f"  Predicted OPEN   : {n_open:,}  ({n_open/max(1,processed)*100:.2f}%)")
    print(f"  Predicted CLOSED : {n_closed:,}  ({n_closed/max(1,processed)*100:.2f}%)")

    avg_open_conf   = open_conf_sum   / max(1, n_open)
    avg_closed_conf = closed_conf_sum / max(1, n_closed)
    print(f"\n  Avg confidence (open)  : {avg_open_conf:.1f}%")
    print(f"  Avg confidence (closed): {avg_closed_conf:.1f}%")

    # Top 10 signals
    print(f"\n  Top 10 signals fired:")
    sorted_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, count in sorted_signals:
        pct = count / max(1, processed) * 100
        print(f"    {name:<35s} {count:>10,}  ({pct:.1f}%)")

    # Top 10 categories by closed rate
    print(f"\n  Top 10 categories by CLOSED prediction rate:")
    cat_rates = []
    for cat, closed in category_closed.items():
        total_cat = category_totals[cat]
        rate = closed / total_cat if total_cat > 0 else 0
        if total_cat >= 10:  # ignore tiny categories
            cat_rates.append((cat, closed, total_cat, rate))
    cat_rates.sort(key=lambda x: x[3], reverse=True)
    print(f"  {'Category':35s}  {'Closed':>8}  {'Total':>8}  {'%Closed':>8}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*8}")
    for cat, closed, total_cat, rate in cat_rates[:10]:
        print(f"  {cat[:35]:35s}  {closed:>8,}  {total_cat:>8,}  {rate*100:>7.1f}%")

    # ── Chain verification ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("CHAIN VERIFICATION — No known chains should be predicted closed")
    print(sep)

    conn2 = psycopg2.connect(db_url.replace("postgresql+psycopg2://", "postgresql://"))
    chain_violations = []
    with conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT name, category, predicted_status, prediction_confidence
            FROM places
            WHERE predicted_status = 'closed'
            LIMIT 50000
            """
        )
        for row in cur.fetchall():
            name_lower = (row["name"] or "").lower()
            if any(chain in name_lower for chain in KNOWN_CHAINS):
                chain_violations.append(dict(row))

    if chain_violations:
        print(f"  WARNING: {len(chain_violations)} chain(s) predicted CLOSED:")
        for v in chain_violations[:10]:
            print(f"    {v['name']} — conf={v['prediction_confidence']:.0f}%")
    else:
        print(f"  PASS — No known chains predicted closed.")

    # ── Spot check 30 high-confidence closed predictions ─────────────────────
    print(f"\n{sep}")
    print("SPOT CHECK — 30 High-Confidence CLOSED Predictions")
    print(sep)

    with conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                p.name,
                p.category,
                p.address,
                p.predicted_status,
                p.prediction_confidence,
                p.metadata_json->>'city'           AS city,
                p.metadata_json->>'website_status' AS website_status,
                (p.metadata_json->>'confidence')::float AS overture_conf,
                p.metadata_json->'websites'        AS websites_raw
            FROM places p
            WHERE predicted_status = 'closed'
              AND prediction_confidence >= 75
            ORDER BY prediction_confidence DESC
            LIMIT 30
            """
        )
        spot_rows = cur.fetchall()

    for i, row in enumerate(spot_rows, 1):
        name     = row["name"] or "Unknown"
        cat      = (row["category"] or "unknown")[:30]
        city     = row["city"] or ""
        conf     = row["prediction_confidence"] or 0
        ov_conf  = row["overture_conf"] or 0.0
        ws       = row["website_status"] or "unchecked"

        # Re-score to get signal names
        place = {
            "name": name,
            "category": row["category"] or "",
            "website_status": ws,
            "confidence": ov_conf,
        }
        result = scorer.score(place)
        signal_names = ", ".join(s.name for s in result.fired_signals)

        print(f"\n  [{i:2d}] {name[:50]}")
        print(f"       Category   : {cat}")
        print(f"       City       : {city}")
        print(f"       Confidence : {conf:.0f}%  |  Overture conf: {ov_conf:.2f}")
        print(f"       Signals    : {signal_names}")

    conn2.close()

    # ── Correctness spot checks ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("CORRECTNESS SPOT CHECKS")
    print(sep)

    test_cases = [
        # (description, place_dict, expected_status, min_confidence)
        (
            "Starbucks (national chain)",
            {"name": "Starbucks", "category": "cafe", "confidence": 0.9,
             "sources": [{"dataset": "a"}, {"dataset": "b"}, {"dataset": "c"}]},
            "open", 85,
        ),
        (
            "McDonalds (national chain)",
            {"name": "McDonalds", "category": "fast_food", "confidence": 0.88,
             "sources": [{"dataset": "a"}, {"dataset": "b"}, {"dataset": "c"}]},
            "open", 85,
        ),
        (
            "Active website + 3 sources + high confidence",
            {"name": "Active Bistro", "category": "restaurant",
             "website_status": "active", "confidence": 0.9,
             "phones": ["+14085550001"],
             "sources": [{"dataset": "a"}, {"dataset": "b"}, {"dataset": "c"}]},
            "open", 85,
        ),
        (
            "operating_status = closed",
            {"name": "Old Diner", "category": "restaurant",
             "operating_status": "closed"},
            "closed", 75,
        ),
        (
            "Dead website + no phone + no social",
            {"name": "Ghost Salon", "category": "salon",
             "website_status": "likely_closed"},
            "closed", 75,
        ),
        (
            "Closure keyword in name",
            {"name": "Former Bookstore", "category": "books",
             "confidence": 0.5},
            "closed", 75,
        ),
        (
            "No website / no phone / valid address (sparse open)",
            {"name": "Corner Deli", "category": "food",
             "address": "123 Main St", "confidence": 0.6,
             "sources": [{"dataset": "a", "confidence": 0.6}]},
            "open", 50,
        ),
        (
            "Single source, decent confidence (open low-confidence)",
            {"name": "Blue Moon Gallery", "category": "art_gallery",
             "confidence": 0.65,
             "sources": [{"dataset": "a", "confidence": 0.65}]},
            "open", 50,
        ),
    ]

    all_pass = True
    for desc, place, expected_status, min_conf in test_cases:
        result = scorer.score(place)
        status_ok = result.status == expected_status
        conf_ok   = result.confidence >= min_conf
        ok = status_ok and conf_ok
        all_pass = all_pass and ok
        tag = "PASS" if ok else "FAIL"
        print(f"\n  [{tag}] {desc}")
        print(f"         Expected : status={expected_status}, conf>={min_conf}%")
        print(
            f"         Got      : status={result.status}, "
            f"conf={result.confidence:.0f}%, raw={result.raw_score:.3f}"
        )
        if not ok:
            sigs = ", ".join(s.name for s in result.fired_signals) or "(none)"
            print(f"         Signals  : {sigs}")

    print(f"\n{sep}")
    if all_pass:
        print("ALL SPOT CHECKS PASSED")
    else:
        print("SOME SPOT CHECKS FAILED — review weights in scorer.py")
    print(sep)
    print("\nDone. Predictions stored in predicted_status, prediction_confidence,")
    print("prediction_updated_at. Confidence is now reported as 50–99%.")


if __name__ == "__main__":
    main()
