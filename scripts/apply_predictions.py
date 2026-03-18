"""
apply_predictions.py — Apply predictions to all 1.4M places.

Mode selection (automatic):
  1. If scripts/models/xgboost_licensed.pkl + feature_columns.json exist,
     use the XGBoost model trained on real business license ground truth data.
  2. Otherwise fall back to the pure-signal scorer (scorer.py).

Updates three columns on the places table:
  predicted_status       VARCHAR(20)  — 'open' | 'closed'
  prediction_confidence  FLOAT        — 50-99 (percentage)
  prediction_updated_at  TIMESTAMP    — when the prediction was last written

Usage (Windows — set PYTHONIOENCODING=utf-8 to avoid cp1252 errors):
    set PYTHONIOENCODING=utf-8
    python scripts/apply_predictions.py [--batch-size N] [--offset N] [--local] [--db-url URL]

    --scorer-only   Force use of signal-based scorer even if model files exist

After completion prints:
  - Mode used (XGBoost licensed model / signal scorer)
  - Total open vs closed counts and percentages
  - Average confidence for open and closed predictions
  - Which signals fired most frequently (top 10, scorer mode only)
  - Top 10 categories with highest closed prediction rate
  - Chain verification (no known chains predicted closed)
  - 30 sample closed predictions
  - Spot check pass/fail results (scorer mode only)
"""

import json
import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from stillopen.backend.app.scorer import PlaceScorer, KNOWN_CHAINS

# ── XGBoost model paths ───────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "scripts", "models")
MODEL_PKL = os.path.join(MODELS_DIR, "xgboost_licensed.pkl")
FEATURE_JSON = os.path.join(MODELS_DIR, "feature_columns.json")

HIGH_TURNOVER_CATEGORIES = {
    "restaurant", "cafe", "bar", "pub", "fast_food", "fast food", "food_court",
    "clothes", "clothing", "shoes", "boutique", "fashion",
    "beauty", "beauty salon", "hair salon", "hairdresser", "nail_salon", "nail salon",
    "dry_cleaning", "dry cleaning", "laundry",
    "gift", "gift shop", "souvenir", "toy", "toys",
    "furniture", "home_goods", "home goods", "interior_decoration",
    "video_games", "video games", "bookstore", "books",
    "department_store", "department store",
    "ice_cream", "ice cream", "dessert", "bakery",
    "florist", "flowers", "art_gallery", "art gallery",
    "antique", "antiques", "vintage",
}

CLOSURE_KEYWORDS = [
    "closed", "former", "defunct", "out of business", "coming soon",
    "vacant", "empty", "available", "for lease", "for rent",
]


def _has(x) -> int:
    if x is None:
        return 0
    if isinstance(x, (list, tuple)):
        return 1 if len(x) > 0 else 0
    if isinstance(x, dict):
        return 1 if len(x) > 0 else 0
    s = str(x).strip()
    return 0 if s.lower() in ("none", "null", "nan", "", "[]", "{}") else 1


def _count(x) -> int:
    return len(x) if isinstance(x, (list, tuple)) else 0


def extract_model_features(row: dict, meta: dict) -> dict:
    """
    Extract the same features used by build_training_set.py / train_xgboost.py.
    License-derived features (has_end_date, etc.) are set to 0/-1 since the DB
    does not have license data — the model was trained to handle this.
    """
    websites = meta.get("websites") or []
    phones = meta.get("phones") or []
    socials = meta.get("socials") or []
    emails = meta.get("emails") or []
    brand = meta.get("brand") or {}
    sources = meta.get("sources") or []

    has_website = _has(websites)
    has_phone = _has(phones)
    has_social = _has(socials)
    has_email = _has(emails)
    has_brand = _has(brand)
    has_address = _has(row.get("address"))
    num_websites = _count(websites)
    num_phones = _count(phones)
    num_socials = _count(socials)

    num_sources = len(sources) if isinstance(sources, list) else 0
    confs, min_days = [], 9999
    for s in (sources if isinstance(sources, list) else []):
        if isinstance(s, dict):
            try:
                confs.append(float(s["confidence"]))
            except (KeyError, TypeError, ValueError):
                pass
            if s.get("update_time"):
                try:
                    ts = pd.to_datetime(str(s["update_time"]), utc=True)
                    d = (datetime.now(ts.tzinfo) - ts).days
                    min_days = min(min_days, max(0, d))
                except Exception:
                    pass
    source_mean_confidence = float(sum(confs) / len(confs)) if confs else 0.0
    days_since_last_update = min_days if min_days != 9999 else 365

    confidence = float(meta.get("confidence") or 0)
    name = str(row.get("name") or "")
    category = str(row.get("category") or "").lower()

    name_length = len(name)
    has_closure_keyword = 1 if any(kw in name.lower() for kw in CLOSURE_KEYWORDS) else 0
    high_turnover = 1 if category in HIGH_TURNOVER_CATEGORIES else 0

    digital_presence = has_website + has_social + has_phone + has_email + has_brand
    metadata_completeness = (
        has_website * 0.2 + has_social * 0.15 + has_phone * 0.2 +
        has_email * 0.1 + has_brand * 0.15 + has_address * 0.1 +
        (1 if num_sources > 1 else 0) * 0.1
    )
    confidence_x_sources = confidence * num_sources
    digital_x_confidence = digital_presence * confidence
    sources_x_recency = num_sources / max(1, days_since_last_update + 1)
    web_to_social_ratio = num_websites / (num_socials + 1)
    phone_to_web_ratio = num_phones / (num_websites + 1)
    is_stale = 1 if days_since_last_update > 180 else 0
    website_verified_closed = 1 if meta.get("website_status") == "likely_closed" else 0

    return {
        # License features — 0/-1 when called from apply_predictions (no license data)
        "has_end_date": 0,
        "has_location_end_date": 0,
        "end_date_days_ago": -1,
        "location_end_date_days_ago": -1,
        "license_age_days": -1,
        "name_match_score": 0,
        "address_match_score": 0,
        "category_match": 0,
        # Overture/OSM features
        "has_website": has_website,
        "num_websites": num_websites,
        "has_social": has_social,
        "num_socials": num_socials,
        "has_phone": has_phone,
        "num_phones": num_phones,
        "has_email": has_email,
        "has_brand": has_brand,
        "has_address": has_address,
        "confidence": confidence,
        "num_sources": num_sources,
        "source_mean_confidence": source_mean_confidence,
        "days_since_last_update": days_since_last_update,
        "name_length": name_length,
        "has_closure_keyword": has_closure_keyword,
        "high_turnover_category": high_turnover,
        "digital_presence": digital_presence,
        "metadata_completeness": metadata_completeness,
        "confidence_x_sources": confidence_x_sources,
        "digital_x_confidence": digital_x_confidence,
        "sources_x_recency": sources_x_recency,
        "web_to_social_ratio": web_to_social_ratio,
        "phone_to_web_ratio": phone_to_web_ratio,
        "is_stale": is_stale,
        "website_verified_closed": website_verified_closed,
    }


def load_xgboost_model():
    """Load the licensed XGBoost model if files exist. Returns (model_bundle, feature_cols) or (None, None)."""
    if not (os.path.exists(MODEL_PKL) and os.path.exists(FEATURE_JSON)):
        return None, None
    try:
        import joblib
        bundle = joblib.load(MODEL_PKL)
        with open(FEATURE_JSON, encoding="utf-8") as f:
            feature_cols = json.load(f)
        return bundle, feature_cols
    except Exception as e:
        print(f"  WARN: Could not load XGBoost model: {e}")
        return None, None


def predict_batch_xgboost(batch_rows: list, model_bundle: dict, feature_cols: list) -> list:
    """
    Run XGBoost predictions on a batch.
    Returns list of (status, confidence) tuples matching order of batch_rows.
    """
    clf = model_bundle["model"]
    threshold = model_bundle.get("optimal_threshold", 0.5)

    records = []
    for row in batch_rows:
        meta = row.get("metadata_json") or {}
        if not isinstance(meta, dict):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        feats = extract_model_features(row, meta)
        records.append(feats)

    df = pd.DataFrame(records)

    # Handle category encoding if the model was trained with it
    cat_freq = model_bundle.get("category_freq", {})
    if "category_freq_score" in feature_cols and "category_freq_score" not in df.columns:
        df["category_freq_score"] = 0.0
    le = model_bundle.get("label_encoder")
    if "category_label" in feature_cols and "category_label" not in df.columns:
        df["category_label"] = 0

    # Align columns to exactly what was trained on
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0).astype(float)

    y_prob = clf.predict_proba(X)[:, 1]  # prob of open (label=1)

    results = []
    for prob in y_prob:
        is_open = prob >= threshold
        status = "open" if is_open else "closed"
        # Map probability distance from threshold to 50–99 confidence range
        dist = abs(prob - threshold) / max(threshold, 1 - threshold)
        confidence = float(round(min(99.0, max(50.0, 50.0 + dist * 49.0))))
        results.append((status, confidence))

    return results


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
    parser.add_argument("--scorer-only", action="store_true",
                        help="Force signal-based scorer even if model files exist")
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

    # ── Model or scorer selection ──────────────────────────────────────────────
    use_xgboost = False
    model_bundle = None
    feature_cols = None

    if not args.scorer_only:
        model_bundle, feature_cols = load_xgboost_model()
        if model_bundle is not None:
            use_xgboost = True

    if use_xgboost:
        print(f"StillOpen — XGBoost Licensed Model Batch Predictor")
        print(f"  Model     : {MODEL_PKL}")
        print(f"  Threshold : {model_bundle.get('optimal_threshold', 0.5):.2f}")
        print(f"  Trained at: {model_bundle.get('trained_at', 'unknown')}")
        print(f"  Features  : {len(feature_cols)}")
    else:
        print(f"StillOpen — Signal-Based Batch Predictor v2")
        if args.scorer_only:
            print(f"  (--scorer-only flag set, skipping XGBoost model)")
        else:
            print(f"  (No XGBoost model found at {MODEL_PKL}, using scorer)")

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

        if use_xgboost:
            # ── XGBoost batch prediction ───────────────────────────────────
            preds = predict_batch_xgboost(batch_rows, model_bundle, feature_cols)
            for row, (status, confidence) in zip(batch_rows, preds):
                cat = (row.get("category") or "unknown").lower()
                category_totals[cat] += 1
                if status == "closed":
                    n_closed += 1
                    closed_conf_sum += confidence
                    category_closed[cat] += 1
                else:
                    n_open += 1
                    open_conf_sum += confidence
                updates.append((status, confidence, now_ts, row["place_id"]))
        else:
            # ── Signal-based scorer ────────────────────────────────────────
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

    # Mode label
    mode_label = "XGBoost Licensed Model" if use_xgboost else "Signal-Based Scorer"
    print(f"\n  Prediction mode  : {mode_label}")

    # Top 10 signals (scorer mode only)
    if not use_xgboost and signal_counts:
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

        print(f"\n  [{i:2d}] {name[:50]}")
        print(f"       Category   : {cat}")
        print(f"       City       : {city}")
        print(f"       Confidence : {conf:.0f}%  |  Overture conf: {ov_conf:.2f}")
        print(f"       Website    : {ws}")

        # Re-score with scorer to show fired signals (informational in both modes)
        place = {
            "name": name,
            "category": row["category"] or "",
            "website_status": ws,
            "confidence": ov_conf,
        }
        rescore = scorer.score(place)
        signal_names = ", ".join(s.name for s in rescore.fired_signals) or "(none)"
        print(f"       Signals    : {signal_names}")

    conn2.close()

    # ── Correctness spot checks (scorer mode only) ───────────────────────────
    if use_xgboost:
        print(f"\n{sep}")
        print("CORRECTNESS SPOT CHECKS — skipped (XGBoost model mode)")
        print(sep)
        print("\nDone. Predictions stored in predicted_status, prediction_confidence,")
        print("prediction_updated_at. (XGBoost Licensed Model)")
        return

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
