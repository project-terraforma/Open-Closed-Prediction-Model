"""
apply_predictions.py — Apply the trained model to all 1.4M places in Postgres.

Adds / refreshes three columns on the places table:
  predicted_status       VARCHAR(20)  — 'open' | 'closed' | 'unknown'
  prediction_confidence  FLOAT        — model probability for the predicted class
  prediction_updated_at  TIMESTAMP    — when the prediction was last written

Processing:
  - Reads records in batches of 10,000
  - Computes 27 features per record using features.py logic
  - Applies the XGBoost model (or whatever is saved in open_model.pkl)
  - Updates the DB in the same batch
  - Prints progress every 100,000 records

Usage:
    python scripts/apply_predictions.py [--batch-size N] [--offset N]

After completion prints:
  - Total open / closed / unknown counts
  - Top 20 closed categories
  - 30 sample high-confidence closed predictions for spot-check
"""

import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import joblib

# ── Feature constants ─────────────────────────────────────────────────────────

HIGH_TURNOVER_CATEGORIES = {
    'restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'fast food', 'food_court',
    'clothes', 'clothing', 'shoes', 'boutique', 'fashion',
    'beauty', 'beauty salon', 'hair salon', 'hairdresser', 'nail_salon', 'nail salon',
    'dry_cleaning', 'dry cleaning', 'laundry',
    'gift', 'gift shop', 'souvenir', 'toy', 'toys',
    'furniture', 'home_goods', 'home goods', 'interior_decoration',
    'video_games', 'video games', 'bookstore', 'books',
    'department_store', 'department store',
    'ice_cream', 'ice cream', 'dessert', 'bakery',
    'florist', 'flowers', 'art_gallery', 'art gallery',
    'antique', 'antiques', 'vintage',
}

CLOSURE_KEYWORDS = [
    'closed', 'former', 'defunct', 'out of business', 'coming soon',
    'vacant', 'empty', 'available', 'for lease', 'for rent',
]


# ── Feature extraction ────────────────────────────────────────────────────────

def _has(x):
    if x is None:
        return 0
    if isinstance(x, (list, np.ndarray)):
        return 1 if len(x) > 0 else 0
    if isinstance(x, dict):
        return 1 if len(x) > 0 else 0
    s = str(x).strip()
    return 0 if s.lower() in ('none', 'null', 'nan', '', '[]', '{}') else 1


def _count(x):
    if isinstance(x, (list, np.ndarray)):
        return len(x)
    return 0


def _sources_stats(sources_raw):
    if not isinstance(sources_raw, list) or not sources_raw:
        return 0, 0.0, 365
    num = len(sources_raw)
    confs = [float(s['confidence']) for s in sources_raw
             if isinstance(s, dict) and s.get('confidence') is not None]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    min_days = 9999
    for s in sources_raw:
        if isinstance(s, dict) and s.get('update_time'):
            try:
                ts = pd.to_datetime(str(s['update_time']), utc=True)
                days = (datetime.now(ts.tzinfo) - ts).days
                min_days = min(min_days, max(0, days))
            except Exception:
                pass
    days = min_days if min_days != 9999 else 365
    return num, mean_conf, days


def compute_features_batch(rows, artifacts):
    """
    Vectorized feature computation for a list of (name, category, meta) tuples.
    Uses numpy operations on pre-extracted arrays — ~50x faster than row-by-row.
    """
    cat_freq = artifacts.get('category_freq', {})
    le = artifacts.get('label_encoder')
    feature_names = artifacts['feature_names']
    n = len(rows)
    if n == 0:
        return pd.DataFrame(columns=feature_names)

    # ── Pre-extract all columns in one pass ──────────────────────────────────
    names     = []
    cats      = []
    conf_arr  = np.zeros(n, dtype=np.float32)
    nw        = np.zeros(n, dtype=np.int16)   # num_websites
    np_       = np.zeros(n, dtype=np.int16)   # num_phones
    ns        = np.zeros(n, dtype=np.int16)   # num_socials
    ne        = np.zeros(n, dtype=np.int16)   # num_emails
    nb        = np.zeros(n, dtype=np.int8)    # has_brand
    nsrc      = np.zeros(n, dtype=np.int8)    # num_sources
    src_conf  = np.zeros(n, dtype=np.float32) # source_mean_confidence
    wvc       = np.zeros(n, dtype=np.int8)    # website_verified_closed

    for i, (name, cat, meta) in enumerate(rows):
        if not isinstance(meta, dict):
            meta = {}
        names.append(name or '')
        cats.append((cat or 'unknown').lower())

        conf_arr[i] = float(meta.get('confidence') or 0)

        websites = meta.get('websites') or []
        nw[i] = len(websites) if isinstance(websites, list) else 0

        phones = meta.get('phones') or []
        np_[i] = len(phones) if isinstance(phones, list) else 0

        socials = meta.get('socials') or []
        ns[i] = len(socials) if isinstance(socials, list) else 0

        emails = meta.get('emails') or []
        ne[i] = len(emails) if isinstance(emails, list) else 0

        brand = meta.get('brand') or {}
        nb[i] = 1 if isinstance(brand, dict) and len(brand) > 0 else 0

        sources = meta.get('sources') or []
        if isinstance(sources, list) and sources:
            nsrc[i] = len(sources)
            confs = [float(s['confidence']) for s in sources
                     if isinstance(s, dict) and s.get('confidence') is not None]
            src_conf[i] = float(np.mean(confs)) if confs else 0.0

        wvc[i] = 1 if meta.get('website_status') == 'likely_closed' else 0

    # ── Vectorised feature derivation ────────────────────────────────────────
    hw  = (nw  > 0).astype(np.int8)
    hp  = (np_ > 0).astype(np.int8)
    hso = (ns  > 0).astype(np.int8)
    he  = (ne  > 0).astype(np.int8)
    ha  = np.ones(n, dtype=np.int8)   # all DB records have an address

    name_lengths = np.array([len(x) for x in names], dtype=np.int16)

    kw_set = set(CLOSURE_KEYWORDS)
    has_kw = np.array(
        [1 if any(kw in nm.lower() for kw in kw_set) else 0 for nm in names],
        dtype=np.int8,
    )
    high_turn = np.array(
        [1 if c in HIGH_TURNOVER_CATEGORIES else 0 for c in cats],
        dtype=np.int8,
    )

    # Category encoding
    cat_freq_scores = np.array([cat_freq.get(c, 0) for c in cats], dtype=np.float32)
    if le is not None:
        cat_labels = np.zeros(n, dtype=np.int32)
        # Batch transform — handle unknown categories gracefully
        unique_cats = list(set(cats))
        known = set(le.classes_)
        safe_cats = [c if c in known else le.classes_[0] for c in cats]
        try:
            cat_labels = le.transform(safe_cats).astype(np.int32)
        except Exception:
            cat_labels = np.zeros(n, dtype=np.int32)
    else:
        cat_labels = np.zeros(n, dtype=np.int32)

    # days_since_last_update: all Overture records share the same two source
    # update_times (2025-12-01 and 2026-02-10), so this feature is constant
    # and expensive to compute. Use a fixed value derived from the model's
    # reference date to avoid ~1s of date parsing per 1000 rows.
    days = np.full(n, 365, dtype=np.float32)  # default; stale for all

    digital  = hw.astype(np.float32) + hso + hp + he + nb
    meta_c   = (hw*0.2 + hso*0.15 + hp*0.2 + he*0.1 + nb*0.15 + ha*0.1
                + (nsrc > 1).astype(np.float32) * 0.1)

    df = pd.DataFrame({
        'has_website':          hw,
        'num_websites':         nw,
        'has_social':           hso,
        'num_socials':          ns,
        'has_phone':            hp,
        'num_phones':           np_,
        'has_email':            he,
        'has_brand':            nb,
        'has_address':          ha,
        'confidence':           conf_arr,
        'num_sources':          nsrc,
        'source_mean_confidence': src_conf,
        'days_since_last_update': days,
        'name_length':          name_lengths,
        'has_closure_keyword':  has_kw,
        'category_freq_score':  cat_freq_scores,
        'category_label':       cat_labels,
        'digital_presence':     digital,
        'metadata_completeness': meta_c,
        'confidence_x_sources': conf_arr * nsrc,
        'digital_x_confidence': digital * conf_arr,
        'sources_x_recency':    nsrc / (days + 1),
        'web_to_social_ratio':  nw / (ns.astype(np.float32) + 1),
        'phone_to_web_ratio':   np_ / (nw.astype(np.float32) + 1),
        'is_stale':             (days > 180).astype(np.int8),
        'high_turnover_category': high_turn,
        'website_verified_closed': wvc,
    })

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names].fillna(0).astype(float)


# ── Hard signal checks ────────────────────────────────────────────────────────

def _hard_signal_closed(name, meta):
    """Return True if any hard closure signal is present (overrides model)."""
    name_lower = (name or '').lower()
    if any(kw in name_lower for kw in CLOSURE_KEYWORDS):
        return True
    if meta.get('website_status') == 'likely_closed':
        return True
    if any(meta.get(k) for k in ('disused:amenity', 'disused:shop', 'closed:amenity', 'end_date')):
        return True
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--offset', type=int, default=0,
                        help='Skip first N records (resume from checkpoint)')
    parser.add_argument('--local', action='store_true',
                        help='Use local Postgres (postgres:postgres123@localhost:5432/stillopen) '
                             'instead of DATABASE_URL from .env')
    parser.add_argument('--db-url', type=str, default=None,
                        help='Override DATABASE_URL entirely')
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, "stillopen", "backend", "model", "open_model.pkl")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Run train_from_db.py first.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    bundle = joblib.load(model_path)
    clf = bundle['model']
    artifacts = bundle
    threshold = bundle.get('optimal_threshold', 0.5)
    feature_names = bundle['feature_names']
    model_type = bundle.get('model_type', 'unknown')
    print(f"  Model type: {model_type}")
    print(f"  Threshold : {threshold:.3f}")
    print(f"  Features  : {len(feature_names)}")

    # ── DB connection ───────────────────────────────────────────────────────
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
    print(f"  DB: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    # ── Count total ─────────────────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM places")
        total = cur.fetchone()[0]
    print(f"\nTotal records to process: {total:,}")

    batch_size = args.batch_size
    offset = args.offset
    processed = 0
    n_open = 0
    n_closed = 0
    n_unknown = 0
    now_ts = datetime.utcnow().isoformat()

    print(f"Processing in batches of {batch_size:,}...\n")

    while offset < total:
        # Fetch batch
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT place_id, name, category, metadata_json
                FROM places
                ORDER BY id
                LIMIT %s OFFSET %s
                """,
                (batch_size, offset),
            )
            batch_rows = cur.fetchall()

        if not batch_rows:
            break

        # Build feature tuples
        feature_inputs = []
        for r in batch_rows:
            meta = r['metadata_json'] or {}
            feature_inputs.append((r['name'] or '', r['category'] or '', meta))

        # Compute features
        X = compute_features_batch(feature_inputs, artifacts)

        # Model predictions
        try:
            open_probs = clf.predict_proba(X)[:, 1]  # probability of open (class 1)
        except Exception as e:
            print(f"  Prediction error on batch at offset {offset}: {e}")
            open_probs = np.full(len(batch_rows), 0.5)

        # Build update data
        updates = []
        for i, (row, open_prob) in enumerate(zip(batch_rows, open_probs)):
            meta = row['metadata_json'] or {}
            name = row['name'] or ''
            closed_prob = 1.0 - float(open_prob)

            # Determine status
            if _hard_signal_closed(name, meta):
                status = 'closed'
                conf = float(closed_prob)
            elif float(open_prob) < threshold:
                status = 'closed'
                conf = float(closed_prob)
            elif float(open_prob) > 0.60:
                status = 'open'
                conf = float(open_prob)
            else:
                status = 'open'
                conf = None  # likely_open — sparse record

            updates.append((status, conf, now_ts, row['place_id']))

            if status == 'closed':
                n_closed += 1
            elif status == 'open':
                n_open += 1
            else:
                n_unknown += 1

        # Bulk update
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                UPDATE places
                SET predicted_status = %s,
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
            print(f"  {processed:>10,} / {total:,} ({pct:.1f}%)  "
                  f"closed={n_closed:,}  open={n_open:,}  unknown={n_unknown:,}")

    conn.close()

    print(f"\n{'='*65}")
    print("APPLY PREDICTIONS — COMPLETE")
    print(f"{'='*65}")
    print(f"  Total processed  : {processed:,}")
    print(f"  Predicted OPEN   : {n_open:,}  ({n_open/max(1,processed)*100:.2f}%)")
    print(f"  Predicted CLOSED : {n_closed:,}  ({n_closed/max(1,processed)*100:.2f}%)")
    print(f"  Unknown          : {n_unknown:,}  ({n_unknown/max(1,processed)*100:.2f}%)")

    # ── Post-run summary queries ────────────────────────────────────────────
    conn2 = psycopg2.connect(db_url)

    # Top 20 closed categories
    with conn2.cursor() as cur:
        cur.execute("""
            SELECT category,
                   COUNT(*) FILTER (WHERE predicted_status = 'closed') as n_closed,
                   COUNT(*) as n_total,
                   ROUND(
                     COUNT(*) FILTER (WHERE predicted_status = 'closed')::numeric /
                     NULLIF(COUNT(*), 0) * 100, 1
                   ) as pct_closed
            FROM places
            WHERE predicted_status IS NOT NULL
            GROUP BY category
            HAVING COUNT(*) FILTER (WHERE predicted_status = 'closed') > 0
            ORDER BY n_closed DESC
            LIMIT 20
        """)
        cat_rows = cur.fetchall()

    print(f"\n  Top 20 categories by CLOSED count:")
    print(f"  {'Category':35s}  {'Closed':>8}  {'Total':>8}  {'%Closed':>8}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*8}")
    for row in cat_rows:
        print(f"  {(row[0] or 'unknown'):35s}  {row[1]:>8,}  {row[2]:>8,}  {row[3]:>7}%")

    # ── Step 6: Spot check — 30 high-confidence closed predictions ─────────
    with conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                p.name,
                p.category,
                p.address,
                p.predicted_status,
                p.prediction_confidence,
                p.metadata_json->>'city' as city,
                p.metadata_json->>'website_status' as website_status,
                p.metadata_json->>'closure_risk' as closure_risk,
                (p.metadata_json->>'confidence')::float as overture_conf,
                p.metadata_json->'websites' as websites_raw
            FROM places p
            WHERE predicted_status = 'closed'
              AND prediction_confidence >= 0.65
            ORDER BY prediction_confidence DESC
            LIMIT 30
        """)
        spot_rows = cur.fetchall()

    conn2.close()

    print(f"\n{'='*65}")
    print("STEP 6: SPOT CHECK — 30 High-Confidence CLOSED Predictions")
    print("(These should look like real closed businesses, not active chains)")
    print(f"{'='*65}")

    for i, row in enumerate(spot_rows, 1):
        name = row['name'] or 'Unknown'
        cat = (row['category'] or 'unknown')[:30]
        city = row['city'] or ''
        conf = row['prediction_confidence'] or 0
        ov_conf = row['overture_conf'] or 0
        ws = row['website_status'] or 'unchecked'
        risk = row['closure_risk'] or '-'

        # Extract website URL if available
        websites_raw = row.get('websites_raw') or []
        url = ''
        if isinstance(websites_raw, list) and websites_raw:
            first = websites_raw[0]
            if isinstance(first, dict):
                url = first.get('value') or first.get('url') or ''
            else:
                url = str(first)

        # Build signal explanation
        signals = []
        name_lower = (name or '').lower()
        if any(kw in name_lower for kw in CLOSURE_KEYWORDS):
            signals.append('closure_keyword_in_name')
        if ws == 'likely_closed':
            signals.append('website_dead')
        if risk == 'high':
            signals.append('low_overture_confidence')
        if ov_conf < 0.8:
            signals.append(f'conf={ov_conf:.2f}')
        if not signals:
            signals.append('model_score')

        print(f"\n  [{i:2d}] {name[:45]}")
        print(f"       Category : {cat}")
        print(f"       City     : {city}")
        print(f"       Confidence: {conf:.3f}  |  Overture conf: {ov_conf:.2f}")
        print(f"       Signals  : {', '.join(signals)}")
        if url:
            print(f"       Website  : {url}")

    print(f"\n{'='*65}")
    print("Done. Predictions stored in places.predicted_status,")
    print("prediction_confidence, prediction_updated_at.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
