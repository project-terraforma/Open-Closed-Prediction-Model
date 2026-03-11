"""
train_from_db.py — Retrain the open/closed prediction model using:
  1. Original project_c_samples.parquet   (has ground-truth labels)
  2. scripts/data/osm_places.json          (has ground-truth labels)
  3. DB records with website_status verified (likely_closed → 0, active → 1)

Trains an XGBoost model (replaces the old Random Forest) with the same 26
feature schema as features.py, plus one new feature:
  website_verified_closed: 1 if website_status == 'likely_closed', else 0

Usage:
    python scripts/train_from_db.py [--db-verified-only]

Options:
    --db-verified-only   Only use website-verified DB records (skips parquet + OSM)

Output:
    stillopen/backend/model/open_model.pkl  (overwrites existing model)
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# HIGH_TURNOVER_CATEGORIES — keep in sync with features.py
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

FEATURE_COLS = [
    'has_website', 'num_websites', 'has_social', 'num_socials',
    'has_phone', 'num_phones', 'has_email', 'has_brand', 'has_address',
    'confidence', 'num_sources', 'source_mean_confidence',
    'days_since_last_update', 'name_length', 'has_closure_keyword',
    'category_freq_score', 'category_label',
    'digital_presence', 'metadata_completeness',
    'confidence_x_sources', 'digital_x_confidence', 'sources_x_recency',
    'web_to_social_ratio', 'phone_to_web_ratio',
    'is_stale', 'high_turnover_category',
    'website_verified_closed',  # NEW: 1 if website checked and found dead
]


# ============================================================================
# FEATURE EXTRACTION HELPERS
# ============================================================================

def _has(x):
    """Return 1 if x is a non-empty list/array/dict/string, else 0."""
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


def _extract_sources(sources_raw):
    """Return (num_sources, mean_confidence, days_since_update)."""
    if not isinstance(sources_raw, (list, np.ndarray)) or len(sources_raw) == 0:
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


def _build_record(name, category, websites, phones, socials, emails, brand,
                  addresses, confidence, sources_raw, open_label,
                  data_source, website_status=None):
    """Build a flat feature dict from raw Overture-style fields."""
    has_website = _has(websites)
    has_phone = _has(phones)
    has_social = _has(socials)
    has_email = _has(emails)
    has_brand = _has(brand)
    has_address = _has(addresses)

    num_websites = _count(websites)
    num_phones = _count(phones)
    num_socials = _count(socials)

    num_sources, mean_conf, days = _extract_sources(sources_raw)

    name_lower = (name or '').lower()
    name_length = len(name or '')
    has_closure_keyword = 1 if any(kw in name_lower for kw in CLOSURE_KEYWORDS) else 0
    high_turnover = 1 if (category or '').lower() in HIGH_TURNOVER_CATEGORIES else 0

    digital_presence = has_website + has_social + has_phone + has_email + has_brand
    metadata_completeness = (
        has_website * 0.2 + has_social * 0.15 + has_phone * 0.2 +
        has_email * 0.1 + has_brand * 0.15 + has_address * 0.1 +
        (1 if num_sources > 1 else 0) * 0.1
    )
    confidence_x_sources = confidence * num_sources
    digital_x_confidence = digital_presence * confidence
    sources_x_recency = num_sources / max(1, days + 1)
    web_to_social_ratio = num_websites / (num_socials + 1)
    phone_to_web_ratio = num_phones / (num_websites + 1)
    is_stale = 1 if days > 180 else 0

    # New feature: website verified as dead (1) or not (0)
    website_verified_closed = 1 if website_status == 'likely_closed' else 0

    return {
        'name': name or '',
        'category': category or 'unknown',
        'has_website': has_website,
        'num_websites': num_websites,
        'has_social': has_social,
        'num_socials': num_socials,
        'has_phone': has_phone,
        'num_phones': num_phones,
        'has_email': has_email,
        'has_brand': has_brand,
        'has_address': has_address,
        'confidence': float(confidence or 0),
        'num_sources': num_sources,
        'source_mean_confidence': mean_conf,
        'days_since_last_update': days,
        'name_length': name_length,
        'has_closure_keyword': has_closure_keyword,
        'high_turnover_category': high_turnover,
        'digital_presence': digital_presence,
        'metadata_completeness': metadata_completeness,
        'confidence_x_sources': confidence_x_sources,
        'digital_x_confidence': digital_x_confidence,
        'sources_x_recency': sources_x_recency,
        'web_to_social_ratio': web_to_social_ratio,
        'phone_to_web_ratio': phone_to_web_ratio,
        'is_stale': is_stale,
        'website_verified_closed': website_verified_closed,
        'open': int(open_label),
        'data_source': data_source,
    }


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_original_parquet():
    path = os.path.join(PROJECT_ROOT, "data", "project_c_samples.parquet")
    if not os.path.exists(path):
        print("  ✗ project_c_samples.parquet not found")
        return []

    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        names = row.get('names', {})
        name = names.get('primary', 'Unknown') if isinstance(names, dict) else 'Unknown'
        cats = row.get('categories', {})
        category = cats.get('primary', 'unknown') if isinstance(cats, dict) else 'unknown'

        websites = row.get('websites')
        phones = row.get('phones')
        socials = row.get('socials')
        emails = row.get('emails')
        brand = row.get('brand')
        addresses = row.get('addresses')
        sources = row.get('sources')

        # Convert numpy arrays to lists
        for field in [websites, phones, socials, emails, addresses, sources]:
            if isinstance(field, np.ndarray):
                field = field.tolist()

        rec = _build_record(
            name=name, category=category,
            websites=websites if isinstance(websites, (list, np.ndarray)) else [],
            phones=phones if isinstance(phones, (list, np.ndarray)) else [],
            socials=socials if isinstance(socials, (list, np.ndarray)) else [],
            emails=emails if isinstance(emails, (list, np.ndarray)) else [],
            brand=brand if isinstance(brand, dict) else {},
            addresses=addresses if isinstance(addresses, (list, np.ndarray)) else [],
            confidence=float(row.get('confidence', 0)),
            sources_raw=sources.tolist() if isinstance(sources, np.ndarray) else (sources or []),
            open_label=int(row.get('open', 1)),
            data_source='original',
        )
        records.append(rec)

    closed = sum(1 for r in records if r['open'] == 0)
    print(f"  ✓ Original parquet: {len(records)} records ({closed} closed / {len(records)-closed} open)")
    return records


def load_osm_data():
    path = os.path.join(PROJECT_ROOT, "scripts", "data", "osm_places.json")
    if not os.path.exists(path):
        print("  ✗ osm_places.json not found")
        return []

    with open(path) as f:
        data = json.load(f)

    records = []
    for item in data:
        meta = item.get('metadata', {})
        if isinstance(meta, str):
            meta = json.loads(meta)

        websites = meta.get('websites', [])
        phones = meta.get('phones', [])
        socials = meta.get('socials', [])

        rec = _build_record(
            name=item.get('name', 'Unknown'),
            category=item.get('category', 'unknown'),
            websites=websites, phones=phones, socials=socials,
            emails=[], brand={}, addresses=[item.get('address', '')],
            confidence=float(meta.get('confidence', 0.5)),
            sources_raw=[{'confidence': float(meta.get('confidence', 0.5))}],
            open_label=int(item.get('open', 1)),
            data_source='osm',
        )
        records.append(rec)

    closed = sum(1 for r in records if r['open'] == 0)
    print(f"  ✓ OSM data: {len(records)} records ({closed} closed / {len(records)-closed} open)")
    return records


def load_db_verified():
    """Load DB records that have been website-verified."""
    try:
        from dotenv import load_dotenv
        import psycopg2
        import psycopg2.extras
        load_dotenv(os.path.join(PROJECT_ROOT, "stillopen", "backend", ".env"))
        # Default to local postgres; override via LOCAL_DB env var or fallback
        db_url = os.environ.get("LOCAL_DB_URL",
                  os.environ.get("DATABASE_URL", "postgresql://localhost:5432/stillopen"))
        db_url = db_url.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"  ✗ DB connection failed: {e}")
        return []

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT place_id, name, category, address, metadata_json
            FROM places
            WHERE metadata_json->>'website_status' IN ('likely_closed', 'active')
        """)
        rows = cur.fetchall()
    conn.close()

    records = []
    for r in rows:
        meta = r['metadata_json'] or {}
        website_status = meta.get('website_status')

        # Label: likely_closed → closed (0), active → open (1)
        open_label = 0 if website_status == 'likely_closed' else 1

        websites = meta.get('websites', [])
        phones = meta.get('phones', [])
        socials = meta.get('socials', [])
        emails = meta.get('emails', [])
        brand = meta.get('brand', {})
        sources_raw = meta.get('sources', [])

        rec = _build_record(
            name=r['name'], category=r['category'],
            websites=websites, phones=phones, socials=socials,
            emails=emails, brand=brand, addresses=[r['address'] or ''],
            confidence=float(meta.get('confidence', 0)),
            sources_raw=sources_raw,
            open_label=open_label,
            data_source='db_verified',
            website_status=website_status,
        )
        records.append(rec)

    closed = sum(1 for r in records if r['open'] == 0)
    print(f"  ✓ DB verified: {len(records)} records ({closed} likely_closed / {len(records)-closed} active)")
    return records


# ============================================================================
# TRAINING
# ============================================================================

def train_xgboost(df):
    """Train XGBoost classifier and return (model, label_encoder, cat_freq)."""
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, f1_score, roc_auc_score
    import warnings
    warnings.filterwarnings('ignore')

    # Category encoding
    cat_freq = df['category'].value_counts(normalize=True).to_dict()
    df = df.copy()
    df['category_freq_score'] = df['category'].map(cat_freq).fillna(0)

    le = LabelEncoder()
    df['category_label'] = le.fit_transform(df['category'].fillna('unknown'))

    X = df[FEATURE_COLS].fillna(0).astype(float)
    y = df['open'].astype(int)

    print(f"\n{'='*60}")
    print("TRAINING DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records : {len(X)}")
    print(f"  Open  (1)     : {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"  Closed (0)    : {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

    by_source = df.groupby('data_source')['open'].agg(['count', 'mean'])
    print("\n  By data source:")
    for src, row in by_source.iterrows():
        print(f"    {src:20s}: {int(row['count']):5d} records ({row['mean']*100:.1f}% open)")

    print(f"\n  Features ({len(FEATURE_COLS)}):")
    for f in FEATURE_COLS:
        print(f"    {f}")

    # XGBoost with scale_pos_weight to handle class imbalance
    # scale_pos_weight = (# negative / # positive) controls weight for minority class (closed)
    n_closed = (y == 0).sum()
    n_open = (y == 1).sum()
    # We use scale_pos_weight on the minority (closed=0) by treating class 0 as "positive" via eval_metric
    # For XGBoost binary:logistic, label 1 is positive. Since we want to penalise missing closed:
    scale_weight = round(n_open / max(1, n_closed), 2)
    print(f"\n  scale_pos_weight (open/closed): {scale_weight}")

    # class_weight dictionary equivalent via sample_weight
    sample_weights = np.where(y == 0, scale_weight, 1.0)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION (5-Fold Stratified)")
    print(f"{'='*60}")

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Compute CV with sample weights not directly supported in cross_val_score;
    # do it manually for the closed-class F1
    f1_scores = []
    roc_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        sw_tr = sample_weights[train_idx]

        clf.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
        y_prob = clf.predict_proba(X_val)[:, 1]
        # Optimal threshold search for closed-class F1
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.2, 0.8, 0.02):
            y_pred_t = (y_prob >= t).astype(int)
            f1 = f1_score(y_val, y_pred_t, pos_label=0, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        f1_scores.append(best_f1)
        try:
            roc_scores.append(roc_auc_score(y_val, y_prob))
        except Exception:
            roc_scores.append(0.5)
        print(f"  Fold {fold}: closed-F1={best_f1:.4f}  ROC-AUC={roc_scores[-1]:.4f}  threshold={best_t:.2f}")

    print(f"\n  Mean closed-F1 : {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
    print(f"  Mean ROC-AUC  : {np.mean(roc_scores):.4f} (±{np.std(roc_scores):.4f})")

    # Final model fit on full dataset
    print(f"\n{'='*60}")
    print("FITTING FINAL MODEL ON FULL DATASET")
    print(f"{'='*60}")
    clf.fit(X, y, sample_weight=sample_weights, verbose=False)
    y_pred = clf.predict(X)
    print("\n  Full-dataset classification report:")
    print(classification_report(y, y_pred, target_names=['CLOSED', 'OPEN'], digits=4))

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("  Top 10 feature importances:")
    for feat, imp in importances.head(10).items():
        bar = '█' * int(imp * 40)
        print(f"    {feat:35s} {imp:.4f}  {bar}")

    # Global threshold optimisation on full dataset
    y_prob_full = clf.predict_proba(X)[:, 1]
    best_f1, best_threshold = 0.0, 0.5
    for t in np.arange(0.2, 0.8, 0.01):
        y_pred_t = (y_prob_full >= t).astype(int)
        f1 = f1_score(y, y_pred_t, pos_label=0, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    print(f"\n  Optimal threshold for closed-class F1: {best_threshold:.2f}  (F1={best_f1:.4f})")

    # How many records the model predicts closed at optimal threshold
    y_pred_final = (y_prob_full >= best_threshold).astype(int)
    n_pred_closed = (y_pred_final == 0).sum()
    print(f"  Predicted closed on training set: {n_pred_closed} / {len(y)} ({n_pred_closed/len(y)*100:.1f}%)")

    return clf, le, cat_freq, best_threshold, importances


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-verified-only", action="store_true",
                        help="Only use DB-verified records for training")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("LOADING DATA SOURCES")
    print(f"{'='*60}")

    all_records = []

    if not args.db_verified_only:
        orig = load_original_parquet()
        osm = load_osm_data()
        all_records.extend(orig)
        all_records.extend(osm)

    db_verified = load_db_verified()
    all_records.extend(db_verified)

    if not all_records:
        print("ERROR: No records loaded. Run verify_businesses.py first to get verified labels.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    total_closed = (df['open'] == 0).sum()
    total_open = (df['open'] == 1).sum()

    print(f"\n  Combined: {len(df)} records — {total_closed} closed / {total_open} open")

    if total_closed == 0:
        print("ERROR: No closed-label records found. Cannot train a meaningful model.")
        print("  Run: python scripts/verify_businesses.py --source db --limit 500 --write-db --filter high_risk")
        sys.exit(1)

    # Category encoding requires fitting on combined data
    # (already done per-source in _build_record for frequency; re-fit here)
    cat_freq_all = df['category'].value_counts(normalize=True).to_dict()
    df['category_freq_score'] = df['category'].map(cat_freq_all).fillna(0)

    from sklearn.preprocessing import LabelEncoder
    le_all = LabelEncoder()
    df['category_label'] = le_all.fit_transform(df['category'].fillna('unknown'))

    clf, le, cat_freq, threshold, importances = train_xgboost(df)

    # Save model
    import joblib
    model_dir = os.path.join(PROJECT_ROOT, "stillopen", "backend", "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "open_model.pkl")

    joblib.dump({
        'model': clf,
        'feature_names': FEATURE_COLS,
        'label_encoder': le_all,
        'category_freq': cat_freq_all,
        'optimal_threshold': threshold,
        'training_samples': len(df),
        'data_sources': list(df['data_source'].unique()),
        'model_type': 'xgboost',
        'trained_at': datetime.now().isoformat(),
    }, model_path)

    size_mb = os.path.getsize(model_path) / 1_000_000
    print(f"\n{'='*60}")
    print(f"  Model saved to {model_path}  ({size_mb:.1f} MB)")
    print(f"  Training samples : {len(df)}")
    print(f"  Optimal threshold: {threshold:.2f}")
    print(f"  Model type       : XGBoost XGBClassifier")
    print(f"{'='*60}")
