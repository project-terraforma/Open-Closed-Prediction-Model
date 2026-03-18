"""
build_training_set.py — Build a labeled training dataset by cross-referencing
business license CSVs (or the golden dataset) against the Overture/OSM PostgreSQL database.

Matching strategy (combined confidence score):
  1. PostGIS ST_DWithin proximity (within ~50 metres)
  2. rapidfuzz token_sort_ratio name fuzzy match (threshold 80)
  3. Normalised address string match

Supported CSV schemas:

  Schema A (golden dataset):
    source, license_id, name, address, city, state, zip,
    latitude, longitude, category, ground_truth, confidence, reason
    — dates are parsed from the `reason` field
      e.g. "dba_end_date set: 2023-06-19", "location_end_date set: 2013-12-31"
      e.g. "No end date; dba_start: 1968-10-01"

  Schema B (raw license CSVs):
    source, license_id, name, street_address, city, state, zip,
    start_date, end_date, location_end_date, category,
    latitude, longitude, ground_truth, confidence, reason

ground_truth: 'open' | 'closed'

Output: training_set.csv with all extracted features + label column

Usage (Windows — set PYTHONIOENCODING=utf-8 to avoid cp1252 errors):
  set PYTHONIOENCODING=utf-8
  python scripts/build_training_set.py ^
      --licenses scripts/data/golden/golden_dataset.csv ^
      --output scripts/data/training_set.csv ^
      --db-url postgresql://postgres:postgres123@localhost:5432/stillopen
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, date

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from rapidfuzz import fuzz
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

TODAY = date.today()

# Regex to pull a YYYY-MM-DD date out of the reason field
_REASON_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_REASON_END_DATE_RE = re.compile(
    r"(?:dba_end_date|end_date)\s+set:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE
)
_REASON_LOC_END_RE = re.compile(
    r"location_end_date\s+set:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE
)
_REASON_START_RE = re.compile(
    r"dba_start:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE
)

# ── Feature columns produced by this script ──────────────────────────────────
# These match what train_xgboost.py expects (plus the existing OSM/Overture features).
LICENSE_FEATURE_COLS = [
    # License-derived features
    "has_end_date",
    "has_location_end_date",
    "end_date_days_ago",
    "location_end_date_days_ago",
    "license_age_days",
    "name_match_score",
    "address_match_score",
    "category_match",
    # Existing Overture/OSM features (mirrored from train_from_db.py / features.py)
    "has_website",
    "num_websites",
    "has_social",
    "num_socials",
    "has_phone",
    "num_phones",
    "has_email",
    "has_brand",
    "has_address",
    "confidence",
    "num_sources",
    "source_mean_confidence",
    "days_since_last_update",
    "name_length",
    "has_closure_keyword",
    "high_turnover_category",
    "digital_presence",
    "metadata_completeness",
    "confidence_x_sources",
    "digital_x_confidence",
    "sources_x_recency",
    "web_to_social_ratio",
    "phone_to_web_ratio",
    "is_stale",
    "website_verified_closed",
    # Label
    "label",
    # Metadata (not used as features)
    "license_id",
    "license_source",
    "license_name",
    "license_address",
    "db_place_id",
    "db_name",
    "db_address",
    "match_confidence",
]

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


# ── Address normalisation ─────────────────────────────────────────────────────

_UNIT_RE = re.compile(
    r"\b(unit|ste|suite|apt|apartment|#|floor|fl|bldg|building)\s*[\w\-]+",
    re.IGNORECASE,
)


def parse_dates_from_reason(reason: str) -> tuple:
    """
    Parse end_date, location_end_date, start_date from a golden-dataset reason string.
    Returns (end_date, location_end_date, start_date) as date | None each.
    Examples:
      "dba_end_date set: 2023-06-19"         → (2023-06-19, None, None)
      "location_end_date set: 2013-12-31"    → (None, 2013-12-31, None)
      "No end date; dba_start: 1968-10-01"   → (None, None, 1968-10-01)
    """
    if not reason or (isinstance(reason, float) and pd.isna(reason)):
        return None, None, None
    reason = str(reason)
    end_date = None
    loc_end_date = None
    start_date = None
    m = _REASON_END_DATE_RE.search(reason)
    if m:
        end_date = parse_date(m.group(1))
    m = _REASON_LOC_END_RE.search(reason)
    if m:
        loc_end_date = parse_date(m.group(1))
    m = _REASON_START_RE.search(reason)
    if m:
        start_date = parse_date(m.group(1))
    return end_date, loc_end_date, start_date


def get_address_field(lic_row: pd.Series) -> str:
    """Return address string regardless of whether column is 'address' or 'street_address'."""
    for col in ("street_address", "address"):
        val = lic_row.get(col)
        if val and not (isinstance(val, float) and pd.isna(val)):
            return str(val)
    return ""


def normalize_address(addr: str) -> str:
    """Strip unit/suite numbers, lowercase, collapse whitespace."""
    if not addr:
        return ""
    addr = _UNIT_RE.sub("", addr)
    addr = re.sub(r"\s+", " ", addr).strip().lower()
    return addr


# ── Date helpers ──────────────────────────────────────────────────────────────

def parse_date(val) -> date | None:
    if not val or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def days_ago(d: date | None) -> int:
    """Return days since date, or -1 if date is None."""
    if d is None:
        return -1
    return max(0, (TODAY - d).days)


# ── Overture/OSM feature extraction ──────────────────────────────────────────

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
    if isinstance(x, (list, tuple)):
        return len(x)
    return 0


def extract_osm_features(meta: dict, db_row: dict) -> dict:
    """Extract Overture/OSM features from a DB row + its metadata_json dict."""
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
    has_address = _has(db_row.get("address"))
    num_websites = _count(websites)
    num_phones = _count(phones)
    num_socials = _count(socials)

    # Sources → num_sources, mean_confidence, days_since_last_update
    num_sources = len(sources) if isinstance(sources, list) else 0
    confs = []
    min_days = 9999
    for s in (sources if isinstance(sources, list) else []):
        if isinstance(s, dict):
            if s.get("confidence") is not None:
                try:
                    confs.append(float(s["confidence"]))
                except (ValueError, TypeError):
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

    confidence = float(meta.get("confidence") or db_row.get("confidence") or 0)
    name = str(db_row.get("name") or "")
    category = str(db_row.get("category") or "").lower()

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


# ── In-memory spatial index ───────────────────────────────────────────────────

RADIUS_METERS = 50  # match radius

# Earth radius in metres — used for BallTree haversine distance
_EARTH_R = 6_371_000.0


def load_db_spatial_index(conn) -> tuple:
    """
    Load (place_id, name, category, address, lat, lon) for every place that has
    a geometry, then build a sklearn BallTree for fast radius searches.

    Returns:
        db_df      — DataFrame with columns [place_id, name, category, address, lat, lon]
        ball_tree  — BallTree fitted on (lat_rad, lon_rad)
    """
    from sklearn.neighbors import BallTree

    print("  Loading DB spatial index into memory...", flush=True)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT place_id, name, category, address,
                   ST_Y(geom) AS lat, ST_X(geom) AS lon
            FROM places
            WHERE geom IS NOT NULL
        """)
        rows = cur.fetchall()

    db_df = pd.DataFrame(rows, columns=["place_id", "name", "category", "address", "lat", "lon"])
    print(f"    Loaded {len(db_df):,} records with geometry.")

    coords_rad = np.radians(db_df[["lat", "lon"]].values)
    ball_tree = BallTree(coords_rad, metric="haversine")

    return db_df, ball_tree


def find_candidates_bulk(
    db_df: pd.DataFrame,
    ball_tree,
    lic_lats: np.ndarray,
    lic_lons: np.ndarray,
) -> list[list[int]]:
    """
    For each (lat, lon) pair, return list of db_df row indices within RADIUS_METERS.
    All license points queried in one vectorised BallTree call.
    """
    query_rad = np.radians(np.column_stack([lic_lats, lic_lons]))
    radius_rad = RADIUS_METERS / _EARTH_R
    indices = ball_tree.query_radius(query_rad, r=radius_rad)
    return indices  # array of arrays of db_df integer positions


def fetch_metadata_bulk(conn, place_ids: list[str]) -> dict[str, dict]:
    """
    Fetch metadata_json for a list of place_ids in a single query.
    Returns dict: place_id → metadata dict.
    """
    if not place_ids:
        return {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT place_id, metadata_json FROM places WHERE place_id = ANY(%s)",
            (place_ids,),
        )
        result = {}
        for r in cur.fetchall():
            meta = r["metadata_json"] or {}
            if not isinstance(meta, dict):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            result[r["place_id"]] = meta
    return result


def compute_match_confidence(
    lic_name: str,
    lic_addr: str,
    lic_category: str,
    db_name: str,
    db_addr: str,
    db_category: str,
) -> tuple[float, float, float, bool]:
    """
    Returns (match_confidence, name_score, address_score, category_match).
    match_confidence is a combined 0–100 score.
    """
    name_score = float(fuzz.token_sort_ratio(
        (lic_name or "").lower(), (db_name or "").lower()
    ))
    address_score = float(fuzz.token_sort_ratio(
        normalize_address(lic_addr), normalize_address(db_addr)
    ))
    category_match = (
        (lic_category or "").lower().strip() == (db_category or "").lower().strip()
    )

    # Weighted combination: name 50%, address 40%, category 10%
    combined = name_score * 0.5 + address_score * 0.4 + (10 if category_match else 0)
    return combined, name_score, address_score, category_match


NAME_THRESHOLD = 80   # minimum rapidfuzz score to consider a name match


# ── Build one training row ────────────────────────────────────────────────────

def build_row(lic_row: pd.Series, match: dict) -> dict:
    db_row = match["db_row"]
    meta = match["meta"]

    # Support both schema A (golden: dates in reason) and schema B (raw: explicit date cols)
    if "end_date" in lic_row.index and not pd.isna(lic_row.get("end_date", float("nan"))):
        end_date = parse_date(lic_row.get("end_date"))
        loc_end_date = parse_date(lic_row.get("location_end_date"))
        start_date = parse_date(lic_row.get("start_date"))
    else:
        # Golden dataset: extract from reason field
        end_date, loc_end_date, start_date = parse_dates_from_reason(lic_row.get("reason"))

    osm_feats = extract_osm_features(meta, db_row)

    row = {
        # License-derived features
        "has_end_date": int(end_date is not None),
        "has_location_end_date": int(loc_end_date is not None),
        "end_date_days_ago": days_ago(end_date),
        "location_end_date_days_ago": days_ago(loc_end_date),
        "license_age_days": days_ago(start_date),
        "name_match_score": match["name_match_score"],
        "address_match_score": match["address_match_score"],
        "category_match": match["category_match"],
        # OSM/Overture features
        **osm_feats,
        # Label
        "label": 0 if str(lic_row.get("ground_truth", "open")).lower() == "closed" else 1,
        # Metadata
        "license_id": lic_row.get("license_id", ""),
        "license_source": lic_row.get("source", ""),
        "license_name": lic_row.get("name", ""),
        "license_address": get_address_field(lic_row),
        "db_place_id": db_row.get("place_id", ""),
        "db_name": db_row.get("name", ""),
        "db_address": db_row.get("address", ""),
        "match_confidence": match["match_confidence"],
    }
    return row


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build labeled training set from business license CSVs + DB."
    )
    parser.add_argument(
        "--licenses", nargs="+",
        default=[os.path.join(PROJECT_ROOT, "scripts", "data", "golden", "golden_dataset.csv")],
        help="Paths to license/golden CSV files (default: scripts/data/golden/golden_dataset.csv)"
    )
    parser.add_argument(
        "--output", default=os.path.join(PROJECT_ROOT, "scripts", "data", "training_set.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://postgres:postgres123@localhost:5432/stillopen",
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max license rows to process per file (for testing)"
    )
    args = parser.parse_args()

    db_url = args.db_url.replace("postgresql+psycopg2://", "postgresql://")
    print(f"\nStillOpen — build_training_set.py")
    print(f"  DB        : {db_url.split('@')[-1] if '@' in db_url else db_url}")
    print(f"  Output    : {args.output}")
    print(f"  Licenses  : {args.licenses}\n")

    conn = psycopg2.connect(db_url)

    # ── Build in-memory spatial index ONCE for all license files ─────────────
    db_df, ball_tree = load_db_spatial_index(conn)

    all_rows = []
    total_licenses = 0
    total_matched = 0

    for csv_path in args.licenses:
        if not os.path.exists(csv_path):
            print(f"  WARN: {csv_path} not found — skipping")
            continue

        df = pd.read_csv(csv_path, low_memory=False, dtype={"license_id": str})
        if args.limit:
            df = df.head(args.limit)

        # Validate required columns (accept 'address' or 'street_address')
        required = {"name", "ground_truth"}
        has_address_col = "address" in df.columns or "street_address" in df.columns
        missing = required - set(df.columns)
        if missing or not has_address_col:
            if not has_address_col:
                missing = missing | {"address or street_address"}
            print(f"  WARN: {csv_path} missing columns {missing} — skipping")
            continue

        df = df.reset_index(drop=True)
        df_geo = df.dropna(subset=["latitude", "longitude"]).copy()
        n_no_latlon = len(df) - len(df_geo)
        print(f"\n  Processing {os.path.basename(csv_path)}  ({len(df):,} rows, {len(df_geo):,} with lat/lon)")
        if n_no_latlon:
            print(f"    Skipping {n_no_latlon:,} rows with missing lat/lon")

        # ── Vectorised BallTree radius search ─────────────────────────────────
        lats = df_geo["latitude"].astype(float).values
        lons = df_geo["longitude"].astype(float).values

        print("  Searching for nearby DB places (BallTree)...", flush=True)
        candidate_indices = find_candidates_bulk(db_df, ball_tree, lats, lons)

        # ── Collect unique place_ids that need metadata fetched ───────────────
        needed_place_ids = set()
        for idxs in candidate_indices:
            for i in idxs:
                needed_place_ids.add(db_df.iloc[i]["place_id"])

        print(f"  Fetching metadata for {len(needed_place_ids):,} candidate places...", flush=True)
        meta_cache = fetch_metadata_bulk(conn, list(needed_place_ids))

        # ── Fuzzy match each license row against its BallTree candidates ──────
        matched = 0
        print("  Fuzzy matching...", flush=True)
        for lic_pos, (_, lic_row) in enumerate(
            tqdm(df_geo.iterrows(), total=len(df_geo), desc="  fuzzy match")
        ):
            cand_idxs = candidate_indices[lic_pos]
            if len(cand_idxs) == 0:
                continue

            best = None
            best_conf = 0.0

            for db_pos in cand_idxs:
                db_rec = db_df.iloc[db_pos]
                place_id = db_rec["place_id"]
                meta = meta_cache.get(place_id, {})

                combined, name_score, addr_score, cat_match = compute_match_confidence(
                    lic_name=str(lic_row.get("name") or ""),
                    lic_addr=get_address_field(lic_row),
                    lic_category=str(lic_row.get("category") or ""),
                    db_name=str(db_rec.get("name") or ""),
                    db_addr=str(db_rec.get("address") or ""),
                    db_category=str(db_rec.get("category") or ""),
                )
                if name_score < NAME_THRESHOLD:
                    continue
                if combined > best_conf:
                    best_conf = combined
                    best = {
                        "db_row": db_rec.to_dict(),
                        "meta": meta,
                        "match_confidence": combined,
                        "name_match_score": name_score,
                        "address_match_score": addr_score,
                        "category_match": int(cat_match),
                    }

            if best is None:
                continue
            row = build_row(lic_row, best)
            all_rows.append(row)
            matched += 1

        print(f"    Matched: {matched:,} / {len(df_geo):,} ({matched/max(1,len(df_geo))*100:.1f}%)")
        total_licenses += len(df)
        total_matched += matched

    conn.close()

    if not all_rows:
        print("\nERROR: No matches found. Check lat/lon columns and DB connectivity.")
        sys.exit(1)

    result_df = pd.DataFrame(all_rows)

    # Reorder columns to put features first, metadata last
    feature_cols = [c for c in LICENSE_FEATURE_COLS if c in result_df.columns]
    extra_cols = [c for c in result_df.columns if c not in feature_cols]
    result_df = result_df[feature_cols + extra_cols]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_csv(args.output, index=False)

    n_open = (result_df["label"] == 1).sum()
    n_closed = (result_df["label"] == 0).sum()

    print(f"\n{'='*60}")
    print(f"BUILD TRAINING SET — COMPLETE")
    print(f"{'='*60}")
    print(f"  Total licenses processed : {total_licenses:,}")
    print(f"  Total matched            : {total_matched:,} ({total_matched/max(1,total_licenses)*100:.1f}%)")
    print(f"  Open  (label=1)          : {n_open:,}")
    print(f"  Closed (label=0)         : {n_closed:,}")
    print(f"  Output                   : {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
