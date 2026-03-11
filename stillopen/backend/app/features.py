import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime

REFERENCE_DATE = datetime(2025, 2, 24)

# Categories with historically high business turnover / closure rates
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

def safe_parse_struct(x):
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        if 'array(' in x:
            x = x.replace("array(", "").replace(", dtype=object)", "").replace(")", "")
        try:
            return json.loads(x.replace("'", '"').replace("None", "null"))
        except:
            try:
                return ast.literal_eval(x)
            except:
                return None
    return None

def has_value(x):
    if x is None: return 0
    if isinstance(x, (int, float)): return 1
    if isinstance(x, (list, dict, np.ndarray)):
        return 1 if len(x) > 0 else 0
    s = str(x).strip()
    if s.lower() in ['none', 'null', 'nan', '', '[]', '{}']:
        return 0
    return 1

def count_items(x):
    """Count items in a list/array field."""
    if x is None: return 0
    parsed = safe_parse_struct(x)
    if isinstance(parsed, (list, np.ndarray)):
        return len(parsed)
    return 0

def compute_features(record: dict, artifacts: dict = None) -> dict:
    """
    Compute all 25 features for a single place record.
    Compatible with the Phase 3 model (RF/GBM/XGBoost/LightGBM).
    
    Feature groups:
    - Basic presence (6): has_website/social/phone/email/brand/address
    - Counts (3): num_websites/socials/phones
    - Source analysis (3): num_sources, source_mean_confidence, days_since_last_update
    - Name features (2): name_length, has_closure_keyword
    - Category (2): category_freq_score, category_label
    - Composite (2): digital_presence, metadata_completeness
    - Interaction (3): confidence_x_sources, digital_x_confidence, sources_x_recency
    - Ratios (2): web_to_social_ratio, phone_to_web_ratio
    - Binary indicators (1): is_stale
    - Passthrough (1): confidence
    """
    features = {}
    
    # ── 1. Binary Presence ──
    # Handle both list-form ('websites') and singular-form ('website') field names,
    # as different data sources (Overture S3 vs Postgres ingest) use different schemas.
    websites = record.get('websites') or ([record['website']] if record.get('website') else None)
    phones   = record.get('phones')   or ([record['phone']]   if record.get('phone')   else None)
    socials  = record.get('socials')
    addresses = record.get('addresses')
    emails   = record.get('emails')
    brand    = record.get('brand')

    features['has_website'] = has_value(websites)
    features['has_social']  = has_value(socials)
    features['has_phone']   = has_value(phones)
    features['has_address'] = has_value(addresses)
    features['has_email']   = has_value(emails)
    features['has_brand']   = has_value(brand)

    # ── 2. Count Features ──
    features['num_websites'] = count_items(websites)
    features['num_socials']  = count_items(socials)
    features['num_phones']   = count_items(phones)
    
    # ── 3. Source Analysis ──
    sources = safe_parse_struct(record.get('sources'))
    
    num_sources = 0
    mean_conf = 0.0
    days_since_update = 365
    
    if sources and isinstance(sources, list):
        num_sources = len(sources)
        confidences = [s['confidence'] for s in sources if isinstance(s, dict) and s.get('confidence') is not None]
        if confidences:
            mean_conf = float(np.mean(confidences))
            
        min_days = 9999
        for s in sources:
            if isinstance(s, dict) and 'update_time' in s:
                try:
                    ts_str = s['update_time']
                    if 'T' in ts_str:
                        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        dt = dt.replace(tzinfo=None)
                        delta = (REFERENCE_DATE - dt).days
                        if delta < min_days:
                            min_days = delta
                except:
                    pass
        if min_days != 9999:
            days_since_update = max(0, min_days)  # Clamp to 0 minimum
            
    features['num_sources'] = num_sources
    features['source_mean_confidence'] = mean_conf
    features['days_since_last_update'] = days_since_update
    
    # ── 4. Confidence ──
    features['confidence'] = float(record.get('confidence', 0))
    
    # ── 5. Category Features ──
    primary_category = 'unknown'
    cats = safe_parse_struct(record.get('categories'))
    if isinstance(cats, dict):
        primary_category = cats.get('primary', 'unknown') or 'unknown'
        
    if artifacts:
        cat_freq = artifacts.get('category_freq', artifacts.get('freq_map', {}))
        le = artifacts.get('label_encoder', artifacts.get('le'))
        
        features['category_freq_score'] = cat_freq.get(primary_category, 0)
        
        if le:
            try:
                features['category_label'] = le.transform([primary_category])[0]
            except:
                features['category_label'] = 0 
        else:
            features['category_label'] = 0
    else:
        features['category_freq_score'] = 0
        features['category_label'] = 0
    
    # ── 6. Name-based Features ──
    name = ''
    names = record.get('names')
    if isinstance(names, dict):
        name = names.get('primary', '') or ''
    elif isinstance(names, str):
        name = names
    if not name:
        name = record.get('name', '')
    
    features['name_length'] = len(name)
    
    name_lower = name.lower()
    features['has_closure_keyword'] = 1 if any(kw in name_lower for kw in [
        'closed', 'former', 'defunct', 'out of business', 'coming soon',
        'vacant', 'empty', 'available', 'for lease', 'for rent'
    ]) else 0
    
    # ── 7. Composite Features ──
    features['digital_presence'] = (
        features['has_website'] + features['has_social'] + features['has_phone'] +
        features['has_email'] + features['has_brand']
    )
    
    features['metadata_completeness'] = (
        features['has_website'] * 0.2 + features['has_social'] * 0.15 + 
        features['has_phone'] * 0.2 + features['has_email'] * 0.1 + 
        features['has_brand'] * 0.15 + features['has_address'] * 0.1 +
        (1 if num_sources > 1 else 0) * 0.1
    )
    
    # ── 8. Interaction Features ──
    features['confidence_x_sources'] = features['confidence'] * num_sources
    features['digital_x_confidence'] = features['digital_presence'] * features['confidence']
    features['sources_x_recency'] = num_sources / max(1, days_since_update + 1)
    
    # ── 9. Ratio Features ──
    features['web_to_social_ratio'] = features['num_websites'] / (features['num_socials'] + 1)
    features['phone_to_web_ratio'] = features['num_phones'] / (features['num_websites'] + 1)
    
    # ── 10. Binary Indicators ──
    features['is_stale'] = 1 if days_since_update > 180 else 0

    # ── 11. High-Turnover Category ──
    features['high_turnover_category'] = 1 if primary_category.lower() in HIGH_TURNOVER_CATEGORIES else 0

    # ── 12. Website Verified Closed ──
    # Hard signal: website_status set by verify_businesses.py
    features['website_verified_closed'] = 1 if record.get('website_status') == 'likely_closed' else 0

    return features
