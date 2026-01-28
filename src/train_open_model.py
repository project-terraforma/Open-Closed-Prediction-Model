import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONSTANTS & CONFIG
# ---------------------------------------------------------
REFERENCE_DATE = datetime(2025, 2, 24) # Approximate date from the sample data viewed earlier

# ---------------------------------------------------------
# DATA LOADING & UTILS
# ---------------------------------------------------------

def safe_parse_struct(x):
    """Safely parses a stringified python/json struct or returns the object if already valid."""
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        # Cleaning numpy-style array string representation
        # e.g. "array(['foo', 'bar'], dtype=object)" -> "['foo', 'bar']"
        if 'array(' in x:
            x = x.replace("array(", "").replace(", dtype=object)", "").replace(")", "")
        
        try:
            return json.loads(x.replace("'", '"').replace("None", "null"))
        except:
            try:
                return ast.literal_eval(x)
            except:
                # If it looks like a list/dict but failed parsing, return None to be safe, 
                # or return the string if simple detection is needed.
                return None
    return None

def has_value(x):
    """Robust presence check"""
    if x is None: return 0
    if isinstance(x, (int, float)): return 1
    if isinstance(x, (list, dict, np.ndarray)):
        return 1 if len(x) > 0 else 0
    
    s = str(x).strip()
    if s.lower() in ['none', 'null', 'nan', '', '[]', '{}']:
        return 0
    return 1

def preprocess_data(df):
    """Performs feature engineering on the raw dataframe."""
    print("Feature Engineering started...")
    
    # 1. Target Variable Check
    if 'open' not in df.columns:
        raise ValueError("Target column 'open' not found in dataset")
    
    # Drops rows where target is NaN (shouldn't happen in ground truth, but good safety)
    df = df.dropna(subset=['open']).copy()
    
    # ---------------------------------------------------------
    # FEATURE: METADATA COMPLETENESS (Boolean/Counts)
    # ---------------------------------------------------------
    df['has_website'] = df['websites'].apply(has_value)
    df['has_social'] = df['socials'].apply(has_value)
    df['has_phone'] = df['phones'].apply(has_value)
    df['has_address'] = df['addresses'].apply(has_value)
    df['has_email'] = df['emails'].apply(has_value)
    df['has_brand'] = df['brand'].apply(has_value)

    # ---------------------------------------------------------
    # FEATURE: SOURCES ANALYSIS
    # ---------------------------------------------------------
    def analyze_sources(x):
        sources = safe_parse_struct(x)
        if not sources or not isinstance(sources, list):
            return pd.Series([0, 0, 9999]) # num_sources, mean_conf, days_since_update
        
        count = len(sources)
        confidences = [s.get('confidence', 0) for s in sources if isinstance(s, dict)]
        mean_conf = np.mean(confidences) if confidences else 0
        
        # Update Time recency
        # Try to parse timestamps. Format seen: '2025-02-24T08:00:00.000Z'
        min_days = 9999
        for s in sources:
            if isinstance(s, dict) and 'update_time' in s:
                try:
                    ts_str = s['update_time']
                    # Simple truncation for iso parsing if needed
                    if 'T' in ts_str:
                        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        # Remove timezone for simple diff
                        dt = dt.replace(tzinfo=None)
                        days = (REFERENCE_DATE - dt).days
                        if days < min_days:
                            min_days = days
                except:
                    pass
        
        return pd.Series([count, mean_conf, min_days])

    print("  Processing 'sources' column (this may take a moment)...")
    source_stats = df['sources'].apply(analyze_sources)
    source_stats.columns = ['num_sources', 'source_mean_confidence', 'days_since_last_update']
    
    # Merge back
    df = pd.concat([df, source_stats], axis=1)

    # Filter outlier dates (e.g. 9999) or fill
    # If 9999, it means no date found. We'll map this to a high number or mean
    mean_days = df[df['days_since_last_update'] != 9999]['days_since_last_update'].mean()
    df['days_since_last_update'] = df['days_since_last_update'].replace(9999, mean_days).fillna(mean_days)

    # ---------------------------------------------------------
    # FEATURE: CATEGORY
    # ---------------------------------------------------------
    def get_primary_category(x):
        cat = safe_parse_struct(x)
        if isinstance(cat, dict):
            return cat.get('primary', 'unknown')
        return 'unknown'

    df['primary_category'] = df['categories'].apply(get_primary_category)
    
    # Encode Categories (Frequency Encoding to handle cardinality/scalability)
    # For very large datasets, OneHot is bad. Target encoding or Freq encoding is better.
    freq_map = df['primary_category'].value_counts(normalize=True).to_dict()
    df['category_freq_score'] = df['primary_category'].map(freq_map)
    
    # Also add a simple numeric Label encoder for tree models
    le = LabelEncoder()
    # Fit on top N categories, others 'other'
    top_cats = df['primary_category'].value_counts().head(50).index
    df['category_label'] = df['primary_category'].apply(lambda c: c if c in top_cats else 'other')
    df['category_label'] = le.fit_transform(df['category_label'])

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------
    # Select feature columns
    feature_cols = [
        'confidence', 
        'has_website', 'has_social', 'has_phone', 'has_address', 'has_email', 'has_brand',
        'num_sources', 'source_mean_confidence', 'days_since_last_update',
        'category_freq_score', 'category_label'
    ]
    
    # Fill remaining NaNs (e.g. from original confidence column)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    return df, feature_cols

# ---------------------------------------------------------
# MODELING
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy:  {model.score(X_test, y_test):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return y_pred

def run_models(df, feature_cols):
    X = df[feature_cols]
    y = df['open'].astype(int)
    
    print(f"\ndataset split: Total samples {len(X)}")
    print(f"Class Balance (Target 'open'):\n{y.value_counts(normalize=True)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Random Forest (Robust baseline)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_test, y_test, "Random Forest")
    
    # 2. Gradient Boosting (Often higher accuracy)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    evaluate_model(gb, X_test, y_test, "Gradient Boosting")
    
    return rf, gb, feature_cols

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Loading parquet data...")
    try:
        df = pd.read_parquet("data/project_c_samples.parquet")
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return

    # Process
    processed_df, features = preprocess_data(df)
    
    # Train
    rf_model, gb_model, feature_names = run_models(processed_df, features)
    
    # Feature Importance (RF)
    print("\n--- Feature Importance (Random Forest) ---")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(len(features)):
        print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")

    # Scalability note
    print("\n--- Scalability & Production Notes ---")
    print("1. Data Volume: For >100M records, migrating this logic to PySpark is recommended.")
    print("2. JSON Parsing: The parsing of 'sources' is CPU intensive. In production, pre-compute these features during ETL.")
    print("3. Model Choice: XGBoost/LightGBM are preferred over sklearn's GradientBoosting for speed and parallelization.")

if __name__ == "__main__":
    main()
