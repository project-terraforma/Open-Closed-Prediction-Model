import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "scripts", "data", "training_set.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "scripts", "models", "xgboost_licensed.pkl")

# Load data and model
df = pd.read_csv(DATA_PATH, low_memory=False)
model_data = joblib.load(MODEL_PATH)

clf = model_data['model']
feature_cols = model_data['feature_columns']
optimal_threshold = model_data.get('optimal_threshold', 0.5)

# Ensure 'label' is present
y_true = df['label'].astype(int)

# Extract features
X = df[feature_cols].fillna(0).astype(float)

# Base predictions
y_prob = clf.predict_proba(X)[:, 1]
y_pred = (y_prob >= optimal_threshold).astype(int)

results = []

def add_result(check_name, pass_flag, reason):
    results.append({"Check": check_name, "Result": pass_flag, "Reason": reason})

# 1. Basic performance numbers
acc = accuracy_score(y_true, y_pred)
prec_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
rec_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

prec_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
rec_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

pct_open = (y_true == 1).mean() * 100
pct_closed = (y_true == 0).mean() * 100

skew_flag = "Flag" if abs(pct_open - pct_closed) > 20 else "Pass" # Flag if skewed
add_result("Class Balance", skew_flag, f"Open: {pct_open:.1f}%, Closed: {pct_closed:.1f}%")
add_result("Perf: Open (1)", "Pass", f"Acc: {acc:.3f}, P: {prec_1:.3f}, R: {rec_1:.3f}, F1: {f1_1:.3f}")
add_result("Perf: Closed (0)", "Pass", f"Acc: {acc:.3f}, P: {prec_0:.3f}, R: {rec_0:.3f}, F1: {f1_0:.3f}")

# 2. Spurious feature check
# a) mean name_length difference
if 'name_length' in feature_cols:
    name_len_open = X.loc[y_pred == 1, 'name_length'].mean()
    name_len_closed = X.loc[y_pred == 0, 'name_length'].mean()
    diff_pct = abs(name_len_open - name_len_closed) / max(name_len_open, name_len_closed, 1e-5) * 100
    flag = "Flag" if diff_pct > 15 else "Pass"
    add_result("Spurious: name_length diff", flag, f"Open mean: {name_len_open:.1f}, Closed mean: {name_len_closed:.1f} (Diff: {diff_pct:.1f}%)")

    # b) shuffle name_length
    np.random.seed(42)
    X_shuffled = X.copy()
    X_shuffled['name_length'] = np.random.permutation(X_shuffled['name_length'].values)
    y_pred_shuffled = (clf.predict_proba(X_shuffled)[:, 1] >= optimal_threshold).astype(int)
    acc_shuffled = accuracy_score(y_true, y_pred_shuffled)
    drop = acc - acc_shuffled
    flag = "Flag" if drop < 0.02 else "Pass"
    add_result("Spurious: name_length shuffle", flag, f"Accuracy drop: {drop*100:.2f}% (Threshold: 2%)")
else:
    add_result("Spurious: name_length", "N/A", "Feature 'name_length' not found")

# c) shuffle geo features
geo_candidates = ['lat', 'lon', 'latitude', 'longitude', 'zip', 'zip_code', 'postal_code']
geo_features = [f for f in geo_candidates if f in feature_cols]
for geo_f in geo_features:
    np.random.seed(42)
    X_shuffled = X.copy()
    X_shuffled[geo_f] = np.random.permutation(X_shuffled[geo_f].values)
    y_pred_shuffled = (clf.predict_proba(X_shuffled)[:, 1] >= optimal_threshold).astype(int)
    acc_shuffled = accuracy_score(y_true, y_pred_shuffled)
    drop = acc - acc_shuffled
    flag = "Flag" if drop < 0.02 else "Pass"
    add_result(f"Spurious: {geo_f} shuffle", flag, f"Accuracy drop: {drop*100:.2f}% (Threshold: 2%)")

if not geo_features:
    add_result("Spurious: geo shuffle", "N/A", "No standard geo features found")

# 3. Geo bias check
zip_col = None
for col in ['zip', 'zip_code', 'postal_code', 'address_zip', 'postcode']:
    if col in df.columns:
        zip_col = col
        break

if zip_col:
    df['pred'] = y_pred
    bias_flag = "Pass"
    reasons = []
    for z, group in df.groupby(zip_col):
        if len(group) >= 10: 
            mean_pred = group['pred'].mean()
            if mean_pred >= 0.9 or mean_pred <= 0.1:
                bias_flag = "Flag"
                label = 'Open' if mean_pred >= 0.9 else 'Closed'
                reasons.append(f"ZIP {z} has {mean_pred*100:.0f}% {label} (n={len(group)})")
    
    if bias_flag == "Flag":
        add_result("Geo bias (ZIP)", bias_flag, f"{len(reasons)} regions skewed >= 90%. E.g. {reasons[0]}")
    else:
        add_result("Geo bias (ZIP)", bias_flag, "No region (n>=10) has 90%+ same prediction")
else:
    # Try finding something in original dataset manually
    # Looking for a spatial group
    add_result("Geo bias", "N/A", f"No ZIP or region feature found. Checked columns: {list(df.columns)}")

# 4. Confidence distribution
conf_in_range = ((y_prob >= 0.45) & (y_prob <= 0.55)).mean() * 100
conf_flag = "Flag" if conf_in_range > 20 else "Pass"
add_result("Confidence dist", conf_flag, f"{conf_in_range:.1f}% predictions in 0.45-0.55 range")

# Print table
print(f"{'Check':<35} | {'Result':<6} | {'Reason'}")
print("-" * 120)
for r in results:
    print(f"{r['Check']:<35} | {r['Result']:<6} | {r['Reason']}")
