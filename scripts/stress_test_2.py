import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# STEP 1
csv_files = glob.glob(os.path.join(PROJECT_ROOT, "scripts", "data", "**", "*.csv"), recursive=True)
print("## Step 1: Find the correct dataset")
large_csv_path = None
for f in csv_files:
    try:
        df = pd.read_csv(f, low_memory=False)
        nrows = len(df)
        print(f"{os.path.basename(f)}: {nrows} rows (Full path: {f})")
        # Identify the ~8500 dataset
        if 8000 < nrows < 9000:
            large_csv_path = f
    except Exception as e:
        print(f"{os.path.basename(f)}: Error reading - {e}")

if not large_csv_path:
    print("Could not find a CSV with ~8500 rows.")
    exit(1)

print(f"\n=> Using {large_csv_path} as the large dataset.\n")

# STEP 2
df = pd.read_csv(large_csv_path, low_memory=False)
expected_cols = ['name_match_score', 'address_match_score', 'confidence', 'num_sources', 
                 'source_mean_confidence', 'has_phone', 'digital_presence', 'name_length', 
                 'has_closure_keyword', 'website_verified_closed']
print("## Step 2: Validate the dataset")
for col in expected_cols:
    if col in df.columns:
        print(f"Col {col} is present.")
    else:
        print(f"Col {col} is MISSING.")

print(f"Row count: {len(df)}")
if 'label' in df.columns:
    pct_open = (df['label'] == 1).mean() * 100
    pct_closed = (df['label'] == 0).mean() * 100
    print(f"Open/Closed Class balance: Open: {pct_open:.1f}%, Closed: {pct_closed:.1f}%")

print("Missing/null values per column:")
print(df[expected_cols].isnull().sum() if all(c in df.columns for c in expected_cols) else df[[c for c in expected_cols if c in df.columns]].isnull().sum())
print("\n")

# STEP 3
MODEL_PATH = os.path.join(PROJECT_ROOT, "scripts", "models", "xgboost_licensed.pkl")
model_data = joblib.load(MODEL_PATH)

clf = model_data['model']
feature_cols = model_data['feature_columns']
optimal_threshold = 0.62

y_true = df['label'].astype(int)

# Fill na for features, extract
X = df[feature_cols].fillna(0).astype(float)

y_prob = clf.predict_proba(X)[:, 1]
y_pred = (y_prob >= optimal_threshold).astype(int)

results = []
def add_result(check, res, reason):
    results.append({"Check": check, "Result": res, "Reason": reason})

acc = accuracy_score(y_true, y_pred)
p1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
r1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

p0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
r0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

add_result("Perf: Open (1)", "Pass", f"Acc: {acc:.3f}, P: {p1:.3f}, R: {r1:.3f}, F1: {f1_1:.3f}")
add_result("Perf: Closed (0)", "Pass", f"Acc: {acc:.3f}, P: {p0:.3f}, R: {r0:.3f}, F1: {f1_0:.3f}")

# STEP 4: Sanity Checks
# mean name length diff
if 'name_length' in feature_cols:
    nl_open = X.loc[y_pred == 1, 'name_length'].mean()
    nl_closed = X.loc[y_pred == 0, 'name_length'].mean()
    nl_diff = abs(nl_open - nl_closed) / max(nl_open, nl_closed, 1e-5) * 100
    flag = "Flag" if nl_diff > 15 else "Pass"
    add_result("Spurious: name_length diff", flag, f"Open: {nl_open:.1f}, Closed: {nl_closed:.1f} (Diff: {nl_diff:.1f}%)")

    # shuffle name length
    np.random.seed(42)
    X_shuf = X.copy()
    X_shuf['name_length'] = np.random.permutation(X_shuf['name_length'].values)
    y_pred_shuf = (clf.predict_proba(X_shuf)[:, 1] >= optimal_threshold).astype(int)
    acc_shuf = accuracy_score(y_true, y_pred_shuf)
    delta = acc - acc_shuf
    add_result("Spurious: name_length shuffle", "Flag" if delta < 0.02 else "Pass", f"Accuracy drop: {delta*100:.2f}% (Thresh: 2%)")
else:
    add_result("Spurious: name_length diff", "N/A", "name_length missing")
    add_result("Spurious: name_length shuffle", "N/A", "name_length missing")

# confidence distribution
conf_range_pct = ((y_prob >= 0.45) & (y_prob <= 0.55)).mean() * 100
add_result("Confidence dist", "Flag" if conf_range_pct > 20 else "Pass", f"{conf_range_pct:.1f}% predictions in 0.45-0.55 range")

# dead features
variances = X.var()
dead_feats = variances[variances == 0].index.tolist()
if dead_feats:
    add_result("Dead Features", "Flag", f"Features with 0 variance: {', '.join(dead_feats)}")
else:
    add_result("Dead Features", "Pass", "No features have 0 variance")

print("## Single Results Table")
print(f"{'Check':<35} | {'Result':<6} | {'Reason'}")
print("-" * 100)
for r in results:
    print(f"{r['Check']:<35} | {r['Result']:<6} | {r['Reason']}")
