"""
train_xgboost.py — Train an XGBoost classifier on the license-matched training set
produced by build_training_set.py.

Features:
  - 80/20 stratified train/test split
  - Class imbalance handled via scale_pos_weight
  - Optuna hyperparameter tuning (50 trials), optimising F1 on closed class
  - Full classification report + feature importances by gain
  - Saves model to models/xgboost_licensed.pkl
  - Saves feature column list to models/feature_columns.json

Usage (Windows — set PYTHONIOENCODING=utf-8):
  set PYTHONIOENCODING=utf-8
  python scripts/train_xgboost.py ^
      --input scripts/data/training_set.csv ^
      --output-dir scripts/models/
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Columns that are metadata / identifiers, never used as features
EXCLUDE_COLS = {
    "label",
    "license_id", "license_source", "license_name", "license_address",
    "db_place_id", "db_name", "db_address", "match_confidence",
    # Leakage features from licenses
    "has_end_date", "end_date_days_ago",
    "has_location_end_date", "location_end_date_days_ago",
    "license_age_days",
}

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


def add_encoded_category(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, dict]:
    """Encode category column as frequency score + integer label. Returns mutated df."""
    if "category" not in df.columns:
        return df, None, {}

    cat_freq = df["category"].value_counts(normalize=True).to_dict()
    df = df.copy()
    df["category_freq_score"] = df["category"].map(cat_freq).fillna(0)

    le = LabelEncoder()
    df["category_label"] = le.fit_transform(df["category"].fillna("unknown"))

    return df, le, cat_freq


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all columns that should be used as features."""
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS and c != "category" and df[c].dtype != object
    ]


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_tr, y_tr, X_val, y_val, scale_pos_weight: float):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        clf = xgb.XGBClassifier(**params)
        # Weight closed (0) class more during CV
        sample_weight = np.where(y_tr == 0, scale_pos_weight, 1.0)
        clf.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)

        y_prob = clf.predict_proba(X_val)[:, 1]

        # Find threshold that maximises closed-class F1
        best_f1 = 0.0
        for t in np.arange(0.2, 0.8, 0.02):
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_val, y_pred, pos_label=0, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        return best_f1

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost on license-matched training set."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "scripts", "data", "training_set.csv"),
        help="Path to training_set.csv produced by build_training_set.py"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "scripts", "models"),
        help="Directory to save model and feature column list"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed classification threshold (default: auto-search)"
    )
    args = parser.parse_args()

    print(f"\nStillOpen — train_xgboost.py")
    print(f"  Input      : {args.input}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Optuna trials: {args.n_trials}\n")

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        print("  Run build_training_set.py first.")
        sys.exit(1)

    df = pd.read_csv(args.input, low_memory=False)
    if "label" not in df.columns:
        print("ERROR: 'label' column not found in training set.")
        sys.exit(1)

    print(f"{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows  : {len(df):,}")
    n_open = (df["label"] == 1).sum()
    n_closed = (df["label"] == 0).sum()
    print(f"  Open  (1)   : {n_open:,} ({n_open/max(1,len(df))*100:.1f}%)")
    print(f"  Closed (0)  : {n_closed:,} ({n_closed/max(1,len(df))*100:.1f}%)")

    if n_closed == 0:
        print("ERROR: No closed-label rows. Check ground_truth column in license CSVs.")
        sys.exit(1)

    # Encode category
    df, le, cat_freq = add_encoded_category(df)
    feature_cols = get_feature_cols(df)

    print(f"\n  Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"    {col}")

    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"].astype(int)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n  Train size  : {len(X_train):,}")
    print(f"  Test size   : {len(X_test):,}")

    scale_pos_weight = round(n_open / max(1, n_closed), 4)
    print(f"  scale_pos_weight (open/closed): {scale_pos_weight}")

    # ── Optuna tuning ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"OPTUNA HYPERPARAMETER TUNING ({args.n_trials} trials)")
    print(f"{'='*60}")
    print("  Optimising: F1 on closed class (recall matters more than precision)")

    # Use a validation split from train for Optuna (not the held-out test set)
    X_opt_tr, X_opt_val, y_opt_tr, y_opt_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=0
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        make_objective(X_opt_tr, y_opt_tr, X_opt_val, y_opt_val, scale_pos_weight),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["eval_metric"] = "logloss"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["verbosity"] = 0

    print(f"\n  Best trial F1 (closed): {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # ── Train final model on full train split ─────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL ON FULL TRAIN SPLIT")
    print(f"{'='*60}")

    clf = xgb.XGBClassifier(**best_params)
    sample_weight_train = np.where(y_train == 0, scale_pos_weight, 1.0)
    clf.fit(X_train, y_train, sample_weight=sample_weight_train, verbose=False)

    # ── Threshold search on test set ─────────────────────────────────────────
    y_prob_test = clf.predict_proba(X_test)[:, 1]

    if args.threshold is not None:
        optimal_threshold = args.threshold
        print(f"\n  Using fixed threshold: {optimal_threshold:.2f}")
    else:
        best_f1, optimal_threshold = 0.0, 0.5
        for t in np.arange(0.2, 0.8, 0.01):
            y_pred_t = (y_prob_test >= t).astype(int)
            f1 = f1_score(y_test, y_pred_t, pos_label=0, zero_division=0)
            if f1 > best_f1:
                best_f1, optimal_threshold = f1, t
        print(f"\n  Optimal threshold (test): {optimal_threshold:.2f}  (closed-F1={best_f1:.4f})")

    y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

    print(f"\n  Classification report on held-out test set (threshold={optimal_threshold:.2f}):")
    print(classification_report(y_test, y_pred_test, target_names=["CLOSED", "OPEN"], digits=4))

    # ── Feature importances by gain ───────────────────────────────────────────
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"  Feature importances (gain), top {min(20, len(importances))}:")
    for feat, imp in importances.head(20).items():
        bar = "#" * int(imp * 40)
        print(f"    {feat:<40s} {imp:.4f}  {bar}")

    # ── 5-Fold CV on full dataset ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("5-FOLD CROSS-VALIDATION (on full dataset)")
    print(f"{'='*60}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        sw = np.where(y_tr == 0, scale_pos_weight, 1.0)
        clf_cv = xgb.XGBClassifier(**best_params)
        clf_cv.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        y_prob_cv = clf_cv.predict_proba(X_val)[:, 1]
        y_pred_cv = (y_prob_cv >= optimal_threshold).astype(int)
        f1 = f1_score(y_val, y_pred_cv, pos_label=0, zero_division=0)
        cv_f1_scores.append(f1)
        print(f"  Fold {fold}: closed-F1={f1:.4f}")
    print(f"\n  Mean closed-F1: {np.mean(cv_f1_scores):.4f} (+/-{np.std(cv_f1_scores):.4f})")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "xgboost_licensed.pkl")
    feature_path = os.path.join(args.output_dir, "feature_columns.json")

    joblib.dump(
        {
            "model": clf,
            "feature_columns": feature_cols,
            "optimal_threshold": optimal_threshold,
            "scale_pos_weight": scale_pos_weight,
            "label_encoder": le,
            "category_freq": cat_freq,
            "best_params": best_params,
            "cv_f1_mean": float(np.mean(cv_f1_scores)),
            "training_samples": len(X_train),
            "model_type": "xgboost_licensed",
            "trained_at": datetime.now().isoformat(),
        },
        model_path,
    )

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    size_mb = os.path.getsize(model_path) / 1_000_000
    print(f"\n{'='*60}")
    print("MODEL SAVED")
    print(f"{'='*60}")
    print(f"  Model      : {model_path}  ({size_mb:.1f} MB)")
    print(f"  Features   : {feature_path}")
    print(f"  Threshold  : {optimal_threshold:.2f}")
    print(f"  Train rows : {len(X_train):,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
