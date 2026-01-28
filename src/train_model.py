import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def extract_primary_name(name_struct):
    if name_struct is None:
        return ""
    # If it's a dictionary (from parquet)
    if isinstance(name_struct, dict):
        return name_struct.get('primary', '') or ''
    # If it's a string (e.g. from bad load), try to parse (fallback)
    if isinstance(name_struct, str):
        try:
            d = eval(name_struct) # Simple eval for stringified dicts
            return d.get('primary', '') or ''
        except:
            return str(name_struct)
    return str(name_struct)

def extract_primary_category(cat_struct):
    if cat_struct is None:
        return None
    if isinstance(cat_struct, dict):
        return cat_struct.get('primary', None)
    if isinstance(cat_struct, str):
        try:
            d = eval(cat_struct)
            return d.get('primary', None)
        except:
            return None
    return None

def main():
    print("Loading data from project_c_samples.parquet...")
    try:
        df = pd.read_parquet("data/project_c_samples.parquet")
    except Exception as e:
        print(f"Failed to load parquet: {e}")
        return

    print(f"Dataset Shape: {df.shape}")
    
    # --- Preprocessing for Classification (Name -> Category) ---
    print("\n--- Task 1: Classify Business Category from Name ---")
    
    # Extract features
    df['clean_name'] = df['names'].apply(extract_primary_name)
    df['target_category'] = df['categories'].apply(extract_primary_category)
    
    # Filter valid rows
    classification_df = df[
        (df['clean_name'].str.len() > 1) & 
        (df['target_category'].notna())
    ].copy()
    
    # Filter to top N categories to ensure enough samples per class
    top_categories = classification_df['target_category'].value_counts().head(10).index.tolist()
    classification_df = classification_df[classification_df['target_category'].isin(top_categories)]
    
    print(f"Training on {len(classification_df)} samples for {len(top_categories)} top categories.")
    print(f"Categories: {', '.join(top_categories)}")

    X_text = classification_df['clean_name']
    y_class = classification_df['target_category']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_text, y_class, test_size=0.2, random_state=42)

    # Simple Text Classification Pipeline
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(X_train_c, y_train_c)
    predictions_c = text_clf.predict(X_test_c)
    
    print("\nClassification Report (Name -> Category):")
    print(classification_report(y_test_c, predictions_c))
    
    # Example predictions
    examples = ["Starbucks", "Main Street Auto Repair", "Hilton Hotel", "Dr. Smith Dentist", "Fresh Market"]
    print("\nExample Inference:")
    preds = text_clf.predict(examples)
    for txt, pred in zip(examples, preds):
        print(f"  '{txt}' -> Predicted: {pred}")

    # --- Preprocessing for Regression (Metadata -> Confidence) ---
    print("\n--- Task 2: Predict Confidence Score from Metadata ---")
    
    # Create simple features based on presence of data
    # Note: parquet columns might be none or empty lists/arrays
    def has_prop(x):
        if x is None: return 0
        if isinstance(x, (list, np.ndarray)) and len(x) > 0: return 1
        if isinstance(x, str) and len(x) > 2: return 1 # for stringified lists
        return 0

    df['has_website'] = df['websites'].apply(has_prop)
    df['has_social'] = df['socials'].apply(has_prop)
    df['has_phone'] = df['phones'].apply(has_prop)
    df['has_address'] = df['addresses'].apply(has_prop)
    
    # Use these binary features to predict confidence
    X_reg = df[['has_website', 'has_social', 'has_phone', 'has_address']]
    y_reg = df['confidence']
    
    # Drop NaNs in target if any
    mask = y_reg.notna()
    X_reg = X_reg[mask]
    y_reg = y_reg[mask]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    reg_model.fit(X_train_r, y_train_r)
    
    y_pred_r = reg_model.predict(X_test_r)
    
    print(f"\nRegression MSE: {mean_squared_error(y_test_r, y_pred_r):.6f}")
    print(f"Regression R2 Score: {r2_score(y_test_r, y_pred_r):.4f}")
    print("Feature Importances:")
    for name, imp in zip(X_reg.columns, reg_model.feature_importances_):
        print(f"  {name}: {imp:.4f}")

if __name__ == "__main__":
    main()
