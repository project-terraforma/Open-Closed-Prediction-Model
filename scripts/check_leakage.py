import joblib
import pandas as pd

try:
    df = pd.read_csv('scripts/data/training_set.csv', low_memory=False)
    
    print("\n============== Correlation with Label (0=closed, 1=open) ==============")
    corrs = df.corr(numeric_only=True)['label'].sort_values()
    print("\nTop negative correlations:")
    print(corrs.head(10))
    print("\nTop positive correlations:")
    print(corrs.tail(10))

    model_data = joblib.load('scripts/models/xgboost_licensed.pkl')
    clf = model_data['model']
    cols = model_data['feature_columns']
    importances = sorted(zip(cols, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    
    print("\n============== Feature Importances (Gain) ==============")
    for f, imp in importances[:15]:
        print(f"{f:<30} {imp:.4f}")
        
except Exception as e:
    print("Error:", e)
