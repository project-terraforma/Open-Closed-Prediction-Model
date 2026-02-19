import joblib
import pandas as pd
import os
from .features import compute_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "open_model.pkl")

class ModelService:
    def __init__(self):
        self.model = None
        self.artifacts = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            loaded = joblib.load(MODEL_PATH)
            if isinstance(loaded, dict):
                self.model = loaded['model']
                self.artifacts = loaded
            else:
                self.model = loaded
                self.artifacts = {}
        else:
            print(f"Model not found at {MODEL_PATH}! Predictions will be mocks.")
            self.model = None

    def predict(self, place_data: dict):
        # Feature Engineering
        features_dict = compute_features(place_data, self.artifacts)
        
        status = "unknown"
        confidence = 0.0
        
        if not self.model:
            # Mock prediction
            status = "open"
            confidence = 0.82
        else:
            feature_cols = self.artifacts.get('features', [])
            
            # DataFrame for prediction
            df_input = pd.DataFrame([features_dict])
            
            # Ensure columns order
            if feature_cols:
                # Add missing cols with 0
                for c in feature_cols:
                    if c not in df_input.columns:
                        df_input[c] = 0
                df_input = df_input[feature_cols]

            prediction_cls = self.model.predict(df_input)[0] 
            # Check if predict_proba available
            if hasattr(self.model, "predict_proba"):
                try:
                    prediction_prob = self.model.predict_proba(df_input)[0][1] 
                except:
                    prediction_prob = float(prediction_cls)
            else:
                prediction_prob = float(prediction_cls)
            
            status = "open" if prediction_cls == 1 else "closed"
            confidence = float(prediction_prob)

        # Generate explanation
        explanation = []
        if status == "open":
            explanation.append("Model predicts this place is likely open.")
        else:
            explanation.append("Model predicts this place is likely closed.")
            
        if features_dict.get('has_website'):
            explanation.append("Website is active.")
        if features_dict.get('has_social'):
            explanation.append("Social media presence detected.")
        if features_dict.get('days_since_last_update', 999) < 30:
            explanation.append("Recent data updates found.")
        
        return {
            "status": status,
            "confidence": confidence,
            "explanation": explanation
        }

model_service = ModelService()

def predict_status(place) -> dict:
    # If a SQLAlchemy Place object is passed, extract metadata_json. Otherwise, use it as a dict.
    place_data = place.metadata_json if hasattr(place, 'metadata_json') else place
    if not isinstance(place_data, dict):
        place_data = {}
        
    return model_service.predict(place_data)
