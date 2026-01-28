# Still Open? Open & Closed Place Prediction

A machine learning project designed to predict whether real-world places (businesses, points of interest) are currently open or closed based on their digital metadata. The project also includes modules for business category classification and data confidence scoring.

## Project Overview

In the world of geospatial data, keeping "open/closed" status accurate is a major challenge. This project leverages scalable features from place metadata—such as the presence of social media links, the number of data sources, and the recency of updates—to build a robust prediction model.

### Key Capabilities:
1.  **Open/Closed Prediction**: Binary classification to determine if a place is operational.
2.  **Category Classification**: Predicting business categories (e.g., "Restaurant", "Retail") from business names.
3.  **Confidence Scoring**: Predicting the reliability of a dataset record based on metadata completeness.

---

## Model Pipeline (Cleaning & Training)

### 1. Data Cleaning & Parsing
The raw data (often in Parquet or JSON formats) contains nested structures and "stringified" objects. Our pipeline:
*   **Safe Struct Parsing**: Uses `ast.literal_eval` and `json.loads` to safely convert stringified metadata into Python objects.
*   **Presence Validation**: Converts complex metadata (lists of websites, phones, emails) into binary "presence" features (1 if exists, 0 if not).
*   **Handling Nulls**: Implements robust checks to treat empty strings, `None`, and empty lists/arrays as missing values.

### 2. Feature Engineering
We transform raw attributes into signals that indicate business activity:
*   **Metadata Completeness**: Calculates a score based on how many contact points (website, phone, socials) are available.
*   **Source Analysis**: 
    *   `num_sources`: Total data providers confirming the location.
    *   `source_mean_confidence`: Average confidence score across all sources.
    *   `days_since_last_update`: Recency of the latest data refresh relative to a reference date.
*   **Category Context**: 
    *   **Frequency Encoding**: Maps categories to their relative popularity to handle high cardinality.
    *   **Label Encoding**: Provides numerical mappings for tree-based models.

### 3. Training & Evaluation
We employ a multi-model approach to ensure performance and interpretability:
*   **Random Forest Classifier**: Chosen for its robustness and ability to provide feature importance rankings.
*   **Gradient Boosting (GBM)**: Used to capture non-linear relationships and improve accuracy on complex edge cases.
*   **Metrics**: Models are evaluated using **ROC-AUC** (for class separation), **F1-Score** (to balance precision/recall), and **Confusion Matrices**.

---

## Repository Structure

```text
├── data/                   # Raw & Sample data (CSV, JSON, Parquet)
├── src/                    # Source code
|   ├── debug_features.py   # Debugging script for feature engineering
│   ├── train_open_model.py # Primary binary classification (Open/Closed)
│   ├── train_model.py      # Category NLP and Confidence Regression
│   └── read_parquet.py     # Data utility script
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd Open-Closed-Prediction-Model-Emilio-Michael
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model**:
   ```bash
   python src/train_open_model.py
   ```

---

## Scalability Note
For large-scale production (e.g., >100M records), the logic is designed to be portable to **PySpark** or specialized gradient boosting libraries like **XGBoost/LightGBM** which offer superior parallelization and speed.
