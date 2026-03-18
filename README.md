# StillOpen — Business Status Prediction Platform

A geospatial web application that predicts whether real-world businesses and points of interest are currently open or permanently closed, built on 1.4 million California place records sourced from Overture Maps.

---

## Project Summary

The project started as a machine learning experiment to classify business operational status using metadata signals from the Overture Maps dataset. It grew into a full-stack platform with a PostgreSQL/PostGIS database, a FastAPI REST API, and a Next.js frontend for interactive search and browsing.

The prediction approach went through several iterations. Early work used Random Forest and then XGBoost models trained on small synthetic or OSM-derived datasets. A signal-based deterministic scorer was built as an interim replacement, but it produced unreliable results in practice — it was too sensitive to surface-level metadata patterns and lacked any grounding in real ground-truth labels. The project ultimately returned to XGBoost, this time trained on a properly labeled dataset derived from public business license records.

The current model is an XGBoost classifier trained on approximately 8,500 labeled records built by cross-referencing San Francisco, Los Angeles, and San Diego business license CSVs against the Overture database. Open/closed labels come directly from license expiration and end-date fields in the official city datasets, giving the model real ground-truth supervision. Hyperparameters are tuned with Optuna over 50 trials, optimizing F1 on the closed class to counteract class imbalance.

---

## Repository Structure

```
Open-Closed-Prediction-Model-Emilio-Michael/
|
|-- stillopen/                        # Full-stack application
|   |-- backend/                      # FastAPI + SQLAlchemy + PostGIS
|   |   |-- app/
|   |   |   |-- main.py               # API routes, Overpass/Nominatim integration
|   |   |   |-- predict.py            # Prediction entry points (wraps XGBoost model)
|   |   |   |-- scorer.py             # Signal-based scorer (legacy, no longer primary)
|   |   |   |-- search.py             # Full-text and geospatial search logic
|   |   |   |-- features.py           # Feature extraction from place metadata
|   |   |   |-- models.py             # SQLAlchemy ORM models
|   |   |   |-- database.py           # DB session and connection setup
|   |   |   |-- categories.py         # Category normalization
|   |   |   `-- utils.py              # Reverse geocoding, shared utilities
|   |   |-- scripts/
|   |   |   |-- ingest_overture.py    # Overture Maps ingestion pipeline
|   |   |   |-- ingest_osm.py         # OpenStreetMap ingestion
|   |   |   |-- ingest_openaddresses.py
|   |   |   |-- enrich_metadata.py    # Metadata enrichment pass
|   |   |   |-- reverse_geocode_addresses.py
|   |   |   `-- pipeline.py           # End-to-end ingestion orchestration
|   |   |-- utils/
|   |   |   `-- canonical_metadata.py # Metadata normalization helpers
|   |   |-- tests/
|   |   |   |-- test_search.py
|   |   |   |-- test_ingestion.py
|   |   |   `-- test_overture.py
|   |   |-- model/
|   |   |   `-- open_model.pkl        # Serialized Random Forest model (early prototype)
|   |   |-- requirements.txt
|   |   |-- .env                      # Active DB connection (Supabase or local)
|   |   `-- .env.supabase             # Supabase connection backup
|   |
|   `-- frontend/                     # Next.js 14 application
|       |-- src/
|       |   |-- app/
|       |   |   |-- page.tsx          # Homepage / search entry
|       |   |   |-- search/page.tsx   # Search results page
|       |   |   |-- browse/page.tsx   # Browse by city/category
|       |   |   `-- place/[id]/page.tsx  # Individual place detail page
|       |   |-- components/
|       |   |   |-- SearchBar.tsx
|       |   |   |-- SearchResults.tsx
|       |   |   |-- ResultCard.tsx
|       |   |   |-- ResultsMap.tsx    # Leaflet map integration
|       |   |   |-- StatusBadge.tsx   # Open/Closed/At-Risk badge
|       |   |   |-- CitySearchResults.tsx
|       |   |   |-- PaginationBar.tsx
|       |   |   |-- Navbar.tsx
|       |   |   |-- Breadcrumbs.tsx
|       |   |   |-- Footer.tsx
|       |   |   `-- LoadingView.tsx
|       |   `-- lib/
|       |       |-- api.ts            # Backend API client
|       |       |-- AppContext.tsx    # Global state
|       |       |-- CitySearchService.ts
|       |       `-- formatters.ts
|       |-- .env.local                # NEXT_PUBLIC_API_URL
|       `-- package.json
|
|-- scripts/                          # Data pipeline and model scripts
|   |-- fetch_california_overture.py  # Downloads Overture parquet for California
|   |-- fetch_osm_california.py       # Downloads OSM data for California
|   |-- fetch_golden_data.py          # Fetches SF/LA/SD business license data
|   |-- fetch_wikidata_ca.py          # Wikidata enrichment fetch
|   |-- seed_postgres.py              # Seeds local PostgreSQL from parquet
|   |
|   |-- build_golden_dataset.py       # Combines SF/LA/SD license CSVs into golden_dataset.csv
|   |-- build_training_set.py         # Cross-references license data against DB to build training_set.csv
|   |-- train_xgboost.py              # Trains XGBoost model on training_set.csv (ACTIVE)
|   |-- apply_predictions.py          # Applies model to all DB records in batch
|   |
|   |-- apply_golden_labels.py        # Writes golden labels into the database
|   |-- verify_businesses.py          # Live OSM/web verification of individual businesses
|   |-- validate_scorer.py            # Evaluates predictions against golden labels
|   |-- model_comparison.py           # Compares model variants
|   |-- check_leakage.py              # Feature leakage detection
|   |-- detect_conflation.py          # Detects duplicate/conflated records
|   |-- enrich_osm_addresses.py       # Adds OSM address data to records
|   |-- train_from_db.py              # Alternative: retrain from DB-verified website records
|   |-- apply_predictions_xgboost_backup.py  # Backup scoring path
|   |-- stress_test.py / stress_test_2.py    # API load testing
|   |
|   |-- data/
|   |   |-- golden/
|   |   |   |-- sf_businesses.csv     # SF Open Data business licenses
|   |   |   |-- la_businesses.csv     # LA Open Data business licenses
|   |   |   |-- sd_businesses.csv     # SD Business Tax Certificate data
|   |   |   `-- golden_dataset.csv    # Combined, deduplicated ground-truth labels
|   |   |-- training_set.csv          # Feature-extracted labeled set (~8,500 records)
|   |   `-- overture_santa_cruz.parquet  # Cached Overture parquet
|   |
|   `-- models/
|       |-- xgboost_licensed.pkl      # Trained XGBoost model (ACTIVE)
|       `-- feature_columns.json      # Feature schema used by xgboost_licensed.pkl
|
|-- start.bat                         # Windows launcher (backend + frontend)
|-- start.sh                          # Unix launcher
`-- README.md
```

---

## Architecture

### Database

PostgreSQL 18 with the PostGIS extension. The primary table is `places`, which stores Overture place records with PostGIS geometry, a JSONB metadata column, and three added columns for prediction output: `predicted_status`, `prediction_confidence`, and `prediction_updated_at`. A `city_cache` table accelerates browse-by-city queries. Indexes include a GIST spatial index, a GIN full-text search index, and B-tree indexes on category and city metadata.

Two deployment targets are supported: a local PostgreSQL instance (1.4M records) and a Supabase-hosted instance (395K records). The active connection is controlled by `stillopen/backend/.env`.

### Backend

FastAPI application serving REST endpoints for search, browse, and place detail. Search supports full-text queries, geospatial radius queries, and category filtering. The API integrates with OpenStreetMap Overpass and Nominatim for live data enrichment and reverse geocoding.

Predictions are applied offline in batch via `scripts/apply_predictions.py` rather than at query time. API responses return pre-computed status and confidence values from the database.

### Prediction Model

The active model is an XGBoost classifier stored at `scripts/models/xgboost_licensed.pkl`.

**Training data** comes from public business license records across three California cities: San Francisco (SF Open Data), Los Angeles (LA Open Data), and San Diego (SD Business Tax Certificates). These are combined and deduplicated into `scripts/data/golden/golden_dataset.csv`. The open/closed label for each record is derived from license end dates and expiration fields — businesses with a lapsed end date are labeled closed, active licenses are labeled open. This produces approximately 8,500 labeled records with real ground-truth supervision.

**Feature extraction** is performed in `build_training_set.py`, which fuzzy-matches each license record against the Overture database using PostGIS proximity (within ~50 metres) and name similarity via rapidfuzz. Matched Overture records provide the feature set: contact info presence, source count and confidence, data recency, category signals, and derived ratios.

**Training** is handled by `train_xgboost.py`, which runs an 80/20 stratified split, uses Optuna for hyperparameter tuning over 50 trials, and optimizes F1 on the closed class to handle class imbalance. The optimal classification threshold is also searched on the held-out test set rather than defaulting to 0.5.

**Why not the signal scorer**: A deterministic signal-based scorer (`scorer.py`) was built and tested as an alternative, but it produced inconsistent results. Without labeled supervision it could not reliably distinguish genuinely closed businesses from businesses that simply had sparse metadata. The license-backed XGBoost model outperformed it on the validated ground-truth set and is the active approach.

### Frontend

Next.js 14 application with Tailwind CSS. Pages cover homepage search, search results with map view (Leaflet), city/category browsing, and individual place detail. The `StatusBadge` component renders open, closed, and at-risk states. The frontend connects to the backend API via the URL configured in `.env.local`.

---

## Data

- Place records: 1,452,268 California Overture records in local PostgreSQL
- Labeled training set: ~8,500 records from SF, LA, and SD business license data
- Ground-truth source: city license expiration dates (open/closed labels)
- Parquet cache: `scripts/data/overture_santa_cruz.parquet`

---

## Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 18 with PostGIS extension

### Backend

```bash
cd stillopen/backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install geopy requests     # not yet in requirements.txt
uvicorn app.main:app --reload --port 8000
```

Configure `stillopen/backend/.env` with your database URL:

```
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/stillopen
```

### Frontend

```bash
cd stillopen/frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:3000` and expects the backend at `http://localhost:8000`.

### Quick Start (Windows)

```bash
start.bat
```

### Rebuilding the Model

```bash
# 1. Fetch license data from city open data APIs
set PYTHONIOENCODING=utf-8
python scripts/fetch_golden_data.py

# 2. Combine and deduplicate into a single labeled set
python scripts/build_golden_dataset.py

# 3. Cross-reference against Overture DB to extract features
python scripts/build_training_set.py \
    --licenses scripts/data/golden/golden_dataset.csv \
    --output scripts/data/training_set.csv \
    --db-url postgresql://postgres:postgres123@localhost:5432/stillopen

# 4. Train XGBoost with Optuna tuning
set PYTHONIOENCODING=utf-8
python scripts/train_xgboost.py \
    --input scripts/data/training_set.csv \
    --output-dir scripts/models/

# 5. Apply predictions to all DB records
python scripts/apply_predictions.py --local --batch-size 50000
```

---

## Model Development History

| Phase | Approach | Training Data | Status |
|---|---|---|---|
| Initial exploration | Random Forest | Small synthetic set | Replaced |
| XGBoost v1 | XGBoost + feature engineering | OSM + website-verified records | Replaced |
| Signal scorer | Deterministic rule-based engine | No labeled data required | Abandoned — poor real-world performance |
| XGBoost v2 (current) | XGBoost + Optuna tuning | ~8,500 business license records (SF/LA/SD) | Active |

The signal scorer approach (`scorer.py`) remains in the codebase for reference but is not used in the active prediction pipeline.
