---
description: How to run the StillOpen data ingestion pipeline
---

The StillOpen data ingestion pipeline is a modular system that populates the `places` table from multiple sources (OpenStreetMap, OpenAddresses, and generic CSVs).

### Prerequisites

1.  **PostGIS**: Ensure your PostgreSQL database has the PostGIS extension enabled.
2.  **osm2pgsql (Optional)**: If importing from OSM, you must first load your OSM PBF data into the `planet_osm_point` and `planet_osm_polygon` tables using `osm2pgsql`.

### Configuration

Ensure your `.env` file contains the correct `DATABASE_URL`:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/stillopen
```

### Running the Pipeline

The master orchestrator is `backend/scripts/pipeline.py`.

#### 1. Full Run
To run all stages (OSM, OpenAddresses, Generic CSVs, and Enrichment):
// turbo
```bash
python backend/scripts/pipeline.py
```

#### 2. Selective Run
To run only specific sources (e.g., only OSM and Enrichment):
// turbo
```bash
python backend/scripts/pipeline.py --sources osm
```

#### 3. Test Run (Dry/Limited)
To test the OSM stage with a limited number of rows:
// turbo
```bash
python backend/scripts/pipeline.py --sources osm --osm-limit 1000
```

### Adding New CSV Datasets

1.  Place your CSV file in `backend/data/csv/`.
2.  Create a JSON configuration file with the **same name** (e.g., `my_data.csv` and `my_data.json`).
3.  Define the column mapping in the JSON file. See `backend/data/csv/example_config.json` for a template.
4.  Run the pipeline with `--sources csv`.

### Stages Breakdown

1.  **Stage 1: OSM**: Fetches POIs from `planet_osm_*` tables.
2.  **Stage 2: OpenAddresses**: Loads CSVs from `backend/data/openaddresses/` into a staging table and enriches existing `places` by geospatial proximity.
3.  **Stage 3: Generic CSVs**: Loads custom datasets from `backend/data/csv/`.
4.  **Stage 4: Enrichment**: SQL-level cleaning, address assembly, and category backfilling.

### Verifying Results

You can run the ingestion unit tests to verify the utility functions:
// turbo
```bash
cd backend && pytest tests/test_ingestion.py -v
```
