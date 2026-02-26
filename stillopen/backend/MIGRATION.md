# StillOpen Backend Migration: PostgreSQL & PostGIS

This document outlines the steps required to migrate the backend architecture from SQLite to PostgreSQL with PostGIS for robust geospatial search and trigram text matching.

## 1. Prerequisites (macOS)
Install PostgreSQL, PostGIS, and the OpenStreetMap import tool (`osm2pgsql`):

```bash
# Install postgres and postgis
brew install postgresql@15 postgis

# Start PostgreSQL service
brew services start postgresql@15

# Install OSM ingestion tool
brew install osm2pgsql
```

## 2. Database Creation & Setup
Create the `stillopen` database and enable the necessary extensions:

```bash
# Connect to postgres default database to create a new one
psql postgres

# Run the following SQL commands:
CREATE DATABASE stillopen;
\c stillopen

-- Enable PostGIS for geospatial data
CREATE EXTENSION IF NOT EXISTS postgis;

-- Enable trigram extension for fuzzy string matching (pg_trgm)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## 3. Environment Variables
In your `.env` or local bash environment, define the `DATABASE_URL` (usually your local system user has a passwordless peer authentication or you can configure a role).

```bash
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/stillopen"
export POSTGIS_ENABLED="true"
```

## 4. Ingesting Data (OpenStreetMap)
Download a regional extract from Geofabrik and use `osm2pgsql` to import the data:

```bash
# 1. Download California or your desired region
wget http://download.geofabrik.de/north-america/us/california-latest.osm.pbf

# 2. Ingest the data into the postgis database
# The -O flex or default can be used. This basic command uses standard schema:
osm2pgsql -d stillopen -c california-latest.osm.pbf
```

## 5. Normalized Schema Setup & Seeding
Once the OSM data is inside Postgres (typically as `planet_osm_point` and `planet_osm_polygon`), you must map it to our structured `places` table.

### 5.1 Initialize Schema
Run the following SQL migration script in `psql -d stillopen` to create the table and indices:

```sql
-- 1. Create the structured Place schema
CREATE TABLE IF NOT EXISTS places (
    id BIGSERIAL PRIMARY KEY,
    place_id TEXT UNIQUE,
    name TEXT,
    category TEXT,
    geom GEOGRAPHY(Point, 4326),
    source TEXT,
    metadata_json JSONB,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- 2. Create proper indexes for scalability
-- Create GiST index for spatial queries
CREATE INDEX idx_places_geom ON places USING GIST (geom);

-- Create trigram index for fast fuzzy searching
CREATE INDEX idx_places_name_trgm ON places USING GIN (name gin_trgm_ops);
```

### 5.2 Run Automated Ingestion Pipeline
We provide a Python-based pipeline that handles OSM, OpenAddresses, and custom CSV datasets with deduplication and metadata enrichment.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (OSM -> OpenAddresses -> Generic CSVs -> Enrichment)
python scripts/pipeline.py

# Or run specific stages
python scripts/pipeline.py --sources osm
```

See [ingest-data workflow](../.agent/workflows/ingest-data.md) for detailed usage of the ingestion scripts.

## 6. Restart Backend
Apply the new file changes (`database.py`, `models.py`, `search.py`), and start the application:

```bash
uvicorn app.main:app --reload
```
