"""
ingest_csv_generic.py — Ingest any arbitrary business/POI CSV dataset
into the `places` table.

This is the catch-all script for datasets that don't fit the OSM or
OpenAddresses schemas (e.g. government business registries, Yelp open
datasets, SafeGraph, Overture Maps parquet, etc.).

The script maps configurable CSV column names to standard fields and
upserts into `places`. It is intentionally generic so new sources can
be added by writing a small JSON config file — no Python changes needed.

Usage:
    python scripts/ingest_csv_generic.py config.json data.csv [--limit N]

Config file format (JSON):
{
    "source_name": "ca_business_registry",
    "id_col":      "BusinessId",          // unique ID column
    "name_col":    "BusinessName",
    "lat_col":     "Latitude",
    "lon_col":     "Longitude",
    "category_col": "BusinessType",       // optional
    "address_col":  "FullAddress",        // optional pre-built address
    "street_col":   "Street",             // OR separate components
    "city_col":     "City",
    "state_col":    "State",
    "zip_col":      "ZipCode",
    "phone_col":    "Phone",
    "website_col":  "Website",
    "hours_col":    "HoursOfOperation",
    "extra_cols": {                       // arbitrary extra metadata keys
        "LicenseType": "license_type",
        "Status":      "business_status"
    }
}
"""

import argparse
import csv
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ingest_utils import (
    get_conn,
    build_address,
    normalize_phone,
    normalize_url,
    batch_upsert,
    normalize_category,
    logger as _base,
)

logger = logging.getLogger("ingest.csv_generic")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _get(row: dict, col: str | None, default="") -> str:
    if not col:
        return default
    return (row.get(col) or "").strip()


def _row_to_place(row: dict, cfg: dict) -> dict | None:
    """Map a CSV row → places row dict using `cfg` column mapping."""
    try:
        lat = float(_get(row, cfg.get("lat_col")) or 0)
        lon = float(_get(row, cfg.get("lon_col")) or 0)
    except ValueError:
        return None
    if lat == 0.0 or lon == 0.0:
        return None

    name = _get(row, cfg.get("name_col"))
    if not name:
        return None

    source_name = cfg.get("source_name", "external")
    raw_id = _get(row, cfg.get("id_col")) or f"{lat:.6f},{lon:.6f}"
    place_id = f"{source_name}_{raw_id}"

    # Address — prefer pre-built col, else assemble from parts
    address_parts: dict = {}
    if cfg.get("address_col"):
        address_parts["full_address"] = _get(row, cfg["address_col"])
    else:
        address_parts["addr:street"] = _get(row, cfg.get("street_col"))
        address_parts["addr:city"]   = _get(row, cfg.get("city_col"))
        address_parts["addr:state"]  = _get(row, cfg.get("state_col"))
        address_parts["addr:postcode"] = _get(row, cfg.get("zip_col"))

    address = build_address(address_parts)

    phone   = normalize_phone(_get(row, cfg.get("phone_col")))
    website = normalize_url(_get(row, cfg.get("website_col")))
    opening_hours = _get(row, cfg.get("hours_col")) or None

    raw_cat = _get(row, cfg.get("category_col"))
    category = normalize_category({raw_cat: raw_cat}) if raw_cat else None
    if not category and raw_cat:
        category = raw_cat.lower().replace("_", " ")[:60]

    metadata: dict = dict(address_parts)
    if address:
        metadata["address"] = address
    if phone:
        metadata["phone"] = phone
    if website:
        metadata["website"] = website
    if opening_hours:
        metadata["opening_hours"] = opening_hours

    # Extra custom columns
    for src_col, dest_key in (cfg.get("extra_cols") or {}).items():
        val = _get(row, src_col)
        if val:
            metadata[dest_key] = val

    return {
        "place_id":     place_id,
        "name":         name,
        "category":     category,
        "source":       source_name,
        "lat":          lat,
        "lon":          lon,
        "metadata_json": metadata,
    }


def ingest(csv_path: str, config: dict, limit: int | None = None) -> int:
    conn = get_conn()
    places: list[dict] = []
    skipped = 0

    logger.info("Reading %s …", csv_path)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            p = _row_to_place(row, config)
            if p:
                places.append(p)
            else:
                skipped += 1

    logger.info("Parsed %d rows (%d skipped — missing lat/lon/name)", len(places), skipped)
    total = batch_upsert(conn, places)
    conn.close()
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest generic CSV business data → places")
    parser.add_argument("config", help="Path to JSON column-mapping config file")
    parser.add_argument("csv_path", help="Path to the CSV data file")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process (testing)")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    if not os.path.isfile(args.csv_path):
        logger.error("CSV file not found: %s", args.csv_path)
        sys.exit(1)

    config = _load_config(args.config)
    total = ingest(args.csv_path, config, args.limit)
    logger.info("Generic CSV ingestion complete. Upserted %d records.", total)


if __name__ == "__main__":
    main()
