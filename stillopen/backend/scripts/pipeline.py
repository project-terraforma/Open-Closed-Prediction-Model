#!/usr/bin/env python
"""
pipeline.py — Master orchestrator for the StillOpen ingestion pipeline.

Runs all data sources in order:
  1. OSM (from planet_osm_* tables)
  2. OpenAddresses CSVs (from --oa-dir directory)
  3. Generic CSVs (from --csv-dir directory, reads *.json + *.csv pairs)
  4. Post-ingestion enrichment pass

Usage:
    # Full pipeline (all sources, auto-discover CSVs):
    python scripts/pipeline.py

    # OSM only (dry run limited to 1000 rows):
    python scripts/pipeline.py --sources osm --osm-limit 1000

    # Everything with vacuum at the end:
    python scripts/pipeline.py --vacuum

    # Skip enrichment:
    python scripts/pipeline.py --skip-enrich

Options:
    --sources       Comma-separated list: osm,openaddresses,csv (default: all)
    --osm-limit     Max rows per OSM table (for testing)
    --osm-table     point|polygon|both (default: both)
    --oa-dir        Directory containing OpenAddresses CSVs (default: data/openaddresses/)
    --oa-radius     Match radius in metres for OpenAddresses (default: 25)
    --csv-dir       Directory containing generic CSV + JSON config pairs (default: data/csv/)
    --skip-enrich   Skip the post-ingestion enrichment pass
    --vacuum        Run VACUUM ANALYZE after the pipeline
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_osm(limit=None, table="both"):
    from ingest_osm import main as _main
    import ingest_osm
    import sys as _sys

    argv = []
    if limit:
        argv += ["--limit", str(limit)]
    if table:
        argv += ["--source-table", table]

    orig = _sys.argv
    _sys.argv = ["ingest_osm.py"] + argv
    try:
        _main()
    finally:
        _sys.argv = orig


def run_openaddresses(oa_dir: str, radius: int = 25):
    import glob
    from ingest_openaddresses import main as _main
    import sys as _sys

    csv_files = glob.glob(os.path.join(oa_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        logger.warning("No CSV files found in %s — skipping OpenAddresses", oa_dir)
        return

    for csv_path in csv_files:
        logger.info("OpenAddresses: %s", csv_path)
        orig = _sys.argv
        _sys.argv = ["ingest_openaddresses.py", csv_path, "--radius", str(radius)]
        try:
            _main()
        finally:
            _sys.argv = orig


def run_generic_csvs(csv_dir: str):
    import glob
    from ingest_csv_generic import ingest, _load_config
    import sys as _sys

    # Pair each *.csv with a *.json config of the same stem name
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    for csv_path in csv_files:
        stem = os.path.splitext(csv_path)[0]
        config_path = stem + ".json"
        if not os.path.isfile(config_path):
            logger.warning("No config found for %s — skipping", csv_path)
            continue
        logger.info("Generic CSV: %s (config: %s)", csv_path, config_path)
        config = _load_config(config_path)
        total = ingest(csv_path, config)
        logger.info("  ✓ %d records from %s", total, csv_path)


def run_enrich(vacuum: bool = False):
    from enrich_metadata import main as _main
    import sys as _sys

    orig = _sys.argv
    args = ["enrich_metadata.py"]
    if vacuum:
        args.append("--vacuum")
    _sys.argv = args
    try:
        _main()
    finally:
        _sys.argv = orig


def main():
    parser = argparse.ArgumentParser(description="StillOpen full ingestion pipeline")
    parser.add_argument("--sources", default="osm,openaddresses,csv",
                        help="Comma-separated list of sources to run (default: osm,openaddresses,csv)")
    parser.add_argument("--osm-limit", type=int, default=None, help="OSM rows limit (for testing)")
    parser.add_argument("--osm-table", default="both", choices=["point", "polygon", "both"])
    parser.add_argument("--oa-dir", default=os.path.join(BACKEND_DIR, "data", "openaddresses"),
                        help="Directory of OpenAddresses CSVs")
    parser.add_argument("--oa-radius", type=int, default=25, help="Match radius in metres")
    parser.add_argument("--csv-dir", default=os.path.join(BACKEND_DIR, "data", "csv"),
                        help="Directory of generic CSV+JSON pairs")
    parser.add_argument("--skip-enrich", action="store_true", help="Skip enrichment pass")
    parser.add_argument("--vacuum", action="store_true", help="VACUUM ANALYZE at the end")
    args = parser.parse_args()

    sources = {s.strip().lower() for s in args.sources.split(",")}
    logger.info("Pipeline sources: %s", sources)

    if "osm" in sources:
        logger.info("=== Stage 1: OSM ===")
        run_osm(limit=args.osm_limit, table=args.osm_table)

    if "openaddresses" in sources:
        logger.info("=== Stage 2: OpenAddresses ===")
        run_openaddresses(args.oa_dir, args.oa_radius)

    if "csv" in sources:
        logger.info("=== Stage 3: Generic CSVs ===")
        run_generic_csvs(args.csv_dir)

    if not args.skip_enrich:
        logger.info("=== Stage 4: Enrichment ===")
        run_enrich(vacuum=args.vacuum)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
