"""
enrich_metadata.py — Post-ingestion enrichment pass.

After primary ingestion (OSM / OpenAddresses / generic CSVs), this script
runs SQL-level passes to:

  1. Denormalize address fields → `metadata_json.address` for any rows that
     still have components but no assembled address string.
  2. Normalize phone numbers in-place.
  3. Add `https://` prefix to bare website URLs.
  4. Backfill `category` column from amenity/shop tags in metadata_json.
  5. Optionally run a VACUUM ANALYZE on the places table.

Usage:
    python scripts/enrich_metadata.py [--vacuum]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ingest_utils import get_conn, logger as _base

logger = logging.getLogger("ingest.enrich")

# ---------------------------------------------------------------------------
# SQL passes
# ---------------------------------------------------------------------------

# 1. assemble address where components exist but address string is blank/missing
SQL_ASSEMBLE_ADDRESS = """
UPDATE places
SET metadata_json = metadata_json || jsonb_build_object(
    'address', concat_ws(', ',
        CASE
            WHEN (metadata_json->>'addr:housenumber') IS NOT NULL
             AND (metadata_json->>'addr:street') IS NOT NULL
            THEN (metadata_json->>'addr:housenumber') || ' ' || (metadata_json->>'addr:street')
            ELSE metadata_json->>'addr:street'
        END,
        metadata_json->>'addr:city',
        metadata_json->>'addr:state',
        metadata_json->>'addr:postcode'
    )
)
WHERE
    (metadata_json->>'address' IS NULL OR metadata_json->>'address' = '')
    AND (
        metadata_json->>'addr:street' IS NOT NULL
        OR metadata_json->>'addr:city'  IS NOT NULL
    );
"""

# 2. Fix website URLs missing scheme
SQL_FIX_WEBSITE = r"""
UPDATE places
SET metadata_json = metadata_json || jsonb_build_object(
    'website', 'https://' || (metadata_json->>'website')
)
WHERE
    metadata_json ? 'website'
    AND metadata_json->>'website' NOT LIKE 'http%'
    AND metadata_json->>'website' != '';
"""

# 3. Backfill category column from metadata tags (if category IS NULL)
SQL_BACKFILL_CATEGORY = """
UPDATE places
SET category = COALESCE(
    metadata_json->>'amenity',
    metadata_json->>'shop',
    metadata_json->>'office',
    metadata_json->>'tourism',
    metadata_json->>'healthcare',
    metadata_json->>'leisure'
)
WHERE
    category IS NULL
    AND (
        metadata_json ? 'amenity'    OR
        metadata_json ? 'shop'       OR
        metadata_json ? 'office'     OR
        metadata_json ? 'tourism'    OR
        metadata_json ? 'healthcare' OR
        metadata_json ? 'leisure'
    );
"""

# 4. Also store the category in metadata_json for consistent frontend access
SQL_SYNC_CATEGORY_TO_META = """
UPDATE places
SET metadata_json = metadata_json || jsonb_build_object('category', category)
WHERE
    category IS NOT NULL
    AND (metadata_json->>'category' IS NULL OR metadata_json->>'category' = '');
"""

# 5. Remove null/empty string values from metadata_json (DB tidy-up)
SQL_STRIP_NULLS = """
UPDATE places
SET metadata_json = (
    SELECT jsonb_object_agg(k, v)
    FROM jsonb_each(metadata_json) AS t(k, v)
    WHERE v::text NOT IN ('null', '""', '')
)
WHERE metadata_json::text LIKE '%null%'
   OR metadata_json::text LIKE '%"""%';
"""


def run_pass(conn, name: str, sql: str) -> int:
    logger.info("Running: %s …", name)
    with conn.cursor() as cur:
        cur.execute(sql)
        affected = cur.rowcount
        conn.commit()
    logger.info("  ✓ %d rows updated", affected)
    return affected


def main():
    parser = argparse.ArgumentParser(description="Post-ingestion metadata enrichment")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM ANALYZE after passes")
    parser.add_argument("--skip-nulls", action="store_true", help="Skip the null-strip pass (slow on large tables)")
    args = parser.parse_args()

    conn = get_conn()

    run_pass(conn, "Assemble address strings", SQL_ASSEMBLE_ADDRESS)
    run_pass(conn, "Fix website URLs",         SQL_FIX_WEBSITE)
    run_pass(conn, "Backfill category column", SQL_BACKFILL_CATEGORY)
    run_pass(conn, "Sync category → metadata", SQL_SYNC_CATEGORY_TO_META)

    if not args.skip_nulls:
        run_pass(conn, "Strip null metadata values", SQL_STRIP_NULLS)

    if args.vacuum:
        conn.autocommit = True
        logger.info("Running VACUUM ANALYZE on places …")
        with conn.cursor() as cur:
            cur.execute("VACUUM ANALYZE places;")
        logger.info("  ✓ Done")

    conn.close()
    logger.info("Enrichment pass complete.")


if __name__ == "__main__":
    main()
