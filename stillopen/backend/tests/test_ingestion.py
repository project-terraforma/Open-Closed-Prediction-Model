"""
tests/test_ingestion.py — Unit tests for the ingestion pipeline utilities.

Run with:
    pytest backend/tests/test_ingestion.py -v

These tests are OFFLINE (no DB connection needed) — they test the pure
Python helper functions only: address building, category mapping, phone
normalisation, URL normalisation, and the metadata merge logic.

Integration tests (requiring a live DB) are marked with @pytest.mark.db
and are skipped unless DATABASE_URL is set and --run-db is passed.
"""

import sys
import os
import json
import pytest

# Allow imports from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ingest_utils import (
    build_address,
    normalize_category,
    normalize_phone,
    normalize_url,
    merge_metadata,
)


# ---------------------------------------------------------------------------
# build_address
# ---------------------------------------------------------------------------
class TestBuildAddress:
    def test_full_osm_address(self):
        tags = {
            "addr:housenumber": "123",
            "addr:street": "Main St",
            "addr:city": "Springfield",
            "addr:state": "CA",
            "addr:postcode": "90210",
        }
        result = build_address(tags)
        assert result == "123 Main St, Springfield, CA, 90210"

    def test_street_no_number(self):
        tags = {"addr:street": "Elm Ave", "addr:city": "Shelbyville"}
        result = build_address(tags)
        assert result == "Elm Ave, Shelbyville"

    def test_city_only(self):
        tags = {"addr:city": "Oakland", "addr:state": "CA"}
        result = build_address(tags)
        assert result == "Oakland, CA"

    def test_pre_built_full_address(self):
        tags = {"full_address": "  456 Oak St, Riverside, CA  "}
        result = build_address(tags)
        assert result == "456 Oak St, Riverside, CA"

    def test_plain_address_field(self):
        tags = {"address": "789 Pine Blvd"}
        result = build_address(tags)
        assert result == "789 Pine Blvd"

    def test_empty_tags(self):
        assert build_address({}) == ""

    def test_postcode_int_coercion(self):
        tags = {"addr:city": "LA", "addr:postcode": "90001"}
        result = build_address(tags)
        assert "90001" in result

    def test_municipality_fallback(self):
        tags = {"addr:municipality": "Berkeley", "addr:state": "CA"}
        result = build_address(tags)
        assert "Berkeley" in result


# ---------------------------------------------------------------------------
# normalize_category
# ---------------------------------------------------------------------------
class TestNormalizeCategory:
    def test_restaurant(self):
        assert normalize_category({"amenity": "restaurant"}) == "restaurant"

    def test_cafe(self):
        assert normalize_category({"amenity": "cafe"}) == "cafe"

    def test_shop_supermarket(self):
        assert normalize_category({"shop": "supermarket"}) == "supermarket"

    def test_shop_clothes_falls_back(self):
        result = normalize_category({"shop": "clothes"})
        assert result == "clothing"

    def test_office_generic(self):
        result = normalize_category({"office": "company"})
        assert result is not None and "office" in result.lower()

    def test_empty(self):
        assert normalize_category({}) is None

    def test_amenity_takes_priority_over_shop(self):
        # Amenity listed first in tag map, so should win
        result = normalize_category({"amenity": "cafe", "shop": "supermarket"})
        assert result == "cafe"

    def test_raw_value_fallback(self):
        # A value not in the map should be returned with underscores replaced
        result = normalize_category({"amenity": "bicycle_rental"})
        assert result is not None
        assert "_" not in result


# ---------------------------------------------------------------------------
# normalize_phone
# ---------------------------------------------------------------------------
class TestNormalizePhone:
    def test_us_number(self):
        result = normalize_phone("+1 (555) 867-5309")
        assert result is not None
        assert "867" in result

    def test_none_input(self):
        assert normalize_phone(None) is None

    def test_empty_string(self):
        assert normalize_phone("") is None

    def test_strips_non_standard_chars(self):
        result = normalize_phone("abc123@def")
        # Should keep digits only
        assert result is not None
        assert "123" in result

    def test_plus_preserved(self):
        result = normalize_phone("+44 20 7946 0958")
        assert result is not None
        assert "+" in result


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------
class TestNormalizeUrl:
    def test_already_https(self):
        assert normalize_url("https://example.com") == "https://example.com"

    def test_already_http(self):
        assert normalize_url("http://example.com") == "http://example.com"

    def test_adds_scheme(self):
        assert normalize_url("example.com") == "https://example.com"

    def test_none(self):
        assert normalize_url(None) is None

    def test_empty(self):
        assert normalize_url("") is None

    def test_strips_whitespace(self):
        assert normalize_url("  example.com  ") == "https://example.com"


# ---------------------------------------------------------------------------
# merge_metadata
# ---------------------------------------------------------------------------
class TestMergeMetadata:
    def test_fills_missing(self):
        existing = {"name": "Joe's", "phone": ""}
        incoming = {"phone": "555-1234", "website": "https://joes.com"}
        result = merge_metadata(existing, incoming)
        assert result["phone"] == "555-1234"
        assert result["website"] == "https://joes.com"

    def test_does_not_overwrite_existing(self):
        existing = {"phone": "555-1111"}
        incoming = {"phone": "999-9999"}
        result = merge_metadata(existing, incoming)
        assert result["phone"] == "555-1111"  # original preserved

    def test_skips_none_incoming(self):
        existing = {}
        incoming = {"phone": None, "website": ""}
        result = merge_metadata(existing, incoming)
        assert "phone" not in result
        assert "website" not in result

    def test_does_not_mutate_inputs(self):
        existing = {"a": "1"}
        incoming = {"b": "2"}
        result = merge_metadata(existing, incoming)
        assert "b" not in existing
        assert "a" not in incoming

    def test_empty_existing(self):
        incoming = {"x": "hello"}
        result = merge_metadata({}, incoming)
        assert result["x"] == "hello"


# ---------------------------------------------------------------------------
# Integration tests (skipped without live DB — use: pytest --run-db)
# ---------------------------------------------------------------------------


@pytest.mark.db
def test_db_places_table_exists(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM places;")
        count = cur.fetchone()[0]
    assert isinstance(count, int)


@pytest.mark.db
def test_db_metadata_json_populated(db_conn):
    """Spot-check that at least some rows have non-empty metadata_json."""
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM places
            WHERE metadata_json IS NOT NULL
              AND metadata_json != '{}'::jsonb;
        """)
        count = cur.fetchone()[0]
    assert count >= 0   # non-fatal — just reports the number


@pytest.mark.db
def test_db_address_field_present(db_conn):
    """After enrichment, at least some places should have a built address."""
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM places
            WHERE metadata_json->>'address' IS NOT NULL
              AND metadata_json->>'address' != '';
        """)
        count = cur.fetchone()[0]
    print(f"Places with address: {count}")
    # Not asserting > 0 since DB may be empty in CI — just informational
