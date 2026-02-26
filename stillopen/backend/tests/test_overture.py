"""
tests/test_overture.py — Unit tests for Overture ingestion logic.

Run with:
    pytest backend/tests/test_overture.py -v
"""

import sys
import os
import pytest
import pandas as pd

# Add backend dir to path for internal imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from overture_ingest.ingest_places import process_overture_row
from utils.canonical_metadata import build_canonical_metadata

def test_build_canonical_from_overture_addresses():
    raw = {
        "names": {"primary": "Test Place"},
        "addresses": [
            {
                "house_number": "1600",
                "street": "Amphitheatre Pkwy",
                "locality": "Mountain View",
                "region": "CA",
                "postcode": "94043",
                "country": "US",
            }
        ],
        "websites": ["example.com"],
        "phones": ["(650) 253-0000"],
    }
    canonical = build_canonical_metadata(raw, lat=37.422, lon=-122.084)
    assert canonical["name"] == "Test Place"
    assert "1600 Amphitheatre Pkwy" in (canonical["formatted_address"] or "")
    assert canonical["website"] == "https://example.com"
    assert canonical["international_phone_number"] == "+16502530000"
    assert canonical["geometry"]["location"]["lat"] == 37.422
    assert canonical["geometry"]["location"]["lng"] == -122.084

def test_process_overture_row():
    # Mock row as a series mirroring DuckDB/Pandas structure
    row = {
        "id": "test-id",
        "names": {"primary": "Test Place"},
        "categories": {"primary": "restaurant"},
        "addresses": [
            {
                "house_number": "123",
                "street": "Main St",
                "locality": "City",
                "region": "ST",
                "postcode": "12345",
            }
        ],
        "websites": ["test.com"],
        "phones": ["(555) 123-4567"],
        "bbox": {"minx": -122.0, "maxx": -121.0, "miny": 37.0, "maxy": 38.0},
        "sources": [],
        "socials": [],
    }
    
    result = process_overture_row(row)
    
    assert result['place_id'] == 'overture_test-id'
    assert result['name'] == 'Test Place'
    assert result['category'] == 'restaurant'
    assert result['lat'] == 37.5
    assert result['lon'] == -121.5
    assert "canonical" in result["metadata_json"]
    assert "raw" in result["metadata_json"]
    assert "123 Main St" in (result["metadata_json"]["canonical"]["formatted_address"] or "")
    assert result["metadata_json"]["canonical"]["website"] == "https://test.com"
    assert result["metadata_json"]["canonical"]["international_phone_number"] == "+15551234567"
