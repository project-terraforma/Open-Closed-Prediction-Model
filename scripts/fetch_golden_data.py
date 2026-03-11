"""
fetch_golden_data.py — Pull SF and LA business license data to build a golden dataset.

Usage:
    python scripts/fetch_golden_data.py [--sf-only] [--la-only] [--limit 50000]

Outputs:
    scripts/data/golden/sf_businesses_raw.json   (raw API response cache)
    scripts/data/golden/la_businesses_raw.json   (raw API response cache)
    scripts/data/golden/sf_businesses.csv
    scripts/data/golden/la_businesses.csv
"""

import argparse
import csv
import json
import time
import sys
from pathlib import Path
from typing import Optional

import requests

# ── Config ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "data" / "golden"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SF_API    = "https://data.sfgov.org/resource/g8m3-pdis.json"
LA_API    = "https://data.lacity.org/resource/r4uk-afju.json"

PAGE_SIZE     = 1000
REQUEST_DELAY = 0.5   # seconds between requests
MAX_RETRIES   = 3
RETRY_DELAY   = 5.0


# ── HTTP helper ────────────────────────────────────────────────────────────────

def _get(url: str, params: dict) -> list[dict]:
    """GET with retry logic. Returns parsed JSON list."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Request failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


# ── SF fetcher ─────────────────────────────────────────────────────────────────
# Real fields confirmed from live API (g8m3-pdis):
#   certificate_number, dba_name, full_business_address, city, state,
#   business_zip, dba_start_date, dba_end_date, location_start_date,
#   location_end_date, location (GeoJSON Point), ownership_name,
#   ttxid, uniqueid, parking_tax, transient_occupancy_tax,
#   data_as_of, data_loaded_at
# No separate lat/lng fields — coordinates are inside location.coordinates

SF_FIELDS = [
    "certificate_number",
    "dba_name",
    "full_business_address",
    "city",
    "state",
    "business_zip",
    "dba_start_date",
    "dba_end_date",
    "location_start_date",
    "location_end_date",
    "location",
]


def fetch_sf(limit: int) -> list[dict]:
    print(f"\n{'='*60}")
    print("FETCHING SAN FRANCISCO BUSINESS LICENSE DATA")
    print(f"Target: {limit:,} records  |  Source: {SF_API}")
    print('='*60)

    all_records: list[dict] = []
    offset = 0

    while len(all_records) < limit:
        batch_size = min(PAGE_SIZE, limit - len(all_records))
        params = {
            "$limit":  batch_size,
            "$offset": offset,
            "$select": ",".join(SF_FIELDS),
            "$order":  "certificate_number ASC",
        }

        batch = _get(SF_API, params)
        if not batch:
            print(f"  No more records at offset {offset:,}. Done.")
            break

        all_records.extend(batch)
        offset += len(batch)

        if len(all_records) % 5000 == 0 or len(batch) < batch_size:
            print(f"  Fetched {len(all_records):,} SF records so far...")

        if len(batch) < batch_size:
            break

        time.sleep(REQUEST_DELAY)

    print(f"  Total SF records fetched: {len(all_records):,}")
    return all_records


def _extract_sf_coords(location_field) -> tuple[str, str]:
    """Extract lat, lng from SF GeoJSON location field."""
    if not location_field:
        return "", ""
    if isinstance(location_field, dict):
        coords = location_field.get("coordinates", [])
        if isinstance(coords, list) and len(coords) == 2:
            return str(coords[1]), str(coords[0])  # GeoJSON is [lng, lat]
    return "", ""


def save_sf_csv(records: list[dict]) -> Path:
    """Clean SF records and save to CSV."""
    path = OUTPUT_DIR / "sf_businesses.csv"

    fieldnames = [
        "source", "license_id", "name", "street_address", "city", "state", "zip",
        "start_date", "end_date", "location_end_date", "category",
        "latitude", "longitude", "ground_truth", "confidence", "reason",
    ]

    written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            name    = (r.get("dba_name") or "").strip()
            address = (r.get("full_business_address") or "").strip()
            if not name or not address:
                continue

            city  = (r.get("city") or "San Francisco").strip()
            state = (r.get("state") or "CA").strip()
            if state.upper() not in ("CA", "CALIFORNIA", ""):
                continue

            dba_end   = (r.get("dba_end_date") or "").strip()
            loc_end   = (r.get("location_end_date") or "").strip()
            dba_start = (r.get("dba_start_date") or "").strip()

            # Determine ground truth
            # dba_end_date = the business registration ended -> closed
            # location_end_date = this address/location ended -> closed
            if dba_end:
                ground_truth = "closed"
                confidence   = "high"
                reason       = f"dba_end_date set: {dba_end[:10]}"
            elif loc_end:
                ground_truth = "closed"
                confidence   = "high"
                reason       = f"location_end_date set: {loc_end[:10]}"
            else:
                ground_truth = "open"
                confidence   = _open_confidence(dba_start)
                reason       = f"No end date; dba_start: {dba_start[:10] if dba_start else 'unknown'}"

            lat, lng = _extract_sf_coords(r.get("location"))

            writer.writerow({
                "source":            "sf_licenses",
                "license_id":        r.get("certificate_number", ""),
                "name":              name,
                "street_address":    address,
                "city":              city,
                "state":             state,
                "zip":               r.get("business_zip", ""),
                "start_date":        dba_start,
                "end_date":          dba_end,
                "location_end_date": loc_end,
                "category":          "",   # SF dataset has no category field
                "latitude":          lat,
                "longitude":         lng,
                "ground_truth":      ground_truth,
                "confidence":        confidence,
                "reason":            reason,
            })
            written += 1

    print(f"  Saved {written:,} cleaned SF records -> {path}")
    return path


def _open_confidence(start_date_str: str) -> str:
    """Return 'high' if renewed within 2 years, else 'medium'."""
    if not start_date_str:
        return "medium"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(start_date_str[:10])
        days = (datetime.now() - dt).days
        return "high" if days <= 730 else "medium"
    except Exception:
        return "medium"


# ── LA fetcher ─────────────────────────────────────────────────────────────────
# Real fields confirmed from live API (r4uk-afju):
#   location_account, business_name, street_address, city, zip_code,
#   location_start_date, location_end_date, location_description,
#   council_district
# NOTE: No expiration_date, no license_number, no naics_description,
#       no separate lat/lng. Labels use location_end_date (same as SF).

LA_FIELDS = [
    "location_account",
    "business_name",
    "street_address",
    "city",
    "zip_code",
    "location_start_date",
    "location_end_date",
    "location_description",
]


def fetch_la(limit: int) -> list[dict]:
    print(f"\n{'='*60}")
    print("FETCHING LOS ANGELES BUSINESS LICENSE DATA")
    print(f"Target: {limit:,} records  |  Source: {LA_API}")
    print('='*60)

    all_records: list[dict] = []
    offset = 0

    while len(all_records) < limit:
        batch_size = min(PAGE_SIZE, limit - len(all_records))
        params = {
            "$limit":  batch_size,
            "$offset": offset,
            "$select": ",".join(LA_FIELDS),
            "$order":  "location_account ASC",
        }

        batch = _get(LA_API, params)
        if not batch:
            print(f"  No more records at offset {offset:,}. Done.")
            break

        all_records.extend(batch)
        offset += len(batch)

        if len(all_records) % 5000 == 0 or len(batch) < batch_size:
            print(f"  Fetched {len(all_records):,} LA records so far...")

        if len(batch) < batch_size:
            break

        time.sleep(REQUEST_DELAY)

    print(f"  Total LA records fetched: {len(all_records):,}")
    return all_records


def save_la_csv(records: list[dict]) -> Path:
    """Clean LA records and save to CSV."""
    path = OUTPUT_DIR / "la_businesses.csv"

    fieldnames = [
        "source", "license_id", "name", "street_address", "city", "state", "zip",
        "start_date", "end_date", "location_end_date", "category",
        "latitude", "longitude", "ground_truth", "confidence", "reason",
    ]

    written = 0

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            name    = (r.get("business_name") or "").strip()
            address = (r.get("street_address") or "").strip()
            if not name or not address:
                continue

            city = (r.get("city") or "Los Angeles").strip()
            if city.upper() not in ("LOS ANGELES", "LA", ""):
                # Keep it — many LA suburbs are still in the dataset and valid CA
                pass

            loc_end   = (r.get("location_end_date") or "").strip()
            loc_start = (r.get("location_start_date") or "").strip()

            # Label logic mirrors SF: location_end_date set -> closed
            if loc_end:
                ground_truth = "closed"
                confidence   = "high"
                reason       = f"location_end_date set: {loc_end[:10]}"
            else:
                ground_truth = "open"
                confidence   = _open_confidence(loc_start)
                reason       = f"No end date; location_start: {loc_start[:10] if loc_start else 'unknown'}"

            writer.writerow({
                "source":            "la_licenses",
                "license_id":        r.get("location_account", ""),
                "name":              name,
                "street_address":    address,
                "city":              city,
                "state":             "CA",
                "zip":               r.get("zip_code", ""),
                "start_date":        loc_start,
                "end_date":          "",        # LA has no separate business end date
                "location_end_date": loc_end,
                "category":          "",        # LA dataset has no category field
                "latitude":          "",        # LA dataset has no coordinates
                "longitude":         "",
                "ground_truth":      ground_truth,
                "confidence":        confidence,
                "reason":            reason,
            })
            written += 1

    print(f"  Saved {written:,} cleaned LA records -> {path}")
    return path


# ── San Diego fetcher ──────────────────────────────────────────────────────────
# SD publishes separate CSVs for active and inactive businesses.
# Active:   https://seshat.datasd.org/business_tax_certificates/sd_businesses_active_datasd.csv
# Inactive: https://seshat.datasd.org/business_tax_certificates/sd_businesses_inactive_2015tocurr_datasd.csv
# Fields confirmed: account_key, account_status, dba_name, naics_description,
#                   address_no, address_road, address_sfx, address_city,
#                   address_state, address_zip, address_suite, lat, lng,
#                   date_business_start, date_cert_expiration

SD_ACTIVE_URL   = "https://seshat.datasd.org/business_tax_certificates/sd_businesses_active_datasd.csv"
SD_INACTIVE_URL = "https://seshat.datasd.org/business_tax_certificates/sd_businesses_inactive_2015tocurr_datasd.csv"

RAW_DIR = OUTPUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _download_csv_rows(url: str, limit: int) -> list[dict]:
    """Stream a CSV URL and return up to limit rows as dicts."""
    import csv as _csv
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            rows: list[dict] = []
            reader = _csv.DictReader(
                line.decode("utf-8", errors="replace")
                for line in r.iter_lines()
            )
            for row in reader:
                rows.append(dict(row))
                if len(rows) >= limit:
                    break
                if len(rows) % 20000 == 0:
                    print(f"    ...{len(rows):,} rows")
            return rows
        except requests.RequestException as exc:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


def _build_sd_address(r: dict) -> str:
    parts = [
        r.get("address_no", "").strip(),
        r.get("address_pd", "").strip(),
        r.get("address_road", "").strip(),
        r.get("address_sfx", "").strip(),
    ]
    base = " ".join(p for p in parts if p)
    suite = r.get("address_suite", "").strip()
    if suite:
        base += f" Ste {suite}"
    return base.strip()


def _save_sd_csv(active_rows: list[dict], inactive_rows: list[dict]) -> Path:
    path = OUTPUT_DIR / "sd_businesses.csv"
    fieldnames = [
        "source", "license_id", "name", "street_address", "city", "state", "zip",
        "start_date", "end_date", "location_end_date", "category",
        "latitude", "longitude", "ground_truth", "confidence", "reason",
    ]
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in active_rows:
            name = (r.get("dba_name") or r.get("business_owner_name") or "").strip()
            address = _build_sd_address(r)
            if not name or not address:
                continue
            start = (r.get("date_business_start") or "").strip()
            exp   = (r.get("date_cert_expiration") or "").strip()
            writer.writerow({
                "source":            "sd_licenses",
                "license_id":        r.get("account_key", ""),
                "name":              name,
                "street_address":    address,
                "city":              (r.get("address_city") or "San Diego").strip(),
                "state":             r.get("address_state", "CA").strip() or "CA",
                "zip":               r.get("address_zip", "").strip(),
                "start_date":        start,
                "end_date":          "",
                "location_end_date": "",
                "category":          r.get("naics_description", "").strip(),
                "latitude":          r.get("lat", "").strip(),
                "longitude":         r.get("lng", "").strip(),
                "ground_truth":      "open",
                "confidence":        _open_confidence(start),
                "reason":            f"Active SD license; cert expires {exp[:10] if exp else 'unknown'}",
            })
            written += 1

        for r in inactive_rows:
            name = (r.get("dba_name") or r.get("business_owner_name") or "").strip()
            address = _build_sd_address(r)
            if not name or not address:
                continue
            exp = (r.get("date_cert_expiration") or "").strip()
            status = (r.get("account_status") or "Cancelled").strip()
            writer.writerow({
                "source":            "sd_licenses",
                "license_id":        r.get("account_key", ""),
                "name":              name,
                "street_address":    address,
                "city":              (r.get("address_city") or "San Diego").strip(),
                "state":             r.get("address_state", "CA").strip() or "CA",
                "zip":               r.get("address_zip", "").strip(),
                "start_date":        (r.get("date_business_start") or "").strip(),
                "end_date":          exp,
                "location_end_date": "",
                "category":          r.get("naics_description", "").strip(),
                "latitude":          r.get("lat", "").strip(),
                "longitude":         r.get("lng", "").strip(),
                "ground_truth":      "closed",
                "confidence":        "high",
                "reason":            f"SD account_status={status}; cert expired {exp[:10] if exp else 'unknown'}",
            })
            written += 1

    print(f"  Saved {written:,} SD records -> {path}")
    return path


def fetch_sd(limit: int) -> None:
    print(f"\n{'='*60}")
    print("FETCHING SAN DIEGO BUSINESS LICENSE DATA")
    print(f"Target: up to {limit:,} records per file")
    print('='*60)

    half = limit // 2  # split budget between active and inactive

    print(f"  Downloading active businesses (limit={half:,})...")
    try:
        active = _download_csv_rows(SD_ACTIVE_URL, half)
        print(f"  Active: {len(active):,} records")
        # Save raw
        raw_path = RAW_DIR / "sd_active_raw.csv"
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            if active:
                writer = csv.DictWriter(f, fieldnames=list(active[0].keys()))
                writer.writeheader()
                writer.writerows(active)
        print(f"  Raw active saved: {raw_path}")
    except Exception as e:
        print(f"  [WARN] SD active fetch failed: {e}")
        active = []

    time.sleep(REQUEST_DELAY)

    print(f"  Downloading inactive businesses (2015-current, limit={half:,})...")
    try:
        inactive = _download_csv_rows(SD_INACTIVE_URL, half)
        print(f"  Inactive: {len(inactive):,} records")
        raw_path = RAW_DIR / "sd_inactive_raw.csv"
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            if inactive:
                writer = csv.DictWriter(f, fieldnames=list(inactive[0].keys()))
                writer.writeheader()
                writer.writerows(inactive)
        print(f"  Raw inactive saved: {raw_path}")
    except Exception as e:
        print(f"  [WARN] SD inactive fetch failed: {e}")
        inactive = []

    if active or inactive:
        _save_sd_csv(active, inactive)
    else:
        print("  [ERROR] No SD data fetched.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch SF and LA business license golden data")
    parser.add_argument("--sf-only",  action="store_true")
    parser.add_argument("--la-only",  action="store_true")
    parser.add_argument("--sd-only",  action="store_true")
    parser.add_argument("--limit",    type=int, default=50000)
    args = parser.parse_args()

    only_one = args.sf_only or args.la_only or args.sd_only
    run_sf = args.sf_only or not only_one
    run_la = args.la_only or not only_one
    run_sd = args.sd_only or not only_one

    if run_sf:
        sf_raw = fetch_sf(args.limit)
        raw_path = OUTPUT_DIR / "sf_businesses_raw.json"
        raw_path.write_text(json.dumps(sf_raw, indent=2), encoding="utf-8")
        print(f"  Raw cache saved: {raw_path}")
        save_sf_csv(sf_raw)

    if run_la:
        la_raw = fetch_la(args.limit)
        raw_path = OUTPUT_DIR / "la_businesses_raw.json"
        raw_path.write_text(json.dumps(la_raw, indent=2), encoding="utf-8")
        print(f"  Raw cache saved: {raw_path}")
        save_la_csv(la_raw)

    if run_sd:
        fetch_sd(args.limit)

    print("\nDone. Next step: python scripts/build_golden_dataset.py")


if __name__ == "__main__":
    main()
