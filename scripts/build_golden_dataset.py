"""
build_golden_dataset.py — Combine SF, LA, and SD license CSVs into a single
standardized golden dataset for scorer validation.

Usage:
    python scripts/build_golden_dataset.py

Input:
    scripts/data/golden/sf_businesses.csv   (SF Open Data API)
    scripts/data/golden/la_businesses.csv   (LA Open Data API)
    scripts/data/golden/sd_businesses.csv   (SD Business Tax Certificates CSV)

Output:
    scripts/data/golden/golden_dataset.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
SF_CSV     = GOLDEN_DIR / "sf_businesses.csv"
LA_CSV     = GOLDEN_DIR / "la_businesses.csv"
SD_CSV     = GOLDEN_DIR / "sd_businesses.csv"
OUT_CSV    = GOLDEN_DIR / "golden_dataset.csv"

OUTPUT_FIELDS = [
    "source", "license_id", "name", "address", "city", "state", "zip",
    "latitude", "longitude", "category",
    "ground_truth", "confidence", "reason",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_str(s: str) -> str:
    return " ".join(s.strip().split()).upper()


def _dedup_key(row: dict) -> str:
    """Key for duplicate detection: normalized name + address."""
    name    = _clean_str(row.get("name", ""))
    address = _clean_str(row.get("address", ""))
    city    = _clean_str(row.get("city", ""))
    return f"{name}||{address}||{city}"


# ── SF loader ─────────────────────────────────────────────────────────────────

def load_sf(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        print(f"  [WARN] SF file not found: {path}")
        return rows

    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name    = r.get("name", "").strip()
            address = r.get("street_address", "").strip()
            if not name or not address:
                continue

            rows.append({
                "source":       "sf_licenses",
                "license_id":   r.get("license_id", ""),
                "name":         name,
                "address":      address,
                "city":         r.get("city", "San Francisco").strip(),
                "state":        r.get("state", "CA").strip() or "CA",
                "zip":          r.get("zip", "").strip(),
                "latitude":     r.get("latitude", "").strip(),
                "longitude":    r.get("longitude", "").strip(),
                "category":     r.get("category", "").strip(),
                "ground_truth": r.get("ground_truth", "").strip(),
                "confidence":   r.get("confidence", "medium").strip(),
                "reason":       r.get("reason", "").strip(),
            })

    print(f"  Loaded {len(rows):,} records from {path.name}")
    return rows


# ── LA loader ─────────────────────────────────────────────────────────────────

def load_la(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        print(f"  [WARN] LA file not found: {path}")
        return rows

    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name    = r.get("name", "").strip()
            address = r.get("street_address", "").strip()
            if not name or not address:
                continue

            rows.append({
                "source":       "la_licenses",
                "license_id":   r.get("license_id", ""),
                "name":         name,
                "address":      address,
                "city":         r.get("city", "Los Angeles").strip(),
                "state":        "CA",
                "zip":          r.get("zip", "").strip(),
                "latitude":     r.get("latitude", "").strip(),
                "longitude":    r.get("longitude", "").strip(),
                "category":     r.get("category", "").strip(),
                "ground_truth": r.get("ground_truth", "").strip(),
                "confidence":   r.get("confidence", "medium").strip(),
                "reason":       r.get("reason", "").strip(),
            })

    print(f"  Loaded {len(rows):,} records from {path.name}")
    return rows


# ── Filters ────────────────────────────────────────────────────────────────────

def filter_records(rows: list[dict]) -> list[dict]:
    """Apply all exclusion rules. Returns cleaned list."""
    pre = len(rows)

    # 1. Drop records with no name
    rows = [r for r in rows if r["name"]]

    # 2. Drop records with no address
    rows = [r for r in rows if r["address"]]

    # 3. Drop records outside California (state field check)
    rows = [
        r for r in rows
        if r.get("state", "CA").upper() in ("CA", "CALIFORNIA", "")
    ]

    # 4. Drop records with no valid ground_truth label
    rows = [r for r in rows if r.get("ground_truth") in ("open", "closed")]

    after_basic = len(rows)
    dropped_basic = pre - after_basic

    # 5. Deduplicate: same name + address → keep the one with higher confidence
    #    (if tie, prefer "closed" — conservative)
    seen: dict[str, dict] = {}
    CONF_RANK = {"high": 2, "medium": 1, "low": 0}

    for r in rows:
        key = _dedup_key(r)
        if key not in seen:
            seen[key] = r
        else:
            existing = seen[key]
            existing_rank = CONF_RANK.get(existing["confidence"], 0)
            new_rank      = CONF_RANK.get(r["confidence"], 0)
            if new_rank > existing_rank:
                seen[key] = r
            elif new_rank == existing_rank and r["ground_truth"] == "closed":
                seen[key] = r   # prefer closed on tie

    deduped = list(seen.values())
    dropped_dupes = after_basic - len(deduped)

    print(f"\n  Filter summary:")
    print(f"    Input:              {pre:>8,}")
    print(f"    Dropped (no name/addr/state/label): {dropped_basic:>6,}")
    print(f"    Dropped (duplicates):               {dropped_dupes:>6,}")
    print(f"    Remaining:          {len(deduped):>8,}")

    return deduped


# ── Stats ──────────────────────────────────────────────────────────────────────

def print_stats(rows: list[dict]) -> None:
    total = len(rows)

    by_truth:  defaultdict[str, int] = defaultdict(int)
    by_conf:   defaultdict[str, int] = defaultdict(int)
    by_source: defaultdict[str, int] = defaultdict(int)
    by_city:   defaultdict[str, int] = defaultdict(int)
    with_coords = 0

    for r in rows:
        by_truth[r["ground_truth"]] += 1
        by_conf[r["confidence"]]    += 1
        by_source[r["source"]]      += 1
        city = r.get("city", "Unknown") or "Unknown"
        by_city[city.title()] += 1
        if r.get("latitude") and r.get("longitude"):
            with_coords += 1

    print(f"\n{'='*60}")
    print("GOLDEN DATASET STATISTICS")
    print('='*60)
    print(f"  Total records:            {total:>8,}")
    print(f"\n  Ground truth split:")
    print(f"    Open:                   {by_truth['open']:>8,}  ({by_truth['open']/total*100:.1f}%)")
    print(f"    Closed:                 {by_truth['closed']:>8,}  ({by_truth['closed']/total*100:.1f}%)")
    print(f"\n  Confidence split:")
    print(f"    High:                   {by_conf['high']:>8,}  ({by_conf['high']/total*100:.1f}%)")
    print(f"    Medium:                 {by_conf['medium']:>8,}  ({by_conf['medium']/total*100:.1f}%)")
    print(f"\n  Source split:")
    for src, cnt in sorted(by_source.items()):
        print(f"    {src:<25} {cnt:>8,}")
    print(f"\n  Records with coordinates: {with_coords:>8,}  ({with_coords/total*100:.1f}%)")
    print(f"\n  Top cities:")
    for city, cnt in sorted(by_city.items(), key=lambda x: -x[1])[:15]:
        print(f"    {city:<30} {cnt:>7,}")
    print('='*60)


# ── SD loader ─────────────────────────────────────────────────────────────────

def load_sd(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        print(f"  [WARN] SD file not found: {path}")
        return rows

    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name    = r.get("name", "").strip()
            address = r.get("street_address", "").strip()
            if not name or not address:
                continue

            rows.append({
                "source":       "sd_licenses",
                "license_id":   r.get("license_id", ""),
                "name":         name,
                "address":      address,
                "city":         r.get("city", "San Diego").strip(),
                "state":        r.get("state", "CA").strip() or "CA",
                "zip":          r.get("zip", "").strip(),
                "latitude":     r.get("latitude", "").strip(),
                "longitude":    r.get("longitude", "").strip(),
                "category":     r.get("category", "").strip(),
                "ground_truth": r.get("ground_truth", "").strip(),
                "confidence":   r.get("confidence", "medium").strip(),
                "reason":       r.get("reason", "").strip(),
            })

    print(f"  Loaded {len(rows):,} records from {path.name}")
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\nBuilding golden dataset...")

    all_rows: list[dict] = []
    all_rows.extend(load_sf(SF_CSV))
    all_rows.extend(load_la(LA_CSV))
    all_rows.extend(load_sd(SD_CSV))

    if not all_rows:
        print("\n[ERROR] No data loaded. Run fetch_golden_data.py first.")
        sys.exit(1)

    filtered = filter_records(all_rows)
    print_stats(filtered)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(filtered)

    print(f"\n  Saved: {OUT_CSV}")
    print("\nNext steps:")
    print("  python scripts/apply_golden_labels.py --local")
    print("  python scripts/validate_scorer.py --local --sample")


if __name__ == "__main__":
    main()
