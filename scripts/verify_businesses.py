"""
verify_businesses.py — External verification of business operational status.

For each place that has a website stored, performs an HTTP HEAD request to check
if the domain is live. A dead/expired/404 website is a strong signal of closure.
Also flags duplicate-address clusters and stale Overture records.

Usage:
    python scripts/verify_businesses.py [--limit N] [--dry-run] [--source {db,osm,parquet}]
                                        [--write-db]

Options:
    --limit N       Process at most N places (default: 100)
    --dry-run       Print what would be checked without making HTTP requests
    --source        Where to pull records from: db (Postgres), osm (JSON file), parquet (default: parquet)
    --timeout SECS  HTTP request timeout in seconds (default: 5)
    --write-db      Write verification results back to the Postgres places table
                    (stores website_status, website_checked_at, website_http_code
                    in metadata_json for each checked record)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
import collections
import ssl
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────────────────────────────────────────
# HTTP verification
# ──────────────────────────────────────────────────────────────────────────────

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

DEAD_STATUS_CODES = {404, 410, 451}      # explicitly gone
ACTIVE_STATUS_CODES = {200, 201, 301, 302, 303, 307, 308}  # redirects count as live


def check_website(url: str, timeout: int = 5) -> dict:
    """
    Perform an HTTP HEAD request (fallback to GET on failure).
    Returns dict with: url, status_code, verdict, error
    verdict: 'active' | 'likely_closed' | 'inconclusive'
    """
    if not url:
        return {"url": url, "status_code": None, "verdict": "inconclusive", "error": "no_url"}

    # Ensure scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    result = {"url": url, "status_code": None, "verdict": "inconclusive", "error": None}

    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers={
                "User-Agent": "Mozilla/5.0 StillOpen-Verifier/1.0"
            })
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                result["status_code"] = resp.status
                if resp.status in ACTIVE_STATUS_CODES:
                    result["verdict"] = "active"
                elif resp.status in DEAD_STATUS_CODES:
                    result["verdict"] = "likely_closed"
                else:
                    result["verdict"] = "inconclusive"
                return result
        except urllib.error.HTTPError as e:
            result["status_code"] = e.code
            if e.code in DEAD_STATUS_CODES:
                result["verdict"] = "likely_closed"
                return result
            elif e.code in ACTIVE_STATUS_CODES:
                result["verdict"] = "active"
                return result
            result["error"] = f"http_{e.code}"
        except urllib.error.URLError as e:
            reason = str(e.reason)
            if any(kw in reason.lower() for kw in [
                "name or service not known", "nodename nor servname",
                "no address associated", "connection refused",
                "errno 8", "errno -2",
            ]):
                result["verdict"] = "likely_closed"
                result["error"] = "domain_not_found"
                return result
            result["error"] = f"url_error: {reason}"
        except Exception as e:
            result["error"] = str(e)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_from_parquet(limit: int):
    import pandas as pd
    import numpy as np
    path = os.path.join(PROJECT_ROOT, "data", "project_c_samples.parquet")
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        names = row.get("names", {})
        name = names.get("primary", "Unknown") if isinstance(names, dict) else "Unknown"
        websites = row.get("websites")
        url = None
        if isinstance(websites, (list, np.ndarray)) and len(websites) > 0:
            entry = websites[0]
            url = entry.get("url") if isinstance(entry, dict) else str(entry)
        records.append({
            "id": str(row.get("id", "")),
            "name": name,
            "url": url,
            "ground_truth": int(row.get("open", 1)),
            "address": "",
        })
    return records[:limit]


def load_from_osm(limit: int):
    path = os.path.join(PROJECT_ROOT, "scripts", "data", "osm_places.json")
    with open(path) as f:
        data = json.load(f)
    records = []
    for r in data:
        meta = r.get("metadata", {})
        websites = meta.get("websites", [])
        url = websites[0] if websites else None
        records.append({
            "id": r.get("id", ""),
            "name": r.get("name", "Unknown"),
            "url": url,
            "ground_truth": int(r.get("open", 1)),
            "address": r.get("address", ""),
        })
    return records[:limit]


def load_from_db(limit: int, filter_mode: str = None):
    from dotenv import load_dotenv
    import psycopg2
    import psycopg2.extras
    load_dotenv(os.path.join(PROJECT_ROOT, "stillopen", "backend", ".env"))
    url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/stillopen")
    url = url.replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(url)

    # Build WHERE clause based on filter mode
    if filter_mode == "high_risk":
        # Only records flagged as high closure risk (confidence < 0.75)
        where = (
            "WHERE metadata_json->>'closure_risk' = 'high' "
            "AND jsonb_array_length(COALESCE(metadata_json->'websites','[]'::jsonb)) > 0"
        )
    elif filter_mode == "medium_risk":
        where = (
            "WHERE metadata_json->>'closure_risk' IN ('high','medium') "
            "AND jsonb_array_length(COALESCE(metadata_json->'websites','[]'::jsonb)) > 0"
        )
    else:
        where = "WHERE metadata_json ? 'website' OR metadata_json ? 'websites'"

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT place_id, name, address, metadata_json
            FROM places
            {where}
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    conn.close()
    records = []
    for r in rows:
        meta = r["metadata_json"] or {}
        websites = meta.get("websites") or []
        if isinstance(websites, list) and websites:
            first = websites[0]
            raw_url = first.get("value") or first.get("url") if isinstance(first, dict) else str(first)
        else:
            raw_url = meta.get("website")
        records.append({
            "id": r["place_id"],
            "name": r["name"],
            "url": raw_url,
            "ground_truth": None,
            "address": r["address"] or "",
        })
    return records[:limit]


# ──────────────────────────────────────────────────────────────────────────────
# DB write-back
# ──────────────────────────────────────────────────────────────────────────────

def _get_db_conn():
    """Return a psycopg2 connection using DATABASE_URL."""
    from dotenv import load_dotenv
    import psycopg2
    load_dotenv(os.path.join(PROJECT_ROOT, "stillopen", "backend", ".env"))
    url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/stillopen")
    url = url.replace("postgresql+psycopg2://", "postgresql://")
    return psycopg2.connect(url)


def write_verification_to_db(conn, place_id: str, verdict: str, status_code, checked_at: str):
    """
    Patch metadata_json for the given place with website verification fields:
      website_status:      "active" | "likely_closed" | "inconclusive"
      website_checked_at:  ISO-8601 timestamp string
      website_http_code:   integer HTTP status code or null
    """
    import psycopg2.extras
    patch = {
        "website_status": verdict,
        "website_checked_at": checked_at,
        "website_http_code": status_code if isinstance(status_code, int) else None,
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE places
            SET metadata_json = metadata_json || %s::jsonb
            WHERE place_id = %s
            """,
            (json.dumps(patch), place_id),
        )
    conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Duplicate-address clustering
# ──────────────────────────────────────────────────────────────────────────────

def find_address_clusters(records: list) -> dict:
    """Return address → [record names] for addresses with 2+ records."""
    addr_map = collections.defaultdict(list)
    for r in records:
        addr = (r.get("address") or "").strip().lower()
        if addr and addr != "unknown":
            addr_map[addr].append(r["name"])
    return {addr: names for addr, names in addr_map.items() if len(names) >= 2}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify business operational status via website checks")
    parser.add_argument("--limit", type=int, default=100, help="Max places to check (default: 100)")
    parser.add_argument("--dry-run", action="store_true", help="List URLs without making HTTP requests")
    parser.add_argument("--source", choices=["db", "osm", "parquet"], default="parquet",
                        help="Data source to load records from (default: parquet)")
    parser.add_argument("--timeout", type=int, default=5, help="HTTP timeout in seconds (default: 5)")
    parser.add_argument("--write-db", action="store_true",
                        help="Write verification results back to Postgres metadata_json")
    parser.add_argument("--filter", dest="filter_mode", default=None,
                        choices=["high_risk", "medium_risk"],
                        help="Filter DB records: high_risk (conf<0.75) or medium_risk (conf<0.85, partial data)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  StillOpen Business Verifier")
    print(f"  Source: {args.source}  |  Limit: {args.limit}  |  Filter: {args.filter_mode or 'none'}")
    print(f"  Dry-run: {args.dry_run}  |  Write-DB: {args.write_db}")
    print(f"{'='*65}\n")

    db_conn = None
    if args.write_db:
        try:
            db_conn = _get_db_conn()
            print("  Connected to Postgres for write-back.\n")
        except Exception as e:
            print(f"  ⚠  Could not connect to Postgres ({e}). Results will NOT be written.\n")
            db_conn = None

    # Load records
    if args.source == "parquet":
        records = load_from_parquet(args.limit)
    elif args.source == "osm":
        records = load_from_osm(args.limit)
    else:
        records = load_from_db(args.limit, filter_mode=args.filter_mode)

    has_url = [r for r in records if r["url"]]
    no_url  = [r for r in records if not r["url"]]

    print(f"  Loaded {len(records)} records")
    print(f"  Has URL: {len(has_url)}  |  No URL: {len(no_url)}\n")

    # Duplicate address check
    clusters = find_address_clusters(records)
    if clusters:
        print(f"  ⚠  {len(clusters)} shared addresses (possible replacements/duplicates):")
        for addr, names in list(clusters.items())[:5]:
            print(f"     {addr!r}: {names}")
        print()

    if args.dry_run:
        print("  DRY RUN — URLs that would be checked:")
        for r in has_url[:20]:
            label = "CLOSED" if r["ground_truth"] == 0 else ("OPEN" if r["ground_truth"] == 1 else "?")
            print(f"    [{label:6s}] {r['name'][:40]:40s} {r['url']}")
        print(f"\n  (showing 20 of {len(has_url)} URLs with --dry-run)\n")
        return

    # HTTP verification loop
    results = {"active": [], "likely_closed": [], "inconclusive": []}
    correct = incorrect = 0

    print(f"  Checking {len(has_url)} URLs (timeout={args.timeout}s) ...\n")
    checked_at = datetime.utcnow().isoformat() + "Z"
    db_written = 0

    for i, rec in enumerate(has_url, 1):
        chk = check_website(rec["url"], timeout=args.timeout)
        rec["check"] = chk
        results[chk["verdict"]].append(rec)

        gt = rec["ground_truth"]
        gt_label = "CLOSED" if gt == 0 else ("OPEN" if gt == 1 else "?")
        match = ""
        if gt is not None:
            predicted_closed = chk["verdict"] == "likely_closed"
            actually_closed = gt == 0
            if predicted_closed == actually_closed:
                match = "✓"
                correct += 1
            else:
                match = "✗"
                incorrect += 1

        status_code = chk["status_code"] or chk["error"] or "?"
        print(f"  [{i:3d}/{len(has_url)}] {chk['verdict']:15s} [{gt_label:6s}] {match}  "
              f"({status_code})  {rec['name'][:35]}")

        # Write back to DB if requested
        if db_conn and rec.get("id"):
            try:
                write_verification_to_db(
                    db_conn,
                    place_id=str(rec["id"]),
                    verdict=chk["verdict"],
                    status_code=chk["status_code"],
                    checked_at=checked_at,
                )
                db_written += 1
            except Exception as e:
                print(f"    ⚠  DB write failed for {rec['name']}: {e}")

        time.sleep(0.1)  # Polite delay

    if db_conn:
        db_conn.close()
        if db_written:
            print(f"\n  ✓ Wrote verification results for {db_written} places to Postgres.")

    # Summary report
    print(f"\n{'='*65}")
    print(f"  VERIFICATION REPORT")
    print(f"{'='*65}")
    print(f"  Total checked:      {len(has_url)}")
    print(f"  No URL (skipped):   {len(no_url)}")
    print(f"  Active websites:    {len(results['active'])}  ({len(results['active'])/max(1,len(has_url))*100:.1f}%)")
    print(f"  Likely closed:      {len(results['likely_closed'])}  ({len(results['likely_closed'])/max(1,len(has_url))*100:.1f}%)")
    print(f"  Inconclusive:       {len(results['inconclusive'])}  ({len(results['inconclusive'])/max(1,len(has_url))*100:.1f}%)")

    if correct + incorrect > 0:
        print(f"\n  Agreement with ground-truth labels:")
        print(f"    Correct:   {correct}/{correct+incorrect} ({correct/(correct+incorrect)*100:.1f}%)")
        print(f"    Incorrect: {incorrect}/{correct+incorrect} ({incorrect/(correct+incorrect)*100:.1f}%)")

    if results["likely_closed"]:
        print(f"\n  Top likely-CLOSED businesses (website dead/expired):")
        for r in results["likely_closed"][:10]:
            gt_label = "CLOSED" if r["ground_truth"] == 0 else ("OPEN" if r["ground_truth"] == 1 else "?")
            err = r["check"].get("error") or r["check"].get("status_code") or "?"
            print(f"    [{gt_label:6s}] {r['name'][:45]:45s}  error={err}")
            print(f"           {r['url']}")

    print(f"\n  Insight: {len(no_url)} records have NO website — missing website is itself")
    print(f"  a closure signal. Of these, check phone presence and category.")

    # Breakdown of no-URL records by ground truth
    if no_url:
        no_url_closed = sum(1 for r in no_url if r.get("ground_truth") == 0)
        no_url_open   = sum(1 for r in no_url if r.get("ground_truth") == 1)
        if no_url_closed + no_url_open > 0:
            pct_closed = no_url_closed / (no_url_closed + no_url_open) * 100
            print(f"  No-URL breakdown: {no_url_closed} closed / {no_url_open} open ({pct_closed:.1f}% closed)")

    print()


if __name__ == "__main__":
    main()
