"""
scorer.py — Signal-based open/closed prediction engine.

Architecture:
  - Every place starts at 0.0 (OPEN assumption).
  - Closed signals push the score toward 1.0.
  - Open signals pull the score toward 0.0.
  - Amplifiers add bonus weight when signals combine.
  - Threshold: score >= 0.5 → CLOSED, score < 0.5 → OPEN.
  - Confidence is reported as 50–99% (50 = uncertain, 99 = certain).

Tuning: change numbers in WEIGHTS only — zero logic changes needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

# ── Tunable weights ────────────────────────────────────────────────────────────
WEIGHTS: dict[str, float] = {
    # Closed signals
    "explicitly_marked_closed":    0.85,
    "dead_website":                0.50,
    "closure_in_name":             0.65,
    "osm_disused_tag":             0.55,
    "no_contact_info":             0.18,
    "very_low_data_confidence":    0.14,
    "single_unverified_source":    0.10,
    "high_turnover_category":      0.07,
    "missing_address":             0.06,
    "placeholder_name":            0.08,
    # Open signals (negative — pull toward 0.0)
    "website_confirmed_active":   -0.45,
    "rich_data_profile":          -0.20,
    "high_data_confidence":       -0.15,
    "multiple_sources_agree":     -0.18,
    "two_sources":                -0.08,
    "national_chain":             -0.60,
    "has_brand_info":             -0.10,
    "recent_verification":        -0.25,
    # Amplifiers (applied after individual signals)
    "dead_web_high_turnover":      0.15,
    "no_contact_high_turnover":    0.12,
    "low_conf_single_source":      0.12,
    "extra_weak_signal":           0.08,
    "active_web_multi_source":    -0.15,
}

# ── Reference data ─────────────────────────────────────────────────────────────
KNOWN_CHAINS: set[str] = {
    "starbucks", "mcdonalds", "walmart", "target", "costco",
    "home depot", "lowes", "cvs", "walgreens", "subway",
    "taco bell", "burger king", "wendys", "chick-fil-a",
    "in-n-out", "panda express", "chipotle", "dominos",
    "pizza hut", "dollar tree", "dollar general", "7-eleven",
    "trader joes", "whole foods", "safeway", "kroger",
    "bank of america", "chase", "wells fargo", "us bank",
    "verizon", "att", "tmobile", "sprint", "apple store",
    "best buy", "autozone", "oreilly", "jiffy lube",
    "petsmart", "petco", "hobby lobby", "michaels",
    "ross", "tj maxx", "marshalls", "old navy", "gap",
    "macys", "nordstrom", "kohls", "jcpenney",
}

HIGH_TURNOVER_CATEGORIES: set[str] = {
    "restaurant", "cafe", "bar", "pub", "fast_food", "fast food", "food_court",
    "nightclub", "food_truck", "food truck", "popup", "pop_up",
    "clothes", "clothing", "shoes", "boutique", "fashion", "retail",
    "beauty", "beauty salon", "hair salon", "hairdresser", "nail_salon", "nail salon",
    "salon",
    "gift", "gift shop", "souvenir",
    "furniture", "home_goods", "home goods",
    "video_games", "video games", "bookstore", "books",
    "ice_cream", "ice cream", "dessert", "bakery",
    "florist", "flowers",
    "antique", "antiques", "vintage",
}

CLOSURE_KEYWORDS: list[str] = [
    "closed", "former", "defunct", "abandoned",
    "out of business", "coming soon", "vacant",
    "for lease", "for rent",
]

# Patterns for placeholder business names
_PLACEHOLDER_RE: list[re.Pattern] = [
    re.compile(r"^unit\s+\d+", re.IGNORECASE),
    re.compile(r"^suite\s+\d+", re.IGNORECASE),
    re.compile(r"^\d+$"),
]

# Signals that count as "hard" for amplifier thresholding
_HARD_SIGNAL_NAMES: set[str] = {
    "explicitly_marked_closed",
    "dead_website",
    "closure_in_name",
    "osm_disused_tag",
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """A single piece of evidence about a place's status."""
    name: str
    weight: float
    direction: str   # "open" or "closed"
    reason: str
    fired: bool = False


@dataclass
class ScorerResult:
    """Output of PlaceScorer.score()."""
    status: str           # "open" or "closed"
    confidence: float     # 50–99 (integer percentage, returned as float)
    raw_score: float      # 0.0–1.0 before conversion
    fired_signals: List[Signal] = field(default_factory=list)

    @property
    def explanation(self) -> list[str]:
        """Human-readable list of reasons (for API explanation field)."""
        return [s.reason for s in self.fired_signals]


# ── Scorer ─────────────────────────────────────────────────────────────────────

class PlaceScorer:
    """
    Signal-based scorer.  Instantiate once at module level; call score() per place.
    Thread-safe: score() creates fresh Signal objects each call.
    """

    def __init__(self) -> None:
        # Store signal definitions (name, direction, reason) — weights come from WEIGHTS dict
        self._defs: list[tuple[str, str, str]] = [
            # name                          direction   reason
            ("explicitly_marked_closed",    "closed",   "Data source explicitly marks this place as closed"),
            ("dead_website",                "closed",   "Business website is dead or domain expired"),
            ("closure_in_name",             "closed",   "Business name suggests it is no longer operating"),
            ("osm_disused_tag",             "closed",   "OpenStreetMap marks this location as disused"),
            ("no_contact_info",             "closed",   "No contact information found for this business"),
            ("very_low_data_confidence",    "closed",   "Data source has very low confidence in this record"),
            ("single_unverified_source",    "closed",   "Only one unverified data source"),
            ("high_turnover_category",      "closed",   "Business category has high closure rate"),
            ("missing_address",             "closed",   "No address on record"),
            ("placeholder_name",            "closed",   "Business name appears to be a placeholder"),
            ("website_confirmed_active",    "open",     "Business website is live and responding"),
            ("rich_data_profile",           "open",     "Business has complete contact information"),
            ("high_data_confidence",        "open",     "Data source has high confidence in this record"),
            ("multiple_sources_agree",      "open",     "Multiple independent sources confirm this place"),
            ("two_sources",                 "open",     "Two independent sources confirm this place"),
            ("national_chain",              "open",     "National chain — extremely unlikely to be closed"),
            ("has_brand_info",              "open",     "Brand information on record"),
            ("recent_verification",         "open",     "Website recently verified as active"),
        ]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _normalize(self, place: dict) -> dict:
        """Extract and normalize all relevant fields from a place dict."""
        # Websites (Overture list or singular OSM field)
        websites = place.get("websites") or []
        has_website = (
            (isinstance(websites, list) and len(websites) > 0)
            or bool(place.get("website") or place.get("website_url"))
        )

        # Phones
        phones = place.get("phones") or []
        has_phone = (
            (isinstance(phones, list) and len(phones) > 0)
            or bool(place.get("phone"))
        )

        # Social media
        socials = place.get("socials") or []
        has_social = isinstance(socials, list) and len(socials) > 0

        # Sources
        sources = place.get("sources") or []
        num_sources = len(sources) if isinstance(sources, list) else 0

        # Overture confidence score
        confidence: Optional[float] = None
        raw_conf = place.get("confidence")
        if raw_conf is not None:
            try:
                confidence = float(raw_conf)
            except (ValueError, TypeError):
                pass

        # Brand
        brand = place.get("brand") or {}
        has_brand = isinstance(brand, dict) and len(brand) > 0

        # Address
        address = str(place.get("address") or "").strip()
        has_address = bool(address)

        # Website verification
        website_status = str(place.get("website_status") or "").lower()
        website_checked_at = place.get("website_checked_at")
        recently_checked_active = False
        if website_status == "active" and website_checked_at:
            try:
                dt = datetime.fromisoformat(
                    str(website_checked_at).replace("Z", "+00:00")
                )
                recently_checked_active = (datetime.now(timezone.utc) - dt).days <= 30
            except Exception:
                pass

        # OSM disused / end_date tags
        osm_disused = any(
            place.get(k)
            for k in ("disused:amenity", "disused:shop", "closed:amenity",
                      "closed:shop", "end_date")
        )

        # Operating status
        operating_status = str(place.get("operating_status") or "").lower()

        # Name / category / address
        name = str(place.get("name") or "").strip()
        category = str(place.get("category") or "").lower().strip()

        return {
            "name": name,
            "name_lower": name.lower(),
            "category": category,
            "address": address,
            "has_website": has_website,
            "has_phone": has_phone,
            "has_social": has_social,
            "has_brand": has_brand,
            "has_address": has_address,
            "num_sources": num_sources,
            "confidence": confidence,
            "website_status": website_status,
            "recently_checked_active": recently_checked_active,
            "osm_disused": osm_disused,
            "operating_status": operating_status,
        }

    def _is_placeholder(self, name: str) -> bool:
        if len(name) < 3:
            return True
        return any(p.match(name) for p in _PLACEHOLDER_RE)

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, place: dict) -> ScorerResult:
        """
        Score a single place dict.

        place should contain any combination of:
          name, category, address, websites, phones, socials, brand,
          confidence, sources, website_status, website_checked_at,
          operating_status, disused:*, end_date
        """
        e = self._normalize(place)

        # Build fresh signal objects for this call (ensures thread safety)
        signals = {
            name: Signal(name, WEIGHTS[name], direction, reason)
            for name, direction, reason in self._defs
        }

        # ── Evaluate closed signals ────────────────────────────────────────────
        if e["operating_status"] in ("closed", "inactive", "defunct"):
            signals["explicitly_marked_closed"].fired = True

        if e["website_status"] == "likely_closed":
            signals["dead_website"].fired = True

        if any(kw in e["name_lower"] for kw in CLOSURE_KEYWORDS):
            signals["closure_in_name"].fired = True

        if e["osm_disused"]:
            signals["osm_disused_tag"].fired = True

        if not e["has_website"] and not e["has_phone"] and not e["has_social"]:
            signals["no_contact_info"].fired = True

        if e["confidence"] is not None and e["confidence"] < 0.4:
            signals["very_low_data_confidence"].fired = True

        if e["num_sources"] == 1 and (
            e["confidence"] is None or e["confidence"] < 0.55
        ):
            signals["single_unverified_source"].fired = True

        if e["category"] in HIGH_TURNOVER_CATEGORIES:
            signals["high_turnover_category"].fired = True

        if not e["has_address"]:
            signals["missing_address"].fired = True

        if self._is_placeholder(e["name"]):
            signals["placeholder_name"].fired = True

        # ── Evaluate open signals ──────────────────────────────────────────────
        if e["website_status"] == "active":
            signals["website_confirmed_active"].fired = True

        if e["has_website"] and e["has_phone"] and e["has_social"]:
            signals["rich_data_profile"].fired = True

        if e["confidence"] is not None and e["confidence"] > 0.85:
            signals["high_data_confidence"].fired = True

        if e["num_sources"] >= 3:
            signals["multiple_sources_agree"].fired = True

        if e["num_sources"] == 2:
            signals["two_sources"].fired = True

        if any(chain in e["name_lower"] for chain in KNOWN_CHAINS):
            signals["national_chain"].fired = True

        if e["has_brand"]:
            signals["has_brand_info"].fired = True

        if e["recently_checked_active"]:
            signals["recent_verification"].fired = True

        # ── Sum fired signal weights ───────────────────────────────────────────
        fired_list = [s for s in signals.values() if s.fired]
        raw_score: float = sum(s.weight for s in fired_list)

        # ── Apply amplifiers ───────────────────────────────────────────────────
        amplifier_signals: list[Signal] = []

        def _amp(name: str, direction: str, reason: str) -> None:
            w = WEIGHTS[name]
            nonlocal raw_score
            raw_score += w
            amplifier_signals.append(Signal(name, w, direction, reason, fired=True))

        if signals["dead_website"].fired and signals["high_turnover_category"].fired:
            _amp("dead_web_high_turnover", "closed",
                 "Dead website for a high-risk category business")

        if signals["no_contact_info"].fired and signals["high_turnover_category"].fired:
            _amp("no_contact_high_turnover", "closed",
                 "No presence for a high-turnover business")

        if signals["very_low_data_confidence"].fired and signals["single_unverified_source"].fired:
            _amp("low_conf_single_source", "closed",
                 "Unverified low-quality record")

        # Count weak closed signals (excluding hard/explicit ones)
        weak_closed = [
            s for s in fired_list
            if s.direction == "closed" and s.name not in _HARD_SIGNAL_NAMES
        ]
        if len(weak_closed) >= 3:
            extra_count = len(weak_closed) - 2  # one bonus per signal beyond 2
            for _ in range(extra_count):
                _amp("extra_weak_signal", "closed",
                     f"Multiple closure indicators present ({len(weak_closed)} weak signals)")

        if signals["website_confirmed_active"].fired and signals["multiple_sources_agree"].fired:
            _amp("active_web_multi_source", "open",
                 "Strong corroborating evidence place is active")

        # ── Clamp to [0.0, 1.0] ───────────────────────────────────────────────
        raw_score = max(0.0, min(1.0, raw_score))

        # ── Convert to status + confidence ────────────────────────────────────
        if raw_score >= 0.50:
            status = "closed"
            confidence = 50.0 + (raw_score - 0.50) * 98.0
        else:
            status = "open"
            confidence = 50.0 + (0.50 - raw_score) * 98.0

        confidence = float(round(min(99.0, max(50.0, confidence))))

        all_fired = fired_list + amplifier_signals

        return ScorerResult(
            status=status,
            confidence=confidence,
            raw_score=raw_score,
            fired_signals=all_fired,
        )


# Module-level singleton — import and reuse across the app
scorer = PlaceScorer()
