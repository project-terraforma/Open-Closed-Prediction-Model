"""
Canonical metadata utilities.

Goal: produce a strict, source-agnostic `canonical` object (Google Places-style,
simplified) from arbitrary `raw` source metadata, while preserving `raw`
unchanged for ML/traceability.
"""

from __future__ import annotations

from typing import Any, Optional
import logging
import re


_BLANK_STRINGS = {"", "none", "null", "nan"}
logger = logging.getLogger("canonical.metadata")


def _none_if_blank(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if s.lower() in _BLANK_STRINGS:
            return None
        return s
    return val


def _first_str(*vals: Any) -> Optional[str]:
    for v in vals:
        v = _none_if_blank(v)
        if isinstance(v, str) and v:
            return v
    return None


def _as_list(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    return [val]


def _normalize_website(raw: Any) -> Optional[str]:
    url = _none_if_blank(raw)
    if url is None:
        return None
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    if url.startswith(("http://", "https://")):
        return url
    return "https://" + url


def _normalize_phone(raw: Any) -> Optional[str]:
    """
    Best-effort international-ish normalization without external deps.
    - Keeps E.164 (+{digits}) when possible.
    - Converts 00-prefixed numbers to +.
    - Assumes US/CA when 10 digits.
    """
    phone = _none_if_blank(raw)
    if phone is None:
        return None
    if not isinstance(phone, str):
        return None

    s = phone.strip()
    if not s:
        return None

    # Keep leading + if present; strip other junk.
    keep_plus = s.startswith("+")
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None

    if s.startswith("00") and len(digits) >= 6:
        return "+" + digits[2:]

    if keep_plus:
        return "+" + digits

    # 10 digits -> assume NANP
    if len(digits) == 10:
        return "+1" + digits

    # 11 digits starting with 1 -> NANP
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits

    # Fallback: return digits-only with + when it looks like a country code number
    if len(digits) >= 12:
        return "+" + digits

    return None


def _extract_addr_from_osm_style(raw: dict) -> dict:
    return {
        "street_number": _first_str(raw.get("addr:housenumber")),
        "route": _first_str(raw.get("addr:street")),
        "locality": _first_str(raw.get("addr:city"), raw.get("addr:municipality"), raw.get("city")),
        "administrative_area_level_1": _first_str(raw.get("addr:state"), raw.get("state")),
        "postal_code": _first_str(raw.get("addr:postcode"), raw.get("postcode")),
        "country": _first_str(raw.get("addr:country"), raw.get("country")),
    }


def _extract_addr_from_overture(raw: dict) -> dict:
    # Overture commonly provides `addresses` (or older code used `overture_addresses`).
    addrs = raw.get("addresses")
    if addrs is None:
        addrs = raw.get("overture_addresses")
    addrs_list = _as_list(addrs)
    if not addrs_list:
        return {}
    first = addrs_list[0]
    if not isinstance(first, dict):
        return {}
    return {
        "street_number": _first_str(first.get("house_number")),
        "route": _first_str(first.get("street")),
        "locality": _first_str(first.get("locality")),
        "administrative_area_level_1": _first_str(first.get("region")),
        "postal_code": _first_str(first.get("postcode")),
        "country": _first_str(first.get("country")),
    }


def _extract_addr_from_openaddresses(raw: dict) -> dict:
    oa = raw.get("openaddresses")
    if not isinstance(oa, dict):
        return {}
    return {
        "street_number": _first_str(oa.get("number")),
        "route": _first_str(oa.get("street")),
        "locality": _first_str(oa.get("city"), oa.get("district")),
        "administrative_area_level_1": _first_str(oa.get("region")),
        "postal_code": _first_str(oa.get("postcode")),
        "country": _first_str(oa.get("country")),
    }


def _build_formatted_address(addr: dict, raw: dict) -> Optional[str]:
    # If the source already has a full formatted address, prefer it.
    direct = _first_str(raw.get("formatted_address"), raw.get("full_address"), raw.get("address"))
    if direct:
        return direct

    street_number = addr.get("street_number")
    route = addr.get("route")
    locality = addr.get("locality")
    admin1 = addr.get("administrative_area_level_1")
    postal = addr.get("postal_code")

    parts: list[str] = []
    if street_number and route:
        parts.append(f"{street_number} {route}")
    elif route:
        parts.append(route)

    if locality:
        parts.append(locality)
    if admin1:
        parts.append(admin1)
    if postal:
        parts.append(postal)

    return ", ".join(parts) if parts else None


def _address_components(addr: dict) -> list[dict]:
    comps: list[dict] = []

    def add_component(value: Optional[str], t: str):
        v = _none_if_blank(value)
        if not isinstance(v, str) or not v:
            return
        comps.append(
            {
                "long_name": v,
                "short_name": v,
                "types": [t],
            }
        )

    add_component(addr.get("street_number"), "street_number")
    add_component(addr.get("route"), "route")
    add_component(addr.get("locality"), "locality")
    add_component(addr.get("administrative_area_level_1"), "administrative_area_level_1")
    add_component(addr.get("postal_code"), "postal_code")
    return comps


def build_canonical_metadata(raw: dict, lat: float, lon: float) -> dict:
    """
    Build a strict canonical metadata object from `raw` + provided geometry.
    Missing fields are explicitly set to None.
    """
    raw = raw if isinstance(raw, dict) else {}

    name = _first_str(
        raw.get("name"),
        (raw.get("names") or {}).get("primary") if isinstance(raw.get("names"), dict) else None,
    )

    # Address extraction (try multiple "equivalent" representations)
    addr = _extract_addr_from_osm_style(raw)
    if not any(addr.values()):
        addr.update(_extract_addr_from_overture(raw))
    if not any(addr.values()):
        addr.update(_extract_addr_from_openaddresses(raw))

    formatted_address = _build_formatted_address(addr, raw)
    address_components = _address_components(addr)

    # Contact info
    website = _normalize_website(
        _first_str(
            raw.get("website"),
            raw.get("contact:website"),
            (raw.get("websites") or [None])[0] if isinstance(raw.get("websites"), list) and raw.get("websites") else None,
        )
    )

    phone = _normalize_phone(
        _first_str(
            raw.get("international_phone_number"),
            raw.get("phone"),
            raw.get("contact:phone"),
            raw.get("telephone"),
            (raw.get("phones") or [None])[0] if isinstance(raw.get("phones"), list) and raw.get("phones") else None,
        )
    )

    # Opening hours (best-effort)
    weekday_text: list[str] = []
    open_now: Optional[bool] = None
    oh = raw.get("opening_hours")
    if isinstance(oh, dict):
        wt = oh.get("weekday_text")
        if isinstance(wt, list):
            weekday_text = [str(x) for x in wt if _none_if_blank(x)]
        on = oh.get("open_now")
        if isinstance(on, bool) or on is None:
            open_now = on
    else:
        oh_str = _first_str(oh)
        if oh_str:
            weekday_text = [oh_str]

    # Photos
    photos: list[dict] = []
    photo_url = _first_str(raw.get("photo_url"))
    if photo_url:
        photos = [
            {
                "photo_reference": photo_url,
                "width": None,
                "height": None,
            }
        ]

    canonical = {
        "name": name,
        "formatted_address": formatted_address,
        "address_components": address_components,
        "international_phone_number": phone,
        "website": website,
        "opening_hours": {
            "weekday_text": weekday_text,
            "open_now": open_now,
        },
        "photos": photos,
        "geometry": {
            "location": {
                "lat": lat if lat is not None else None,
                "lng": lon if lon is not None else None,
            }
        },
    }
    return canonical


def validate_canonical_metadata(canonical: dict) -> None:
    """
    Validate the canonical schema. Raises ValueError on hard failures.
    (Ingestion scripts are expected to catch and log warnings only.)
    """
    if not isinstance(canonical, dict):
        raise ValueError("canonical must be a dict")

    for key in ("name", "formatted_address", "geometry"):
        if key not in canonical:
            raise ValueError(f"canonical missing key: {key}")

    geom = canonical.get("geometry")
    if not isinstance(geom, dict):
        raise ValueError("canonical.geometry must be a dict")
    loc = geom.get("location")
    if not isinstance(loc, dict) or "lat" not in loc or "lng" not in loc:
        raise ValueError("canonical.geometry.location must include lat/lng")

    # Warning-only validations (callers should catch and log)
    formatted = canonical.get("formatted_address")
    if not isinstance(formatted, str) or not formatted.strip():
        logger.warning("canonical.formatted_address missing/blank")

    if canonical.get("international_phone_number") in (None, ""):
        logger.warning("canonical.international_phone_number missing/blank")

    if canonical.get("website") in (None, ""):
        logger.warning("canonical.website missing/blank")

