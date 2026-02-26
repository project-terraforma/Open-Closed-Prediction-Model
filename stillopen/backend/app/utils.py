"""
app/utils.py — Shared application-level utilities.
"""

def build_address(metadata: dict) -> str:
    """
    Compose a human-readable address from OSM addr:* fields or a plain 'address' field.
    """
    # If a pre-built full address exists, use it directly
    if metadata.get("address"):
        return metadata["address"].strip()
    if metadata.get("full_address"):
        return metadata["full_address"].strip()
        
    parts = []

    # Street-level: number + street
    house = (metadata.get("addr:housenumber") or "").strip()
    street = (metadata.get("addr:street") or "").strip()
    if house and street:
        parts.append(f"{house} {street}")
    elif street:
        parts.append(street)

    # City / municipality
    city = (
        metadata.get("addr:city")
        or metadata.get("addr:municipality")
        or metadata.get("city")
        or ""
    ).strip()
    if city:
        parts.append(city)

    # State / province
    state = (metadata.get("addr:state") or metadata.get("state") or "").strip()
    if state:
        parts.append(state)

    # Postcode
    postcode = str(metadata.get("addr:postcode") or metadata.get("postcode") or "").strip()
    if postcode:
        parts.append(postcode)

    return ", ".join(parts) if parts else ""
