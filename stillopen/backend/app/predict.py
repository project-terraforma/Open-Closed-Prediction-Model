"""
predict.py — Thin wrapper around PlaceScorer for API consumption.

The old XGBoost ModelService has been replaced by a pure-Python signal-based
scorer (see scorer.py).  The external interface is unchanged:

  predict_status(place)        → dict with status, confidence, explanation
  predict_place(place_data)    → dict with status, confidence, raw_score, signals_fired
  predict_batch(place_list)    → list of dicts (no signals_fired for performance)
"""

from .scorer import scorer


def predict_place(place_data: dict) -> dict:
    """
    Score a single place metadata dict.
    Returns status, confidence (50–99), raw_score, and signals_fired.
    Called by /place/{id} endpoint (full detail view).
    """
    if not isinstance(place_data, dict):
        place_data = {}
    result = scorer.score(place_data)
    return {
        "status": result.status,
        "confidence": result.confidence,
        "raw_score": result.raw_score,
        "signals_fired": result.fired_signals,
        "explanation": result.explanation,
        "prediction_type": result.status,   # binary: "open" or "closed"
    }


def predict_status(place) -> dict:
    """
    Predict status for a single place.
    Accepts a place ORM object (with .metadata_json) or a plain dict.
    Returns dict compatible with the existing /place/{id} response shape.
    """
    place_data = place.metadata_json if hasattr(place, "metadata_json") else place
    if not isinstance(place_data, dict):
        place_data = {}
    return predict_place(place_data)


def predict_batch(place_data_list: list) -> list:
    """
    Batch predict for a list of place metadata dicts.
    signals_fired is intentionally omitted from batch results for performance.
    """
    results = []
    for place_data in place_data_list:
        if not isinstance(place_data, dict):
            place_data = {}
        result = scorer.score(place_data)
        results.append({
            "status": result.status,
            "confidence": result.confidence,
            "raw_score": result.raw_score,
            "prediction_type": result.status,
            "explanation": result.explanation,
        })
    return results
