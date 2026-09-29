import pandas as pd

from .config import FAILURE_MODES, RAW_FEATURES, RISK_BANDS
from .data_preprocessing import coerce_record, validate_inputs


def risk_band(probability: float) -> tuple[str, str]:
    for cutoff, label, color in RISK_BANDS:
        if probability < cutoff:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def predict_record(pipeline, record: dict, threshold=0.5) -> dict:
    errors = validate_inputs(record)
    if errors:
        return {"ok": False, "errors": errors}
    clean = coerce_record(record)
    frame = pd.DataFrame([clean])
    proba = float(pipeline.predict_proba(frame[RAW_FEATURES])[0, 1])
    label, _ = risk_band(proba)
    failed = bool(proba >= threshold)
    return {
        "ok": True,
        "inputs": clean,
        "failure_probability": proba,
        "threshold": threshold,
        "predicted_failure": failed,
        "status": "HIGH RISK" if failed else label,
    }


def heuristic_failure_modes(record: dict, context: dict) -> list[dict]:
    wear = record["Tool wear"]
    delta = context.get("temperature_delta", 0.0)
    power = context.get("mechanical_power", 0.0)
    torque = record["Torque"]
    scores = {
        "TWF": min(0.95, wear / 240.0),
        "HDF": min(0.95, max(0.0, 1.0 - abs(delta - 8.1) / 1.5) * 0.9 + 0.05),
        "PWF": min(0.95, max(0.0, power - 8000.0) / 8000.0 + 0.05),
        "OSF": min(0.95, max(0.0, torque - 40.0) / 40.0 * 0.7 + wear / 400.0),
        "RNF": 0.05,
    }
    base = context.get("failure_probability", 0.0)
    predicted = context.get("predicted_failure", base >= 0.5)
    floor = 0.6 * base if predicted else 0.0
    modes = []
    for mode in FAILURE_MODES:
        confidence = round(float(min(0.99, max(floor, 0.4 * scores[mode] + 0.6 * base))), 3)
        modes.append({"mode": mode, "confidence": confidence})
    return sorted(modes, key=lambda item: item["confidence"], reverse=True)
