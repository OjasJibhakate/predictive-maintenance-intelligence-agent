import pandas as pd

from src.config import FAILURE_MODE_LABELS, RAW_FEATURES
from src.data_preprocessing import coerce_record, validate_inputs
from src.explainability import shap_explanation
from src.predict import heuristic_failure_modes, risk_band
from src.recommendations import build_guidance
from src.work_order import build_work_order, work_order_json


class ToolError(Exception):
    pass


def validate_sensor_inputs(record: dict) -> dict:
    try:
        errors = validate_inputs(record)
    except Exception as exc:
        raise ToolError(f"validation crashed: {exc}") from exc
    if errors:
        raise ToolError("; ".join(errors))
    try:
        clean = coerce_record(record)
    except Exception as exc:
        raise ToolError(f"could not coerce inputs to numbers: {exc}") from exc
    return {"clean_record": clean}


def engineer_machine_features(clean_record: dict) -> dict:
    try:
        torque = float(clean_record["Torque"])
        rpm = float(clean_record["Rotational speed"])
        power = torque * rpm * (2 * 3.141592653589793 / 60.0)
        delta = float(clean_record["Process temperature"]) - float(clean_record["Air temperature"])
        interaction = float(clean_record["Tool wear"]) * torque
    except Exception as exc:
        raise ToolError(f"feature engineering failed: {exc}") from exc
    return {
        "engineered": {
            "Mechanical Power W": power,
            "Temperature Delta K": delta,
            "Wear-Torque Interaction": interaction,
        },
        "formulas": {
            "Mechanical Power W": "Torque x RPM x 2pi / 60",
            "Temperature Delta K": "Process temperature - Air temperature",
            "Wear-Torque Interaction": "Tool wear x Torque",
        },
    }


def predict_failure_risk(pipeline, clean_record: dict, threshold: float) -> dict:
    try:
        frame = pd.DataFrame([{k: float(clean_record[k]) for k in RAW_FEATURES}])
        proba = float(pipeline.predict_proba(frame[RAW_FEATURES])[0, 1])
    except Exception as exc:
        raise ToolError(f"model inference failed: {exc}") from exc
    return {
        "failure_probability": proba,
        "predicted_failure": bool(proba >= threshold),
        "threshold": threshold,
    }


def classify_risk_level(failure_probability: float, predicted_failure: bool) -> dict:
    label, _ = risk_band(failure_probability)
    status = "HIGH RISK" if predicted_failure else label
    return {
        "status": status,
        "band": label,
        "human_review_required": bool(predicted_failure),
    }


def analyze_failure_modes(
    clean_record: dict,
    engineered: dict,
    failure_probability: float,
    predicted_failure: bool,
) -> dict:
    try:
        context = {
            "mechanical_power": engineered["Mechanical Power W"],
            "temperature_delta": engineered["Temperature Delta K"],
            "failure_probability": failure_probability,
            "predicted_failure": predicted_failure,
        }
        modes = heuristic_failure_modes(clean_record, context)
    except Exception as exc:
        raise ToolError(f"failure-mode analysis failed: {exc}") from exc
    labeled = [
        {
            "mode": m["mode"],
            "meaning": FAILURE_MODE_LABELS.get(m["mode"], m["mode"]),
            "confidence": m["confidence"],
        }
        for m in modes
    ]
    return {
        "modes": labeled,
        "method": (
            "Illustrative heuristic aligned to AI4I generation bands; "
            "not a trained classifier or certified diagnosis."
        ),
    }


def explain_prediction(pipeline, clean_record: dict, background=None) -> dict:
    try:
        result = shap_explanation(pipeline, clean_record, background=background)
    except Exception as exc:
        raise ToolError(f"explanation failed: {exc}") from exc
    return {
        "drivers": result["drivers"],
        "latency_ms": result["latency_ms"],
        "method": result.get("method", "unknown"),
    }


def generate_maintenance_guidance(
    prediction: dict, drivers: list, modes: list, human_review_required: bool
) -> dict:
    try:
        plain_modes = [
            {"mode": m["mode"], "confidence": m["confidence"]} for m in modes
        ]
        guidance = build_guidance(prediction, drivers, plain_modes)
    except Exception as exc:
        raise ToolError(f"guidance generation failed: {exc}") from exc
    if human_review_required:
        guidance.insert(
            0,
            "Human review required before any maintenance action: have qualified "
            "personnel confirm operating conditions against documentation. "
            "This system does not issue shutdown commands or control equipment.",
        )
    return {"guidance": guidance}


def generate_work_order(
    clean_record: dict,
    engineered: dict,
    prediction: dict,
    drivers: list,
    modes: list,
    guidance: list,
    explanation_method: str,
    human_review_required: bool,
    trace: list,
    model_name: str,
    model_version: str,
    dataset: str,
    threshold: float,
    disclaimer: str,
) -> dict:
    try:
        plain_modes = [
            {"mode": m["mode"], "confidence": m["confidence"]} for m in modes
        ]
        order = build_work_order(
            clean_record,
            prediction,
            drivers,
            plain_modes,
            guidance,
            model_version,
            threshold,
            model_name=model_name,
            dataset=dataset,
            engineered_features={k: round(float(v), 3) for k, v in engineered.items()},
            explanation_method=explanation_method,
            human_review_required=human_review_required,
            tools_executed=[
                {"step": t["step"], "tool": t["tool"], "status": t["status"]}
                for t in trace
            ],
            disclaimer=disclaimer,
        )
        return {"work_order": order, "json": work_order_json(order)}
    except Exception as exc:
        raise ToolError(f"work-order generation failed: {exc}") from exc
