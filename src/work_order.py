import json
from datetime import datetime, timezone


def build_work_order(
    record: dict,
    prediction: dict,
    drivers: list[dict],
    failure_modes: list[dict],
    guidance: list[str],
    model_version: str,
    threshold: float,
    model_name: str | None = None,
    dataset: str | None = None,
    engineered_features: dict | None = None,
    explanation_method: str | None = None,
    human_review_required: bool = False,
    tools_executed: list[dict] | None = None,
    disclaimer: str | None = None,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "dataset": dataset,
        "threshold": threshold,
        "telemetry": record,
        "sensor_inputs": record,
        "engineered_features": engineered_features or {},
        "prediction": {
            "failure_probability": prediction.get("failure_probability"),
            "predicted_failure": prediction.get("predicted_failure"),
            "status": prediction.get("status"),
            "note": "Uncalibrated model risk score, not a calibrated real-world probability.",
        },
        "explanation_drivers": drivers,
        "explanation_method": explanation_method,
        "failure_modes": failure_modes,
        "failure_mode_indicators": failure_modes,
        "failure_mode_method": (
            "Illustrative heuristic aligned to AI4I generation bands; "
            "not a trained classifier or certified diagnosis."
        ),
        "guidance": guidance,
        "human_review_required": human_review_required,
        "agent_tools_executed": tools_executed or [],
        "disclaimer": disclaimer
        or (
            "This system produces an uncalibrated model risk score based on "
            "synthetic educational data. It is not a certified industrial "
            "safety or maintenance system."
        ),
        "limitations": (
            "Synthetic UCI AI4I 2020 data; educational prototype only. "
            "Not a certified industrial safety or maintenance system."
        ),
    }


def work_order_json(work_order: dict) -> str:
    return json.dumps(work_order, indent=2)
