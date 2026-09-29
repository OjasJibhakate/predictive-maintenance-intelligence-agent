import time

from .tools import (
    ToolError,
    analyze_failure_modes,
    classify_risk_level,
    engineer_machine_features,
    explain_prediction,
    generate_maintenance_guidance,
    generate_work_order,
    predict_failure_risk,
    validate_sensor_inputs,
)

STATES = [
    "INPUT_VALIDATION",
    "FEATURE_ENGINEERING",
    "PREDICTION",
    "RISK_ASSESSMENT",
    "FAILURE_ANALYSIS",
    "EXPLANATION",
    "GUIDANCE",
    "WORK_ORDER",
    "HUMAN_REVIEW",
]

DISCLAIMER = (
    "This system produces an uncalibrated model risk score based on "
    "synthetic educational data. It is not a certified industrial "
    "safety or maintenance system."
)


def _record(trace: list, step: str, tool: str, status: str, detail: str, elapsed_ms: float | None):
    trace.append(
        {
            "step": step,
            "tool": tool,
            "status": status,
            "detail": detail,
            "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms is not None else None,
        }
    )


def run_maintenance_agent(pipeline, record: dict, threshold: float, background=None,
                          model_name="xgboost", model_version="v1.0-baseline",
                          dataset="UCI AI4I 2020") -> dict:
    trace: list[dict] = []
    context: dict = {
        "threshold": threshold,
        "model_name": model_name,
        "model_version": model_version,
        "dataset": dataset,
        "disclaimer": DISCLAIMER,
    }

    def attempt(step: str, tool: str, fn, *args, **kwargs):
        started = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
        except ToolError as exc:
            _record(trace, step, tool, "failed", str(exc),
                    (time.perf_counter() - started) * 1000.0)
            raise
        _record(trace, step, tool, "ok", _summarize(step, out),
                (time.perf_counter() - started) * 1000.0)
        return out

    try:
        validated = attempt("INPUT_VALIDATION", "validate_sensor_inputs",
                            validate_sensor_inputs, record)
        context["clean_record"] = validated["clean_record"]

        engineered = attempt("FEATURE_ENGINEERING", "engineer_machine_features",
                             engineer_machine_features, context["clean_record"])
        context["engineered"] = engineered["engineered"]
        context["formulas"] = engineered["formulas"]

        predicted = attempt("PREDICTION", "predict_failure_risk",
                            predict_failure_risk, pipeline,
                            context["clean_record"], threshold)
        prediction = {
            "failure_probability": predicted["failure_probability"],
            "predicted_failure": predicted["predicted_failure"],
            "threshold": threshold,
        }
        context["prediction"] = prediction

        risk = attempt("RISK_ASSESSMENT", "classify_risk_level",
                       classify_risk_level,
                       predicted["failure_probability"],
                       predicted["predicted_failure"])
        prediction["status"] = risk["status"]
        context["human_review_required"] = risk["human_review_required"]

        modes = attempt("FAILURE_ANALYSIS", "analyze_failure_modes",
                        analyze_failure_modes, context["clean_record"],
                        context["engineered"],
                        predicted["failure_probability"],
                        predicted["predicted_failure"])
        context["failure_modes"] = modes["modes"]
        context["failure_mode_method"] = modes["method"]

        explained = attempt("EXPLANATION", "explain_prediction",
                            explain_prediction, pipeline,
                            context["clean_record"], background)
        context["drivers"] = explained["drivers"]
        context["explanation_latency_ms"] = explained["latency_ms"]
        context["explanation_method"] = explained["method"]

        guided = attempt("GUIDANCE", "generate_maintenance_guidance",
                         generate_maintenance_guidance, prediction,
                         context["drivers"], context["failure_modes"],
                         context["human_review_required"])
        context["guidance"] = guided["guidance"]

        ordered = attempt("WORK_ORDER", "generate_work_order",
                          generate_work_order, context["clean_record"],
                          context["engineered"], prediction,
                          context["drivers"], context["failure_modes"],
                          context["guidance"], context["explanation_method"],
                          context["human_review_required"], trace,
                          model_name, model_version, dataset, threshold,
                          DISCLAIMER)
        context["work_order"] = ordered["work_order"]
        context["work_order_json"] = ordered["json"]

        _record(trace, "HUMAN_REVIEW", "human_review_gate",
                "ok" if context["human_review_required"] else "skipped",
                "Qualified human review required before any maintenance action."
                if context["human_review_required"]
                else "Low risk: continue routine monitoring; no review gate triggered.",
                None)
        context["ok"] = True
    except ToolError as exc:
        context["ok"] = False
        context["error"] = str(exc)
    context["trace"] = trace
    return context


def _summarize(step: str, out: dict) -> str:
    if step == "INPUT_VALIDATION":
        return "5 sensor inputs present, numeric, within plausible ranges"
    if step == "FEATURE_ENGINEERING":
        eng = out["engineered"]
        return (
            f"power={eng['Mechanical Power W']:.0f}W, "
            f"delta={eng['Temperature Delta K']:.1f}K, "
            f"wear-torque={eng['Wear-Torque Interaction']:.0f}"
        )
    if step == "PREDICTION":
        return (
            f"XGBoost score={out['failure_probability']:.1%} vs "
            f"threshold={out['threshold']:.2f} -> "
            f"{'FAIL' if out['predicted_failure'] else 'no-fail'}"
        )
    if step == "RISK_ASSESSMENT":
        extra = "; human review required" if out["human_review_required"] else ""
        return f"status={out['status']}{extra}"
    if step == "FAILURE_ANALYSIS":
        top = out["modes"][0] if out["modes"] else {}
        return (
            f"top indicator={top.get('mode')} {top.get('confidence', 0):.0%}; "
            "heuristic, not a certified diagnosis"
        )
    if step == "EXPLANATION":
        top = out["drivers"][0] if out["drivers"] else {}
        return (
            f"{out['method']}; top driver={top.get('feature')} "
            f"({top.get('contribution')}); {out['latency_ms']}ms"
        )
    if step == "GUIDANCE":
        return f"{len(out['guidance'])} cautious steps prepared"
    if step == "WORK_ORDER":
        return "JSON work order prepared with trace + disclaimer"
    return "done"
