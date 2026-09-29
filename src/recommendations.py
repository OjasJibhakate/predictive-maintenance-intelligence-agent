from .config import FAILURE_MODE_LABELS


DISCLAIMER = (
    "Educational prototype on synthetic data. "
    "Inspect according to approved procedures; not a certified maintenance instruction."
)


def build_guidance(prediction: dict, drivers: list[dict], failure_modes: list[dict]) -> list[str]:
    steps = []
    proba = prediction.get("failure_probability", 0.0)
    status = prediction.get("status", "NORMAL")
    top_modes = [m["mode"] for m in failure_modes[:2]] if failure_modes else []
    top_features = [d["feature"] for d in drivers[:3]] if drivers else []

    if status == "HIGH RISK":
        steps.append(
            f"Potential risk factor flagged ({proba:.1%} vs threshold "
            f"{prediction.get('threshold', 0.5):.2f}): schedule a review before continued operation."
        )
    elif status == "WATCHLIST":
        steps.append(
            f"Elevated watchlist signal ({proba:.1%}): increase monitoring frequency and log telemetry."
        )
    else:
        steps.append(
            f"No immediate failure signal ({proba:.1%}): continue routine monitoring."
        )

    for mode in top_modes:
        label = FAILURE_MODE_LABELS.get(mode, mode)
        if mode == "TWF":
            steps.append(
                f"{mode} ({label}) is a potential risk factor: check tool condition/cutting insert per documented procedures."
            )
        elif mode == "HDF":
            steps.append(
                f"{mode} ({label}) is a potential risk factor: review thermal conditions and cooling-related procedures."
            )
        elif mode == "PWF":
            steps.append(
                f"{mode} ({label}) is a potential risk factor: verify operating conditions and power-related inspection procedures."
            )
        elif mode == "OSF":
            steps.append(
                f"{mode} ({label}) is a potential risk factor: check load, torque, and wear conditions."
            )

    if top_features:
        steps.append(
            "Top model drivers for this prediction (review required): "
            + ", ".join(top_features)
            + "."
        )
    steps.append(DISCLAIMER)
    return steps
