import time

import numpy as np
import pandas as pd

from .config import DISPLAY_NAMES, MODEL_FEATURES, RAW_FEATURES


def shap_explanation(pipeline, record: dict, background: pd.DataFrame | None = None) -> dict:
    import shap

    started = time.perf_counter()
    model = pipeline.named_steps["model"]
    physics = pipeline.named_steps["physics"]
    frame = pd.DataFrame([{k: float(record[k]) for k in RAW_FEATURES}])
    transformed = physics.transform(frame)[MODEL_FEATURES]

    if background is not None and len(background) > 0:
        sample = background.sample(min(100, len(background)), random_state=42)
        bg = physics.transform(sample[RAW_FEATURES])[MODEL_FEATURES]
    else:
        bg = transformed

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values).reshape(-1)
        method = "shap_tree_explainer"
    except Exception:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            values = np.zeros(len(MODEL_FEATURES))
            method = "zero_fallback (no SHAP support and no feature importances)"
        else:
            values = np.asarray(importances, dtype=float)
            values = values / (values.sum() or 1.0)
            method = "feature_importance_fallback (SHAP unavailable)"

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    drivers = sorted(
        (
            {
                "feature": DISPLAY_NAMES.get(feat, feat),
                "contribution": round(float(val), 4),
            }
            for feat, val in zip(MODEL_FEATURES, values)
        ),
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )
    return {"drivers": drivers, "latency_ms": round(elapsed_ms, 2), "method": method}
