import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import RAW_FEATURES


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold=0.5) -> dict:
    proba = pipeline.predict_proba(X_test[RAW_FEATURES])[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(
            y_test, preds, output_dict=True, zero_division=0
        ),
    }


def load_pipeline(path: str):
    return joblib.load(path)
