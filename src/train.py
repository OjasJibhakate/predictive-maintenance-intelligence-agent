import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import MODEL_FEATURES, RANDOM_SEED, RAW_FEATURES, TARGET
from .feature_engineering import PhysicsFeatures


def build_pipeline(model) -> Pipeline:
    steps = [("physics", PhysicsFeatures())]
    if isinstance(model, LogisticRegression):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def candidate_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=10,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def compare_models(X: pd.DataFrame, y: pd.Series, cv_folds: int = 3) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1"}
    results = {}
    for name, model in candidate_models().items():
        pipe = build_pipeline(model)
        scores = cross_validate(pipe, X[RAW_FEATURES], y, cv=cv, scoring=scoring)
        results[name] = {
            metric: float(scores[f"test_{metric}"].mean()) for metric in scoring
        }
    return results


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    pipeline_path: str,
    metadata: dict | None = None,
):
    model = candidate_models()[model_name]
    pipe = build_pipeline(model)
    pipe.fit(X_train[RAW_FEATURES], y_train)
    joblib.dump(pipe, pipeline_path)
    if metadata is not None:
        with open(metadata["path"], "w") as f:
            json.dump(
                {
                    **metadata["payload"],
                    "model": model_name,
                    "features": MODEL_FEATURES,
                    "seed": RANDOM_SEED,
                },
                f,
                indent=2,
            )
    return pipe


def threshold_report(
    y_true: pd.Series, proba, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6)
) -> list[dict]:
    rows = []
    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "f1": float(f1_score(y_true, preds, zero_division=0)),
                "false_negatives": int(((preds == 0) & (y_true == 1)).sum()),
                "false_positives": int(((preds == 1) & (y_true == 0)).sum()),
            }
        )
    return rows
