import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    COLUMN_ALIASES,
    METADATA_PATH,
    MODEL_FEATURES,
    MODEL_VERSION,
    PIPELINE_PATH,
    RANDOM_SEED,
    RAW_FEATURES,
    TARGET,
)
from src.evaluate import evaluate_model
from src.train import compare_models, threshold_report, train_final_model

FN_COST = 10
FP_COST = 1


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_ALIASES.get(c, str(c).strip()) for c in df.columns})


def load_dataset(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        print(f"Loading dataset from {path}")
        return normalize_columns(pd.read_csv(path))
    default = os.path.join("data", "ai4i2020.csv")
    if os.path.exists(default):
        print(f"Loading dataset from {default}")
        return normalize_columns(pd.read_csv(default))
    print("Local CSV not found, fetching UCI AI4I 2020 (id=601) via ucimlrepo...")
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=601)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = [str(c).strip() for c in df.columns]
    df = normalize_columns(df)
    os.makedirs("data", exist_ok=True)
    df.to_csv(default, index=False)
    print(f"Saved dataset to {default}")
    return df


def pick_threshold(report: list[dict]) -> dict:
    scored = [
        {**row, "cost": FN_COST * row["false_negatives"] + FP_COST * row["false_positives"]}
        for row in report
    ]
    scored.sort(key=lambda r: (r["cost"], -r["recall"]))
    return scored[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", help="Path to ai4i2020.csv")
    parser.add_argument("--model", default="xgboost", choices=["logistic_regression", "random_forest", "xgboost"])
    parser.add_argument("--auto", action="store_true", help="Pick best-recall model from CV comparison")
    args = parser.parse_args()

    df = load_dataset(args.data)
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing columns: {missing}\nFound: {list(df.columns)}")
    df = df.dropna(subset=RAW_FEATURES + [TARGET])
    print(f"Rows: {len(df)}, failures: {int(df[TARGET].sum())}")

    X = df[RAW_FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    print("Comparing models (3-fold CV)...")
    comparison = compare_models(X_train, y_train)
    for name, scores in comparison.items():
        print(f"  {name}: recall={scores['recall']:.3f} precision={scores['precision']:.3f} f1={scores['f1']:.3f}")

    model_name = args.model
    if args.auto:
        model_name = max(comparison, key=lambda n: (comparison[n]["recall"], comparison[n]["f1"]))
    print(f"Training final model: {model_name}")

    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
    pipe = train_final_model(X_train, y_train, model_name, PIPELINE_PATH)

    proba = pipe.predict_proba(X_test[RAW_FEATURES])[:, 1]
    report = threshold_report(y_test, proba)
    chosen = pick_threshold(report)
    print("Threshold analysis (FN cost=10, FP cost=1):")
    for row in report:
        cost = FN_COST * row["false_negatives"] + FP_COST * row["false_positives"]
        mark = " <-- selected" if row["threshold"] == chosen["threshold"] else ""
        print(
            f"  t={row['threshold']:.2f} recall={row['recall']:.3f} "
            f"precision={row['precision']:.3f} f1={row['f1']:.3f} "
            f"FN={row['false_negatives']} FP={row['false_positives']} cost={cost}{mark}"
        )

    metrics = evaluate_model(pipe, X_test, y_test, threshold=chosen["threshold"])
    print(f"Test @ t={chosen['threshold']:.2f}: recall={metrics['recall']:.3f} "
          f"precision={metrics['precision']:.3f} f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f}")

    metadata = {
        "model_version": MODEL_VERSION,
        "model": model_name,
        "threshold": chosen["threshold"],
        "threshold_assumptions": {"fn_cost": FN_COST, "fp_cost": FP_COST, "note": "Illustrative costs, no real maintenance cost data"},
        "features": MODEL_FEATURES,
        "seed": RANDOM_SEED,
        "dataset_rows": len(df),
        "test_metrics": metrics,
        "cv_comparison": comparison,
        "threshold_report": report,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "limitations": "Synthetic UCI AI4I 2020 data; educational prototype only.",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved pipeline -> {PIPELINE_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
