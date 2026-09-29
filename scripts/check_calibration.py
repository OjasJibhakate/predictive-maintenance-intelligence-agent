import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import METADATA_PATH, RAW_FEATURES, RANDOM_SEED, TARGET
from src.predict import predict_record

SCENARIOS = {
    "Baseline Operating Run": {"Air temperature": 298.0, "Process temperature": 308.0, "Rotational speed": 1500.0, "Torque": 40.0, "Tool wear": 50.0},
    "Simulate Dull Tool Wear": {"Air temperature": 298.0, "Process temperature": 309.0, "Rotational speed": 1400.0, "Torque": 55.0, "Tool wear": 250.0},
    "Simulate Thermal Conditions": {"Air temperature": 300.0, "Process temperature": 320.0, "Rotational speed": 1600.0, "Torque": 45.0, "Tool wear": 120.0},
    "Power-Related Scenario": {"Air temperature": 297.0, "Process temperature": 307.0, "Rotational speed": 2700.0, "Torque": 65.0, "Tool wear": 80.0},
}

df = pd.read_csv("data/ai4i2020.csv")
pipe = joblib.load("models/predictive_maintenance_pipeline.pkl")
with open(METADATA_PATH) as f:
    meta = json.load(f)
threshold = float(meta.get("threshold", 0.3))
print(f"model={meta.get('model')} threshold={threshold}")

X = df[RAW_FEATURES]
y = df[TARGET].astype(int)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
proba = pipe.predict_proba(X_test[RAW_FEATURES])[:, 1]
preds = (proba >= threshold).astype(int)

print("\n=== HOLDOUT METRICS @ t=%.2f ===" % threshold)
print(f"roc_auc={roc_auc_score(y_test, proba):.3f} pr_auc={average_precision_score(y_test, proba):.3f} "
      f"brier={brier_score_loss(y_test, proba):.4f}")
print(f"recall={recall_score(y_test, preds):.3f} precision={precision_score(y_test, preds):.3f} f1={f1_score(y_test, preds):.3f}")
print("confusion_matrix [tn fp / fn tp]:\n", confusion_matrix(y_test, preds))

print("\n=== CALIBRATION (predicted bin -> observed failure rate) ===")
frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
for mp, fp in zip(mean_pred, frac_pos):
    gap = mp - fp
    flag = " <-- OVERCONFIDENT" if gap > 0.15 else ""
    print(f"  mean_pred={mp:.3f} observed={fp:.3f} gap={gap:+.3f}{flag}")

print("\n=== SCENARIO PROBABILITIES ===")
for name, rec in SCENARIOS.items():
    r = predict_record(pipe, rec, threshold=threshold)
    print(f"  {name}: proba={r['failure_probability']:.3f} status={r['status']}")

print("\n=== WHY IS POWER SCENARIO 99%? ===")
print("Dataset context: PWF cases in AI4I are driven by extreme power (<3500W or >9000W).")
print(f"  scenario power = 65*2700*2pi/60 = {65*2700*2*3.14159/60:.0f}W (way above normal ~4-9kW band)")
print(f"  dataset power P1/P99: {((df['Torque']*df['Rotational speed']*2*3.14159/60).quantile([0.01,0.99])).to_dict()}")
print(f"  PWF rate @ power>9000W: {df[(df['Torque']*df['Rotational speed']*2*3.14159/60)>9000]['Machine failure'].mean():.3f} "
      f"(n={(df['Torque']*df['Rotational speed']*2*3.14159/60>9000).sum()})")
print(f"  PWF rate @ power 3000-9000W: {df[((df['Torque']*df['Rotational speed']*2*3.14159/60)>=3000)&((df['Torque']*df['Rotational speed']*2*3.14159/60)<=9000)]['Machine failure'].mean():.4f}")
print("\nConclusion: 99% is UNCALIBRATED model confidence on an extreme out-of-distribution-ish input,")
print("not a calibrated real-world failure probability. Wording fix needed in UI.")
