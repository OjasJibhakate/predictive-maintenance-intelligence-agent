---
title: Predictive Maintenance Intelligence Agent
emoji: ⚙️
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# Predictive Maintenance Intelligence Agent

A tool-orchestrated maintenance agent: sensor inputs flow through input validation,
physics feature engineering, XGBoost risk prediction, risk decision, failure-mode
risk indicators, SHAP explanation, cautious maintenance guidance, and JSON work-order
generation, with a human-review gate on high-risk results.

> This system produces an uncalibrated model risk score based on synthetic educational
> data. It is not a certified industrial safety or maintenance system.

## Problem statement

Unplanned machine failures are expensive. This project demonstrates how an agentic
workflow can wrap a classical ML classifier with validation, explanation, and
human-in-the-loop safeguards to produce reviewable maintenance intelligence.

## Objectives

- Estimate failure risk from 5 operating-condition inputs
- Explain which features drove each prediction (SHAP)
- Flag illustrative failure-mode risk indicators (TWF/HDF/PWF/OSF/RNF)
- Generate cautious, review-gated maintenance guidance
- Export a traceable JSON work order with the full agent trace

## Agentic AI relevance

The subject is Agentic AI & Automation. This project is honestly a **tool-orchestrated
AI workflow** (deterministic orchestrator + real tool functions), not an autonomous
LLM-reasoning agent. The `agent/` package coordinates 8 tools through 9 workflow
states; every step executes real Python code and its output feeds the next step.

## System architecture

```
sensor inputs -> Maintenance Agent (agent/orchestrator.py)
  -> validate_sensor_inputs      (INPUT_VALIDATION)
  -> engineer_machine_features   (FEATURE_ENGINEERING)
  -> predict_failure_risk        (PREDICTION, existing XGBoost pipeline)
  -> classify_risk_level         (RISK_ASSESSMENT + threshold decision)
  -> analyze_failure_modes       (FAILURE_ANALYSIS, illustrative heuristic)
  -> explain_prediction          (EXPLANATION, SHAP TreeExplainer + fallback)
  -> generate_maintenance_guidance (GUIDANCE, cautious + review gate)
  -> generate_work_order         (WORK_ORDER + HUMAN_REVIEW gate)
  -> Gradio UI: risk gauge, Agent Execution Trace, indicators, SHAP, guidance, JSON
```

## Agent tools (`agent/tools.py`)

1. `validate_sensor_inputs()` — types, missing values, plausible ranges
2. `engineer_machine_features()` — Mechanical Power, Temperature Delta, Wear-Torque Interaction
3. `predict_failure_risk()` — existing XGBoost pipeline, same preprocessing, configurable threshold
4. `classify_risk_level()` — NORMAL / WATCHLIST / HIGH RISK + human-review flag
5. `analyze_failure_modes()` — TWF/HDF/PWF/OSF/RNF indicators (heuristic, labeled as such)
6. `explain_prediction()` — SHAP TreeExplainer with feature-importance fallback (method reported)
7. `generate_maintenance_guidance()` — cautious steps + review banner on high risk
8. `generate_work_order()` — JSON with telemetry, engineered features, risk, threshold, modes, drivers, guidance, review flag, tool trace, disclaimer

## Dataset

UCI AI4I 2020 Predictive Maintenance Dataset (synthetic, 10,000 rows, 339 failures).
Primary target: binary `Machine failure`. Failure-mode columns (TWF/HDF/PWF/OSF/RNF)
are used as illustrative indicators only.

## Feature engineering

- Mechanical Power = Torque x RPM x 2pi / 60
- Temperature Delta = Process temperature - Air temperature
- Wear-Torque Interaction = Tool wear x Torque

Computed inside the sklearn pipeline (`PhysicsFeatures`) so training and inference
stay consistent; the agent recomputes the same values for display/trace.

## XGBoost model

Leakage-free pipeline: physics transformer -> XGBoost (400 trees, depth 6, lr 0.05,
scale_pos_weight 10). Model comparison on 3-fold CV: XGBoost F1 0.809, RF 0.791,
Logistic 0.273. Model name, version (`v1.0-baseline`), threshold, and metrics are
shown in the UI identity card and stored in `models/pipeline_metadata.json`.

## SHAP explainability

`shap.TreeExplainer` on the fitted XGBoost model; falls back to normalized feature
importance if SHAP is unavailable (the active method is displayed). Text states that
feature importance does not establish causation or guarantee physical failure.

## Risk threshold

Default 0.5 is not assumed optimal. Validation sweep over {0.2..0.6} with an
illustrative cost model (FN cost 10, FP cost 1) selected **t=0.30**. Shown in the UI
with assumptions; configurable via retraining.

## Gradio interface

Title + agent description, model identity card, calibration disclaimer, demonstration
scenario selector (auto-fills inputs), sensor inputs, Analyze Health button, risk
result + Plotly gauge, **Agent Execution Trace** (tool, status, result, timing),
Failure-Mode Risk Indicators, SHAP chart + method note, Maintenance Guidance with
human-review banner, downloadable JSON work order, disclaimer section.

## JSON work order

Timestamp, model name/version, dataset, sensor inputs, engineered features, risk
score + status + threshold, failure-mode indicators + method note, top feature
drivers + explanation method, guidance, human-review flag, agent tools executed,
disclaimer. Validated in `scripts/verify_agent.py`.

## Evaluation metrics (XGBoost, holdout 20%)

- CV (3-fold): XGBoost F1 0.809, RF 0.791, Logistic 0.273
- Test @ t=0.30: recall 0.824, precision 0.737, F1 0.778, ROC-AUC 0.983, PR-AUC 0.861, Brier 0.011
- Confusion matrix: TN 1912, FP 20, FN 12, TP 56
- Probability outputs are uncalibrated model scores, not calibrated real-world chances

## Limitations

- Synthetic data; educational prototype, not a certified safety or maintenance system
- Failure-mode indicators are heuristic, not a trained multi-label classifier
- No real-time sensor integration, no equipment control, no shutdown commands
- No probability calibration; high scores (e.g. 99%) are model confidence on extremes

## Installation

```bash
pip install -r requirements.txt
pip install ucimlrepo   # only needed to auto-fetch the dataset
```

## Run locally

```bash
python scripts/train_pipeline.py --model xgboost   # uses data/ai4i2020.csv or fetches UCI id=601
python scripts/verify_agent.py                     # runs all agent checks
python app.py                                      # open http://127.0.0.1:7860
```

## Future improvements

- Probability calibration (Platt/isotonic) with reliability curves in the UI
- Trained multi-label failure-mode classifier replacing the heuristic
- Input-drift monitoring and prediction logging to a database
- Real sensor/API ingestion and Remaining Useful Life estimation

## Project structure

- `app.py` — Gradio dashboard wired to the agent orchestrator
- `agent/__init__.py`, `agent/orchestrator.py`, `agent/tools.py` — agent workflow
- `src/` — validation, physics features, training, evaluation, inference, SHAP, guidance, work orders
- `scripts/train_pipeline.py`, `scripts/check_calibration.py`, `scripts/verify_agent.py` — training + checks
- `notebooks/` — 01_eda, 02_baseline_models, 03_threshold_analysis
- `models/` — trained XGBoost pipeline + metadata (generated)
- `data/ai4i2020.csv` — dataset (generated/fetched)

## Resume bullets

- Built a tool-orchestrated maintenance agent on UCI AI4I 2020: 8 cooperating tools across validation, physics features, XGBoost inference, risk decision, SHAP explanation, and review-gated work orders
- XGBoost reached F1 0.78, ROC-AUC 0.98 on holdout; cost-sensitive threshold tuning (t=0.30, recall 0.82)
- Shipped a Gradio dashboard with live agent execution trace, Plotly risk gauge, failure-mode indicators, and downloadable JSON work orders
