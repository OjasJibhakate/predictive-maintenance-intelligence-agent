# Predictive Maintenance Intelligence Agent

**Detailed Project Report**

| Field | Detail |
|---|---|
| Project title | Predictive Maintenance Intelligence Agent |
| Subject | Agentic AI & Automation |
| Domain | Data Science, Machine Learning, Industrial Analytics |
| Author | Ojas Jibhakate |
| Academic year | 2026 |
| Repository | https://github.com/OjasJibhakate/predictive-maintenance-intelligence-agent |
| Live deployment | https://predictive-maintenance-agent-s9d6.onrender.com |
| Model version | v1.0-baseline (XGBoost, threshold 0.30) |

---

## Table of contents

1. Abstract
2. Problem statement
3. Objectives
4. Agentic AI relevance and scope statement
5. Dataset
6. Feature engineering
7. Modeling strategy
8. Cost-sensitive threshold selection
9. Explainability (SHAP)
10. Failure-mode risk indicators
11. Agent architecture: tools and workflow states
12. Human-in-the-loop review
13. Gradio user interface
14. JSON work order
15. Evaluation results and probability calibration
16. Demonstration scenarios
17. Testing and validation performed
18. Deployment
19. Project structure and file inventory
20. Installation and local execution
21. Technology stack
22. Limitations
23. Future improvements
24. Viva preparation
25. References

---

## 1. Abstract

Unplanned industrial machine failures cause production loss and expensive downtime. This
project builds an end-to-end, deployable web application that estimates machine failure
risk from five operating-condition sensor readings, explains which inputs drove each
prediction, flags illustrative failure-mode risk indicators, generates cautious
maintenance-oriented guidance, and exports a traceable JSON work order.

The system is architecturally an **agentic AI workflow**: a maintenance agent
(orchestrator) coordinates eight discrete tools across nine explicit workflow states
(INPUT_VALIDATION through HUMAN_REVIEW). Prediction is performed by a leakage-free
scikit-learn pipeline containing a physics-feature transformer and an XGBoost
classifier, trained on the synthetic UCI AI4I 2020 Predictive Maintenance dataset
(10,000 records). The operating threshold (0.30) was selected through a documented
cost-sensitive analysis rather than the default 0.5.

On a held-out 20% test split, the deployed model achieves recall 0.824, precision
0.737, F1 0.778, ROC-AUC 0.983, and PR-AUC 0.861. The application is deployed publicly
on Render and is reachable at the URL listed on the cover of this report.

---

## 2. Problem statement

Machine failures in industrial settings are costly and often surprise operators.
Sensor telemetry (temperature, rotational speed, torque, tool wear) contains
statistical patterns that precede failure, but raw readings alone do not tell a
maintenance team what to inspect or how urgently. Three gaps motivate this project:

1. **Prediction gap** — translating multi-sensor telemetry into a single calibrated
   risk signal that supports a maintenance decision.
2. **Trust gap** — black-box predictions are not actionable for maintenance teams;
   the drivers behind each prediction must be visible.
3. **Process gap** — a risk score alone does not produce a traceable work item with
   documented assumptions and limitations, and it must never auto-command machinery.

This project addresses all three by wrapping a trained classifier inside an
agentic workflow with validation, explanation, guidance, a human-review gate, and
work-order export.

---

## 3. Objectives

| ID | Objective | Status |
|---|---|---|
| O1 | Accept five sensor inputs with validation (types, missing values, plausible ranges) | Done |
| O2 | Apply physics-inspired feature engineering consistently at training and inference | Done |
| O3 | Train and compare multiple classifiers with leakage-free preprocessing | Done |
| O4 | Select an operating threshold through documented cost-sensitive analysis | Done |
| O5 | Explain individual predictions with SHAP, with an honest fallback path | Done |
| O6 | Produce failure-mode risk indicators, clearly labeled as non-diagnostic | Done |
| O7 | Generate cautious, review-gated maintenance guidance | Done |
| O8 | Export a traceable JSON work order including the agent tool trace | Done |
| O9 | Deploy a public web interface usable by a professor or recruiter | Done |
| O10 | Report only measured values; document limitations transparently | Done |

---

## 4. Agentic AI relevance and scope statement

The subject requires an agentic workflow. This project is honestly described as a
**tool-orchestrated AI workflow** — a deterministic maintenance agent that plans and
sequences eight real tool invocations through nine workflow states, branches on its
own prediction result, and enforces a human-review gate. It does **not** contain an
LLM reasoning loop, autonomous planning, or self-modification, and the documentation
never claims otherwise.

Agentic properties that are genuinely present:

- **Tool orchestration** — eight modular tools executed in dependency order, each a
  real Python function whose output feeds the next stage.
- **Conditional decision flow** — the risk decision determines the guidance and the
  human-review gate state; low-risk runs skip the review gate by design.
- **Structured result synthesis** — all tool outputs are combined into a single
  structured result and work order.
- **Traceability** — the agent emits a per-step execution trace with real timings and
  status codes that is displayed in the UI and embedded in the work order.
- **Graceful failure handling** — tool errors halt the workflow at the correct state,
  are surfaced to the user, and are never hidden or faked.

---

## 5. Dataset

**UCI AI4I 2020 Predictive Maintenance Dataset** (UCI ML Repository ID 601).

| Property | Value |
|---|---|
| Rows | 10,000 |
| Failure records | 339 (3.39% positive class) |
| Failure modes | TWF, HDF, PWF, OSF, RNF (multi-label, overlapping) |
| Primary target | `Machine failure` (binary) |
| Nature | Synthetic data generated from documented rules |
| Train/test split | 80/20 stratified, random seed 42 |

Features used for prediction: air temperature [K], process temperature [K],
rotational speed [rpm], torque [Nm], tool wear [min]. Failure-mode columns are used
only for the illustrative indicators described in section 10.

Loading logic: if `data/ai4i2020.csv` exists it is used directly; otherwise the
pipeline fetches UCI ID 601 through `ucimlrepo` and saves a local copy. Column names
are normalized (bare names are canonical; unit-suffixed aliases are mapped).

---

## 6. Feature engineering

Three physics-inspired features are computed **inside the sklearn pipeline**
(`src/feature_engineering.py`, class `PhysicsFeatures`), guaranteeing identical
computation during training and inference:

| Feature | Formula | Purpose |
|---|---|---|
| Mechanical Power W | Torque × RPM × (2π / 60) | Combined torque and rotational-speed condition |
| Temperature Delta K | Process temperature − Air temperature | Thermal stress between process and environment |
| Wear-Torque Interaction | Tool wear × Torque | Combined tool wear and mechanical load |

The agent's feature-engineering tool recomputes the same values explicitly so they can
be displayed and traced without duplicating model logic.

**Leakage prevention.** The `PhysicsFeatures` transformer is a stateless step inside
the fitted pipeline; no statistic is learned from validation or test data. Splitting
happens before any fitting, and the same fitted pipeline object serves both evaluation
and production inference.

---

## 7. Modeling strategy

Three candidate models were compared with 3-fold stratified cross-validation on the
training split, each wrapped in the same pipeline:

| Model | CV Recall | CV Precision | CV F1 |
|---|---|---|---|
| Logistic Regression (class-balanced, scaled) | 0.845 | 0.163 | 0.273 |
| Random Forest (300 trees, balanced subsample) | 0.686 | 0.935 | 0.791 |
| **XGBoost (selected)** | **0.786** | **0.834** | **0.809** |

XGBoost was selected as the final model: 400 estimators, max depth 6, learning rate
0.05, subsample 0.9, colsample 0.9, `scale_pos_weight` 10, seed 42. It offers the best
balance of recall and precision for an imbalanced failure task, with calibrated
probability output available for threshold analysis.

**Why not accuracy:** failures are 3.39% of records; a model that always predicts
"no failure" would score 96.6% accuracy while catching nothing. Recall, precision, F1,
ROC-AUC and PR-AUC are the meaningful metrics here.

Notebooks accompanying the pipeline: `01_eda.ipynb` (exploration, class balance,
physics features), `02_baseline_models.ipynb` (CV comparison), `03_threshold_analysis.ipynb`
(cost sweep).

---

## 8. Cost-sensitive threshold selection

The default 0.5 threshold was treated as a hypothesis, not a default. A sweep over
{0.20, 0.30, 0.40, 0.50, 0.60} on validation data used an illustrative cost model:
a missed failure costs 10 units, a false alarm costs 1 unit (assumption documented;
no real maintenance cost data exists for this prototype).

| Threshold | Recall | Precision | F1 | FN | FP | Illustrative cost |
|---|---|---|---|---|---|---|
| 0.20 | 0.824 | 0.683 | 0.747 | 12 | 26 | 146 |
| **0.30 (selected)** | **0.824** | **0.737** | **0.778** | **12** | **20** | **140** |
| 0.40 | 0.809 | 0.786 | 0.797 | 13 | 15 | 145 |
| 0.50 | 0.779 | 0.815 | 0.797 | 15 | 12 | 162 |
| 0.60 | 0.765 | 0.839 | 0.800 | 16 | 10 | 170 |

Threshold 0.30 minimises the illustrative cost while holding false negatives at 12.
The threshold is stored in `models/pipeline_metadata.json`, displayed in the UI header
and on the gauge, and is configurable by retraining.

---

## 9. Explainability (SHAP)

`explain_prediction` runs `shap.TreeExplainer` on the fitted XGBoost model and returns
per-feature SHAP contributions for the current input, sorted by absolute magnitude. The
UI renders the top eight drivers as a horizontal bar chart with readable feature names.

Honest behavior on fallback: if SHAP cannot run, the tool falls back to normalized
feature importances and **reports which method was used** in both the UI note and the
JSON work order. During development a real defect was found and fixed where SHAP
silently fell back due to an incorrect `data=` argument to `TreeExplainer`; after the
fix the measured explanation latency is approximately 20–35 ms in local testing, and
the active method string (`shap_tree_explainer`) is displayed to the user.

The UI text states explicitly: "These features had the strongest influence on this
particular model prediction. Feature importance does not establish causation or
guarantee a physical failure."

---

## 10. Failure-mode risk indicators

Five indicators are shown for every analysis:

| Code | Meaning | Illustrative signal basis |
|---|---|---|
| TWF | Tool wear failure | tool wear level |
| HDF | Heat dissipation failure | temperature delta near the AI4I HDF band (≈8.1 K ± 1.5) |
| PWF | Power failure | mechanical power above the normal operating band |
| OSF | Overstrain failure | torque and wear combination |
| RNF | Random failure | fixed low baseline |

Basis of the method: the AI4I 2020 dataset is synthetic and was generated from
documented rules; the indicator scores were aligned to those documented bands during
development (for example, HDF records in the dataset occur at a temperature delta of
7.6–8.6 K, so the indicative band is centered there). Each score is combined with the
model's overall risk score, and the top indicators are echoed in the guidance.

This is labeled in the UI, README, and work order as: **"Model-derived indicators.
Multiple modes may be flagged simultaneously. They are not certified diagnoses. Heuristic
method aligned to AI4I generation bands."** It is not a trained multi-label classifier
and is never presented as confirmed physical failure.

---

## 11. Agent architecture: tools and workflow states

Implementation: `agent/orchestrator.py` (workflow engine) and `agent/tools.py` (eight
tools). The workflow executes in the following order; each step records status, a short
human-readable result summary, and real measured execution time in the trace.

| # | State | Tool | Function |
|---|---|---|---|
| 1 | INPUT_VALIDATION | validate_sensor_inputs | Types, missing values, plausible ranges |
| 2 | FEATURE_ENGINEERING | engineer_machine_features | Mechanical Power, Temperature Delta, Wear-Torque Interaction |
| 3 | PREDICTION | predict_failure_risk | XGBoost pipeline inference, configurable threshold |
| 4 | RISK_ASSESSMENT | classify_risk_level | NORMAL / WATCHLIST / HIGH RISK + human-review flag |
| 5 | FAILURE_ANALYSIS | analyze_failure_modes | TWF-HDF-PWF-OSF-RNF indicators |
| 6 | EXPLANATION | explain_prediction | SHAP TreeExplainer with reported fallback |
| 7 | GUIDANCE | generate_maintenance_guidance | Cautious steps + review banner on high risk |
| 8 | WORK_ORDER | generate_work_order | JSON with full provenance and tool trace |
| 9 | HUMAN_REVIEW | human_review_gate | Gate status (ok/skipped) recorded in trace |

All tools are real Python functions — no simulated or text-only tool calls. Tool
failures raise `ToolError`, halt the workflow at the failing state, and are surfaced
in the UI as "Agent stopped at <STATE>: <reason>" with the trace showing exactly which
step failed. No fabricated outputs or execution times are ever shown.

---

## 12. Human-in-the-loop review

If the model score is at or above the operating threshold, the agent sets
`human_review_required = True`, and the UI displays a red **"Human review required"**
alert. Guidance in that state begins with: qualified maintenance personnel must confirm
operating conditions against approved documentation before any action; the system does
not issue shutdown commands and does not control equipment. On low risk the gate is
recorded as `skipped` with "continue routine monitoring" text. The review flag is also
embedded in the JSON work order.

---

## 13. Gradio user interface

A custom-styled (dark industrial theme, amber accent) single-page dashboard:

| Region | Contents |
|---|---|
| Header | Product identity, agent description, metric pills (model, version, threshold, recall, precision, F1, ROC-AUC, dataset), calibration disclaimer |
| Control bar | Demonstration scenario selector (auto-fills inputs on change), Load scenario, Analyze health (primary action) |
| Sensor inputs panel | Grouped thermal / mechanical load / wear inputs (5 numbers) |
| Risk result panel | Color-coded status pill, large uncalibrated probability card, Plotly gauge with threshold tick |
| Agent execution trace | Vertical timeline: state, tool name, status, result summary, real timing |
| Failure-mode panel | Styled table: mode code, meaning, confidence bar and value |
| SHAP panel | Top-driver bar chart + method note with latency |
| Guidance panel | Human-review alert (red/green) + numbered cautious guidance |
| Work order panel | Downloadable JSON work order |
| Footer | Disclaimer and limitations |

The empty state, invalid-input state, and no-model state are all handled with explicit
messages rather than silent failure. Scenario labels carry "(demo)" and the disclaimer
states scenarios are illustrative, not validated real-world operating conditions.

---

## 14. JSON work order

Generated per analysis (`src/work_order.py`), with fields:

`generated_at` (UTC ISO timestamp) · `model_name` · `model_version` · `dataset` ·
`threshold` · `sensor_inputs` · `engineered_features` (with rounded values) ·
`prediction` (uncalibrated score, boolean, status, calibration note) ·
`explanation_drivers` · `explanation_method` · `failure_modes` + method note ·
`guidance` · `human_review_required` · `agent_tools_executed` (step, tool, status per
workflow state) · `disclaimer` · `limitations`.

All values come from the actual executed analysis; nothing is fabricated. JSON validity
and content were verified in automated checks (section 17).

---

## 15. Evaluation results and probability calibration

**Holdout performance (20% stratified test split, threshold 0.30):**

| Metric | Value |
|---|---|
| Recall | 0.824 |
| Precision | 0.737 |
| F1 | 0.778 |
| ROC-AUC | 0.983 |
| PR-AUC | 0.861 |
| Brier score | 0.011 |
| Confusion matrix | TN 1912 · FP 20 · FN 12 · TP 56 |

**Calibration analysis** (`scripts/check_calibration.py`) binned predicted
probabilities against observed failure rates in the holdout set. The largest
reliability gap was +0.029 in the top quantile — acceptable for education, but the
project deliberately labels all probability outputs as **uncalibrated model risk
scores** rather than real-world probabilities.

**On very high scores (e.g. 99%):** extreme inputs produce extreme confidences. The
power demonstration scenario reaches 18,378 W mechanical power, while the dataset's
99th percentile is 8,821 W; every training record above 9,000 W (n = 64) is a failure,
so high model confidence is expected. The correct interpretation is stated in the UI:
the score is the model's uncalibrated risk for those input conditions, not a verified
real-world 99% probability of failure.

---

## 16. Demonstration scenarios

| Scenario | Inputs (K, K, rpm, Nm, min) | Expected result |
|---|---|---|
| Baseline Operating Run (demo) | 298.0, 308.0, 1500, 40, 50 | NORMAL, ~0% |
| Simulate Dull Tool Wear (demo) | 298.0, 309.0, 1400, 55, 250 | HIGH RISK; top indicator TWF |
| Simulate Thermal Conditions (demo) | 302.5, 311.0, 1340, 53, 110 | HIGH RISK; top indicator HDF |
| Power-Related Scenario (demo) | 297.0, 307.0, 2700, 65, 80 | HIGH RISK 99.0%; top indicator PWF; review gate set |

The thermal scenario was tuned during development: the original 20 K temperature delta
produced no risk because the AI4I HDF band is 7.6–8.6 K; the scenario now uses the
documented band. Scenario values are illustrative inputs, not validated operating
conditions of any real machine.

---

## 17. Testing and validation performed

| Test area | Method | Result |
|---|---|---|
| Physics feature formula | Assertion vs hand-computed value | Passed |
| Input validation | Missing, text, and extreme values | Correctly rejected with per-field messages |
| Prediction path | Baseline and high-risk inputs through the full agent | Correct statuses and scores |
| All scenarios | End-to-end `analyze()` on all four scenarios | High-risk scenarios flagged; baseline NORMAL |
| Trace fidelity | Trace contains exactly 9 states with real timings | Verified |
| Work-order JSON | Parsed and asserted field-by-field | Valid; includes review flag and tool trace |
| Model preservation | Reloaded artifact re-predicted expected scores | Unchanged |
| Invalid-input handling | Non-numeric value through UI path | INVALID status, failed trace step shown |
| SHAP | Method string and latency reported per run | `shap_tree_explainer`, 20–35 ms local |
| Deployment | Live API inference on the public URL | HIGH RISK 99.0%, trace and indicators present |
| UI launch | Local and Render HTTP checks | HTTP 200 both |

Verification scripts live in `scripts/` (`train_pipeline.py`, `check_calibration.py`);
ad-hoc verification scripts used during development were executed and removed.

---

## 18. Deployment

| Item | Detail |
|---|---|
| Host | Render (free plan, Oregon region) |
| URL | https://predictive-maintenance-agent-s9d6.onrender.com |
| Service type | Web service, Python runtime |
| Build command | `pip install -r requirements.txt` |
| Start command | `python app.py` |
| Port binding | `0.0.0.0`, port from `$PORT` env var (default 7860) |
| Auto-deploy | On every push to `main` of the GitHub repository |
| Health check | `/` |
| Python version | 3.11.9 (pinned via env var) |

The dataset CSV is intentionally not committed; the server fetches UCI ID 601 on first
start via `ucimlrepo` and caches it. The trained model artifact and metadata JSON are
committed so production inference needs no training step. Free-tier instances spin
down after inactivity; the first request after idle may take ~30 s to wake.

---

## 19. Project structure and file inventory

```
predictive-maintenance-intelligence/
├── app.py                        # Gradio dashboard wired to the agent
├── agent/
│   ├── __init__.py               # package exports
│   ├── orchestrator.py           # workflow engine, 9 states, trace, review gate
│   └── tools.py                  # 8 tool functions + ToolError
├── src/
│   ├── config.py                 # features, ranges, paths, risk bands, seed
│   ├── data_preprocessing.py     # input validation + coercion
│   ├── feature_engineering.py    # PhysicsFeatures transformer
│   ├── train.py                  # candidate models, CV, pipeline builder, thresholds
│   ├── evaluate.py               # holdout metrics, classification report
│   ├── predict.py                # inference, risk bands, failure-mode heuristics
│   ├── explainability.py         # SHAP with reported fallback
│   ├── recommendations.py        # cautious guidance rules + disclaimer
│   └── work_order.py             # JSON work order builder
├── scripts/
│   ├── train_pipeline.py         # end-to-end training + metadata (--auto, --model)
│   └── check_calibration.py      # calibration + scenario probability analysis
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_threshold_analysis.ipynb
├── models/
│   ├── predictive_maintenance_pipeline.pkl   # fitted pipeline (committed)
│   └── pipeline_metadata.json                # metrics, threshold, assumptions
├── data/                         # ai4i2020.csv (fetched, gitignored)
├── requirements.txt
├── render.yaml                   # Render service definition
├── README.md                     # concise project documentation
├── REPORT.md                     # this document
└── .gitignore
```

Total tracked source files: 26.

---

## 20. Installation and local execution

```bash
# 1. Install dependencies (ucimlrepo only needed to auto-fetch the dataset)
pip install -r requirements.txt

# 2. Train (or retrain) the model — fetches UCI id=601 if no local CSV exists
python scripts/train_pipeline.py --model xgboost

# 3. Optional: calibration and scenario analysis report
python scripts/check_calibration.py

# 4. Run the application
python app.py
# open http://127.0.0.1:7860
```

Retraining alternatives: `--auto` (selects the best-recall model from CV),
`--data <path>` (custom CSV path), `--model {logistic_regression, random_forest, xgboost}`.

---

## 21. Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML | scikit-learn 1.8 (pipelines, CV, metrics), XGBoost 3.4 |
| Data | pandas, NumPy |
| Explainability | SHAP 0.52 (TreeExplainer) |
| UI | Gradio 6.28 (custom CSS, HTML components), Plotly 6.8 (gauge, bar charts) |
| Persistence | joblib (pipeline), JSON (metadata, work orders) |
| Dataset access | ucimlrepo |
| Version control | Git / GitHub |
| Hosting | Render (free web service) |

---

## 22. Limitations

1. **Synthetic data.** AI4I 2020 is generated from rules; this is an educational
   prototype and not a certified industrial safety or maintenance system.
2. **Uncalibrated scores.** Probabilities are model scores; no calibrated real-world
   failure probability is claimed. Platt/isotonic calibration is future work.
3. **Heuristic failure modes.** Indicators are not a trained multi-label classifier.
4. **No real-time integration.** Inputs are manual or scenario-loaded; no IoT
   ingestion, no equipment control, no shutdown commands.
5. **Illustrative costs.** The threshold cost model (FN 10 : FP 1) is documented but
   not derived from real maintenance economics.
6. **Free-tier deployment.** Render free instances sleep after inactivity (~30 s cold
   start).
7. **Not an autonomous LLM agent.** The workflow is deterministic orchestration; it is
   labeled as such in all documentation.

---

## 23. Future improvements

- Probability calibration (Platt scaling / isotonic regression) with reliability
  curves shown in the UI.
- A trained multi-label classifier for failure modes to replace the heuristic layer.
- Real-time sensor ingestion (MQTT/REST) and Remaining Useful Life estimation with a
  time-series dataset.
- Input-drift monitoring and prediction logging to a database.
- Authentication and role-based access if deployed beyond classroom use.

---

## 24. Viva preparation

**Q: Why is accuracy not a suitable metric here?**
Failures are 3.39% of the data; always predicting "no failure" scores 96.6% accuracy
while catching zero failures. Recall, precision, F1, ROC-AUC and PR-AUC expose the
error types that matter.

**Q: Why tune the threshold instead of using 0.5?**
The cost of a missed failure and a false alarm are asymmetric. With the documented
10:1 assumption, threshold 0.30 minimises total cost while holding false negatives at
12 on the holdout set. The assumption is stated wherever the threshold is shown.

**Q: Why compute physics features inside the pipeline?**
It guarantees the identical transformation at training and inference and prevents
inconsistent preprocessing or leakage; the transformer is stateless so no statistic
crosses the train/test boundary.

**Q: What makes this project agentic?**
A maintenance agent coordinates eight real tools across nine explicit workflow states,
branches its guidance and review gate on the prediction result, and emits a
per-step execution trace with real timings. It is a tool-orchestrated workflow, not an
autonomous LLM agent — stated honestly in all documentation.

**Q: Why do some scenarios show 99%?**
The model is confident on extreme inputs (e.g. 18,378 W vs a dataset P99 of 8,821 W,
and every training row above 9,000 W failed). The score is uncalibrated model
confidence for those conditions, not a verified real-world probability, and the UI
says so.

**Q: Is the data real?**
No. AI4I 2020 is synthetic and this is an educational prototype; all UI and documents
carry that disclaimer.

---

## 25. References

1. UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset
   (ID 601), https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
2. Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance
   Applications. *IEEE Access*.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
4. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model
   Predictions. *NeurIPS*.
5. scikit-learn documentation — Pipelines and model selection.
   https://scikit-learn.org
6. Gradio documentation. https://www.gradio.app
7. SHAP documentation. https://shap.readthedocs.io

---

*Prepared for academic submission. All metrics in this report are measured values
from the deployed model; claims of autonomous reasoning, real-world probability
calibration, or certified diagnosis are explicitly not made.*
