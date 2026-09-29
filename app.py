import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import joblib
import pandas as pd
import plotly.graph_objects as go

from agent.orchestrator import DISCLAIMER, run_maintenance_agent
from src.config import (
    DEFAULT_THRESHOLD,
    MODEL_VERSION,
    RAW_FEATURES,
)
from src.predict import risk_band

PIPELINE_PATH = os.path.join("models", "predictive_maintenance_pipeline.pkl")
METADATA_PATH = os.path.join("models", "pipeline_metadata.json")

SCENARIOS = {
    "Baseline Operating Run (demo)": {
        "Air temperature": 298.0,
        "Process temperature": 308.0,
        "Rotational speed": 1500.0,
        "Torque": 40.0,
        "Tool wear": 50.0,
    },
    "Simulate Dull Tool Wear (demo)": {
        "Air temperature": 298.0,
        "Process temperature": 309.0,
        "Rotational speed": 1400.0,
        "Torque": 55.0,
        "Tool wear": 250.0,
    },
    "Simulate Thermal Conditions (demo)": {
        "Air temperature": 302.5,
        "Process temperature": 311.0,
        "Rotational speed": 1340.0,
        "Torque": 53.0,
        "Tool wear": 110.0,
    },
    "Power-Related Scenario (demo)": {
        "Air temperature": 297.0,
        "Process temperature": 307.0,
        "Rotational speed": 2700.0,
        "Torque": 65.0,
        "Tool wear": 80.0,
    },
}


def load_artifacts():
    pipeline = joblib.load(PIPELINE_PATH) if os.path.exists(PIPELINE_PATH) else None
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
    background = None
    data_path = os.path.join("data", "ai4i2020.csv")
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path, usecols=RAW_FEATURES)
            background = df.sample(min(500, len(df)), random_state=42)
        except Exception:
            background = None
    return pipeline, metadata, background


PIPELINE, METADATA, BACKGROUND = load_artifacts()
THRESHOLD = float(METADATA.get("threshold", DEFAULT_THRESHOLD))
MODEL_NAME = METADATA.get("model", "xgboost")
ACTIVE_MODEL_VERSION = METADATA.get("model_version", MODEL_VERSION + " (untrained demo)")
DATASET_NAME = "UCI AI4I 2020"
test_metrics = METADATA.get("test_metrics", {})

STATUS_COLORS = {
    "HIGH RISK": ("#f04438", "rgba(240,68,56,0.14)"),
    "WATCHLIST": ("#f5a524", "rgba(245,165,36,0.14)"),
    "NORMAL": ("#2ebd7a", "rgba(46,189,122,0.14)"),
    "INVALID": ("#98a2b3", "rgba(152,162,179,0.14)"),
    "NO MODEL": ("#98a2b3", "rgba(152,162,179,0.14)"),
}

CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.hero { background: linear-gradient(135deg, #14181f 0%, #1b2230 60%, #232b3d 100%);
  border: 1px solid #2b3446; border-radius: 16px; padding: 28px 32px; margin-bottom: 16px; }
.hero h1 { margin: 0 0 4px 0; font-size: 30px; letter-spacing: -0.02em; color: #f2f4f7; }
.hero h1 .accent { color: #f5a524; }
.hero p { margin: 6px 0 0 0; color: #b6bfcf; font-size: 14.5px; line-height: 1.55; max-width: 76ch; }
.stat-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.stat-pill { font-size: 12.5px; font-weight: 600; color: #e4e9f2; background: #222b3b;
  border: 1px solid #334052; border-radius: 999px; padding: 6px 12px; white-space: nowrap; }
.stat-pill b { color: #f5a524; font-weight: 700; }
.cal-note { margin-top: 12px; font-size: 12.5px; color: #98a2b3; line-height: 1.5; }
.panel { background: #161b24 !important; border: 1px solid #2b3446 !important; border-radius: 14px !important; padding: 20px !important; }
.panel h3, .panel-title { margin: 0 0 4px 0 !important; font-size: 15px !important; color: #f2f4f7 !important;
  text-transform: uppercase; letter-spacing: 0.08em; }
.panel-sub { font-size: 12.5px; color: #98a2b3; margin: 0 0 14px 0; line-height: 1.5; }
.control-bar { align-items: end; }
#analyze-btn { height: 100%; min-height: 58px; font-size: 16px !important; font-weight: 700 !important; }
.group-label { font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
  color: #f5a524; margin: 14px 0 2px 0; }
.group-label:first-of-type { margin-top: 2px; }
.status-pill { display: flex; align-items: center; gap: 10px; border-radius: 12px; padding: 14px 16px;
  font-size: 20px; font-weight: 800; letter-spacing: 0.02em; border: 1px solid; margin-bottom: 12px; }
.status-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.prob-card { border-radius: 12px; padding: 10px 16px 14px 16px; background: #10141b;
  border: 1px solid #2b3446; margin-bottom: 4px; }
.prob-card .prob-label { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.1em; color: #98a2b3; }
.prob-card .prob-value { font-size: 34px; font-weight: 800; color: #f2f4f7; line-height: 1.1; font-variant-numeric: tabular-nums; }
.prob-card .prob-sub { font-size: 12px; color: #98a2b3; }
.trace-list { display: flex; flex-direction: column; gap: 0; margin-top: 6px; }
.trace-step { display: grid; grid-template-columns: 28px 1fr; gap: 12px; position: relative; padding: 9px 0; }
.trace-step:not(:last-child)::before { content: ""; position: absolute; left: 13px; top: 34px; bottom: -4px;
  width: 2px; background: #2b3446; }
.trace-dot { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 14px; font-weight: 800; border: 2px solid; background: #10141b; z-index: 1; }
.trace-ok .trace-dot { color: #2ebd7a; border-color: #2ebd7a; }
.trace-failed .trace-dot { color: #f04438; border-color: #f04438; }
.trace-skipped .trace-dot { color: #98a2b3; border-color: #475467; }
.trace-body { font-size: 13.5px; color: #d0d5dd; line-height: 1.5; }
.trace-body code { background: #222b3b; border: 1px solid #334052; border-radius: 6px; padding: 1px 6px;
  font-size: 12px; color: #f5a524; }
.trace-meta { font-size: 12px; color: #98a2b3; }
.trace-detail { font-size: 12.5px; color: #98a2b3; }
.trace-empty { color: #98a2b3; font-size: 13.5px; padding: 12px 0; }
.modes-table { width: 100%; border-collapse: collapse; margin-top: 6px; }
.modes-table th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
  color: #98a2b3; padding: 8px 8px; border-bottom: 1px solid #2b3446; }
.modes-table td { padding: 10px 8px; border-bottom: 1px solid #1f2633; font-size: 13.5px; color: #e4e9f2; vertical-align: middle; }
.modes-table tr:last-child td { border-bottom: none; }
.mode-code { font-weight: 800; background: #222b3b; border: 1px solid #334052; border-radius: 6px;
  padding: 2px 8px; font-size: 12.5px; white-space: nowrap; }
.conf-bar { height: 8px; background: #222b3b; border-radius: 999px; overflow: hidden; min-width: 90px; }
.conf-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #f5a524, #f04438); }
.conf-val { font-variant-numeric: tabular-nums; font-weight: 700; white-space: nowrap; }
.review-alert { border-radius: 12px; padding: 14px 16px; font-size: 13.5px; line-height: 1.55; margin-bottom: 12px; border: 1px solid; }
.review-alert.high { background: rgba(240,68,56,0.1); border-color: #f04438; color: #ffd9d4; }
.review-alert.low { background: rgba(46,189,122,0.1); border-color: #2ebd7a; color: #c8f0db; }
.review-alert strong { display: block; font-size: 14.5px; margin-bottom: 2px; }
.guidance-list { margin: 4px 0 0 0; padding-left: 20px; color: #d0d5dd; font-size: 13.5px; line-height: 1.65; }
.guidance-list li { margin-bottom: 6px; }
.method-note { font-size: 12px; color: #98a2b3; line-height: 1.55; }
.footer-note { background: #14181f; border: 1px solid #2b3446; border-radius: 12px; padding: 16px 20px;
  font-size: 12.5px; color: #98a2b3; line-height: 1.6; }
@media (max-width: 900px) { .hero { padding: 20px; } .hero h1 { font-size: 23px; } }
"""


def make_gauge(probability, status):
    _, color = risk_band(probability)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100.0,
            number={"suffix": "%", "font": {"size": 44, "color": "#f2f4f7"}},
            title={"text": status, "font": {"size": 20, "color": "#f2f4f7"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#98a2b3", "tickfont": {"color": "#98a2b3"}},
                "bar": {"color": color},
                "bgcolor": "#10141b",
                "bordercolor": "#2b3446",
                "steps": [
                    {"range": [0, 30], "color": "rgba(46,189,122,0.22)"},
                    {"range": [30, 65], "color": "rgba(245,165,36,0.25)"},
                    {"range": [65, 100], "color": "rgba(240,68,56,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#f2f4f7", "width": 4},
                    "thickness": 0.8,
                    "value": THRESHOLD * 100.0,
                },
            },
        )
    )
    fig.update_layout(
        height=300, margin=dict(l=24, r=24, t=56, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e4e9f2"},
    )
    return fig


def make_shap_fig(drivers):
    top = drivers[:8][::-1]
    fig = go.Figure(
        go.Bar(
            x=[d["contribution"] for d in top],
            y=[d["feature"] for d in top],
            orientation="h",
            marker_color=["#f04438" if d["contribution"] >= 0 else "#2ebd7a" for d in top],
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "Strongest drivers of this prediction", "font": {"size": 14, "color": "#f2f4f7"}},
        xaxis_title="Contribution",
        height=360,
        margin=dict(l=16, r=16, t=48, b=44),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d0d5dd", "size": 12},
        xaxis={"gridcolor": "#232b3b", "zerolinecolor": "#475467"},
        yaxis={"gridcolor": "#232b3b"},
    )
    return fig


def empty_gauge():
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=0,
            number={"suffix": "%", "font": {"color": "#98a2b3"}},
            title={"text": "Awaiting analysis", "font": {"color": "#98a2b3", "size": 18}},
            gauge={"axis": {"range": [0, 100], "tickcolor": "#475467"},
                   "bar": {"color": "#475467"}, "bgcolor": "#10141b", "bordercolor": "#2b3446"},
        )
    )
    fig.update_layout(height=300, margin=dict(l=24, r=24, t=56, b=16),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def status_html(status: str) -> str:
    color, bg = STATUS_COLORS.get(status, ("#98a2b3", "rgba(152,162,179,0.14)"))
    return (
        f"<div class='status-pill' style='color:{color};background:{bg};border-color:{color};'>"
        f"<span class='status-dot' style='background:{color};'></span>{status}</div>"
    )


def prob_html(probability: float, threshold: float) -> str:
    return (
        "<div class='prob-card'><div class='prob-label'>Failure probability "
        "(uncalibrated model score)</div>"
        f"<div class='prob-value'>{probability:.1%}</div>"
        f"<div class='prob-sub'>Operating threshold {threshold:.2f} "
        "(black tick on gauge)</div></div>"
    )


def trace_html(trace: list) -> str:
    if not trace:
        return "<div class='trace-empty'>Run an analysis to see each agent tool execute here.</div>"
    parts = ["<div class='trace-list'>"]
    for entry in trace:
        status = entry["status"]
        icon = "✓" if status == "ok" else ("○" if status == "skipped" else "✕")
        timing = f" · {entry['elapsed_ms']} ms" if entry.get("elapsed_ms") is not None else ""
        parts.append(
            f"<div class='trace-step trace-{status}'>"
            f"<div class='trace-dot'>{icon}</div>"
            f"<div class='trace-body'><strong>{entry['step']}</strong> "
            f"<code>{entry['tool']}</code> "
            f"<span class='trace-meta'>— {status}{timing}</span><br>"
            f"<span class='trace-detail'>{entry['detail']}</span></div></div>"
        )
    parts.append("</div>")
    return "".join(parts)


def modes_html(modes: list) -> str:
    if not modes:
        return "<div class='trace-empty'>No indicators yet. Run an analysis first.</div>"
    rows = []
    for m in modes:
        conf = float(m["confidence"])
        rows.append(
            "<tr><td><span class='mode-code'>" + m["mode"] + "</span></td>"
            f"<td>{m['meaning']}</td>"
            f"<td><div class='conf-bar'><div class='conf-fill' style='width:{conf:.0%};'></div></div></td>"
            f"<td class='conf-val'>{conf:.1%}</td></tr>"
        )
    return (
        "<table class='modes-table'><thead><tr><th>Mode</th><th>Meaning</th>"
        "<th style='min-width:110px;'>Indicator</th><th>Value</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def guidance_html(review_required: bool, guidance: list) -> str:
    if review_required:
        banner = (
            "<div class='review-alert high'><strong>Human review required</strong>"
            "High-risk score produced. Qualified maintenance personnel must confirm "
            "operating conditions against approved documentation before any action. "
            "This system does not issue shutdown commands or control equipment.</div>"
        )
    else:
        banner = (
            "<div class='review-alert low'><strong>No review gate triggered</strong>"
            "Low risk. Continue routine monitoring.</div>"
        )
    items = "".join(f"<li>{g}</li>" for g in guidance)
    return banner + f"<ol class='guidance-list'>{items}</ol>"


def empty_modes_html() -> str:
    return "<div class='trace-empty'>No indicators yet. Run an analysis first.</div>"


def analyze(air, process, rpm, torque, wear):
    if PIPELINE is None:
        notice = (
            "No trained pipeline found at models/predictive_maintenance_pipeline.pkl. "
            "Run `python scripts/train_pipeline.py` first, then restart the app."
        )
        return (
            status_html("NO MODEL"), prob_html(0.0, THRESHOLD), empty_gauge(),
            trace_html([]), empty_modes_html(), empty_gauge(),
            f"<div class='trace-empty'>{notice}</div>",
            f"<div class='review-alert high'><strong>Model unavailable</strong>{notice}</div>",
            None,
        )
    record = {
        "Air temperature": air,
        "Process temperature": process,
        "Rotational speed": rpm,
        "Torque": torque,
        "Tool wear": wear,
    }
    result = run_maintenance_agent(
        PIPELINE, record, THRESHOLD, background=BACKGROUND,
        model_name=MODEL_NAME, model_version=ACTIVE_MODEL_VERSION,
        dataset=DATASET_NAME,
    )
    if not result.get("ok"):
        failed = [t for t in result["trace"] if t["status"] == "failed"]
        where = failed[-1]["step"] if failed else "agent"
        err = f"Agent stopped at {where}: {result.get('error', 'unknown error')}"
        return (
            status_html("INVALID"), prob_html(0.0, THRESHOLD), empty_gauge(),
            trace_html(result["trace"]), empty_modes_html(), empty_gauge(),
            "<div class='method-note'>Explanation unavailable: input validation failed.</div>",
            f"<div class='review-alert high'><strong>Invalid input</strong>{err}</div>",
            None,
        )

    prediction = result["prediction"]
    drivers = result["drivers"]
    modes = result["failure_modes"]
    guidance = result["guidance"]
    gauge = make_gauge(prediction["failure_probability"], prediction["status"])
    shap_fig = make_shap_fig(drivers)
    explainer_note = (
        f"<div class='method-note'>Method: <b>{result['explanation_method']}</b> "
        f"in {result['explanation_latency_ms']} ms. These features had the strongest "
        "influence on this particular model prediction. Feature importance does not "
        "establish causation or guarantee a physical failure.</div>"
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
    tmp.write(result["work_order_json"])
    tmp.close()
    return (
        status_html(prediction["status"]),
        prob_html(prediction["failure_probability"], THRESHOLD),
        gauge,
        trace_html(result["trace"]),
        modes_html(modes),
        shap_fig,
        explainer_note,
        guidance_html(result["human_review_required"], guidance),
        tmp.name,
    )


def fill_scenario(name):
    values = SCENARIOS.get(name, SCENARIOS["Baseline Operating Run (demo)"])
    return (
        values["Air temperature"],
        values["Process temperature"],
        values["Rotational speed"],
        values["Torque"],
        values["Tool wear"],
    )


HEADER_HTML = """
<div class="hero">
  <h1>Predictive Maintenance <span class="accent">Intelligence Agent</span></h1>
  <p>A tool-orchestrated maintenance agent: sensor inputs flow through validation,
  physics feature engineering, XGBoost risk prediction, risk decision, failure-mode
  indicators, SHAP explanation, cautious guidance, and work-order generation,
  with a human-review gate on high risk. Educational prototype on synthetic data.</p>
  <div class="stat-row">
    <span class="stat-pill">Model <b>{model}</b></span>
    <span class="stat-pill">Version <b>{version}</b></span>
    <span class="stat-pill">Threshold <b>{threshold:.2f}</b></span>
    <span class="stat-pill">Recall <b>{recall:.2f}</b></span>
    <span class="stat-pill">Precision <b>{precision:.2f}</b></span>
    <span class="stat-pill">F1 <b>{f1:.2f}</b></span>
    <span class="stat-pill">ROC-AUC <b>{roc:.2f}</b></span>
    <span class="stat-pill">Dataset <b>{dataset}</b></span>
  </div>
  <p class="cal-note">Probability is the model's uncalibrated risk score for the input
  conditions, not a calibrated real-world chance of failure. Threshold 0.30 chosen on
  validation (recall 0.82, precision 0.74, ROC-AUC 0.98, PR-AUC 0.86).</p>
</div>
""".format(
    model=MODEL_NAME,
    version=ACTIVE_MODEL_VERSION,
    threshold=THRESHOLD,
    recall=test_metrics.get("recall", float("nan")),
    precision=test_metrics.get("precision", float("nan")),
    f1=test_metrics.get("f1", float("nan")),
    roc=test_metrics.get("roc_auc", float("nan")),
    dataset=DATASET_NAME,
)

with gr.Blocks(title="Predictive Maintenance Intelligence Agent") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(elem_classes="control-bar"):
        scenario = gr.Dropdown(
            choices=list(SCENARIOS.keys()),
            value="Baseline Operating Run (demo)",
            label="Demonstration scenario (illustrative, not validated real-world conditions)",
            scale=5,
        )
        load_btn = gr.Button("Load scenario", scale=2)
        analyze_btn = gr.Button("Analyze health", variant="primary", scale=2, elem_id="analyze-btn")

    with gr.Row():
        with gr.Column(scale=5, elem_classes="panel"):
            gr.Markdown("### Sensor inputs")
            gr.Markdown("Manual or scenario-loaded telemetry. All values validated before inference.",
                        elem_classes="panel-sub")
            gr.Markdown("<div class='group-label'>Thermal</div>")
            air = gr.Number(value=298.0, label="Air temperature [K]")
            process = gr.Number(value=308.0, label="Process temperature [K]")
            gr.Markdown("<div class='group-label'>Mechanical load</div>")
            rpm = gr.Number(value=1500.0, label="Rotational speed [rpm]")
            torque = gr.Number(value=40.0, label="Torque [Nm]")
            gr.Markdown("<div class='group-label'>Wear</div>")
            wear = gr.Number(value=50.0, label="Tool wear [min]")
        with gr.Column(scale=7, elem_classes="panel"):
            gr.Markdown("### Risk result")
            gr.Markdown("Status, uncalibrated score, and gauge update on every analysis.",
                        elem_classes="panel-sub")
            status = gr.HTML(value=status_html("NORMAL"))
            prob = gr.HTML(value=prob_html(0.0, THRESHOLD))
            gauge = gr.Plot(value=empty_gauge, label="Risk gauge")

    with gr.Row():
        with gr.Column(elem_classes="panel"):
            gr.Markdown("### Agent execution trace")
            gr.Markdown("Each tool executes real Python code. Status, result, and timing shown per step.",
                        elem_classes="panel-sub")
            trace_out = gr.HTML(value=trace_html([]))

    with gr.Row():
        with gr.Column(elem_classes="panel"):
            gr.Markdown("### Failure-mode risk indicators")
            gr.Markdown("Model-derived indicators. Multiple modes may flag at once. Not certified diagnoses. Heuristic method aligned to AI4I generation bands.",
                        elem_classes="panel-sub")
            modes_out = gr.HTML(value=empty_modes_html())
        with gr.Column(elem_classes="panel"):
            gr.Markdown("### SHAP explanation")
            gr.Markdown("Strongest drivers of this specific prediction.",
                        elem_classes="panel-sub")
            shap_plot = gr.Plot(label="Feature drivers")
            explainer_note = gr.HTML(
                value="<div class='method-note'>Run an analysis to generate the explanation.</div>")

    with gr.Row():
        with gr.Column(scale=7, elem_classes="panel"):
            gr.Markdown("### Maintenance guidance")
            gr.Markdown("Cautious next steps. High risk triggers a human-review gate.",
                        elem_classes="panel-sub")
            guidance_out = gr.HTML(
                value="<div class='trace-empty'>Guidance appears here after analysis.</div>")
        with gr.Column(scale=5, elem_classes="panel"):
            gr.Markdown("### Work order")
            gr.Markdown("Traceable JSON: telemetry, engineered features, risk, modes, drivers, guidance, review flag, tool trace.",
                        elem_classes="panel-sub")
            work_order_file = gr.File(label="Download JSON work order")

    gr.HTML(
        "<div class='footer-note'><b>Disclaimer and limitations.</b> "
        + DISCLAIMER + " Recommendations require qualified human review. "
        "No automatic shutdown commands; no equipment control. "
        "Scenarios are illustrative demonstrations.</div>"
    )

    load_btn.click(fill_scenario, inputs=scenario, outputs=[air, process, rpm, torque, wear])
    scenario.change(fill_scenario, inputs=scenario, outputs=[air, process, rpm, torque, wear])
    analyze_btn.click(
        analyze,
        inputs=[air, process, rpm, torque, wear],
        outputs=[status, prob, gauge, trace_out, modes_out, shap_plot, explainer_note, guidance_out, work_order_file],
    )

    if PIPELINE is None:
        gr.Markdown(
            "> No trained pipeline found. Run `python scripts/train_pipeline.py` "
            "to train on the UCI AI4I 2020 dataset, then restart the app."
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True, css=CUSTOM_CSS)
