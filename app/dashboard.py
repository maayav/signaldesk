"""Interactive local client for the SignalDesk inference API.

Run from the repository root:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import html
import io
import json
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from theme import COLORS, apply_theme, chart_layout


API_URL = os.environ.get("SIGNALDESK_API_URL", "http://127.0.0.1:8000").rstrip("/")
EXAMPLES = {
    "Card swallowed": "The ATM kept my card and won't give it back.",
    "Missing transfer": "I sent a bank transfer yesterday but the recipient still has not received it.",
    "Duplicate charge": "I have been charged twice for the same card transaction.",
    "Outside the domain": "Can you recommend a good movie to watch tonight?",
}


class APIError(RuntimeError):
    """Displayable inference-service error."""


def api_request(method: str, endpoint: str, payload: dict | None = None) -> Any:
    try:
        response = requests.request(
            method, f"{API_URL}{endpoint}", json=payload, timeout=(3, 90)
        )
    except requests.RequestException as exc:
        raise APIError(f"The inference service is unavailable at {API_URL}.") from exc
    if not response.ok:
        try:
            detail = response.json().get("detail", response.reason)
        except (ValueError, AttributeError):
            detail = response.reason
        if isinstance(detail, list):
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
        raise APIError(f"Request failed ({response.status_code}): {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise APIError("The inference service returned an invalid response.") from exc


@st.cache_data(ttl=30, show_spinner=False)
def get_model(api_url: str, model_version: str | None = None) -> dict:
    # Both arguments form the cache key: an API restart with a new model must
    # refresh its routing threshold immediately, even at the same address.
    return api_request("GET", "/model")


@st.cache_data(ttl=30, show_spinner=False)
def get_metrics(api_url: str, model_version: str | None = None) -> dict:
    return api_request("GET", "/metrics")


def humanize(intent: str) -> str:
    return str(intent).replace("_", " ").replace("?", "").capitalize()


def pct(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{digits}%}"


def metric_row(items: list[tuple[str, Any, str]]) -> None:
    for column, (label, value, help_text) in zip(st.columns(len(items)), items):
        column.metric(label, value, help=help_text)


def header(overline: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="sd-overline">{html.escape(overline)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="sd-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def show_api_error(exc: APIError) -> None:
    st.error(str(exc))


def set_example(text: str) -> None:
    st.session_state["ticket_text"] = text
    st.session_state.pop("single_prediction", None)


def prediction_view(prediction: dict, model: dict) -> None:
    is_route = prediction.get("decision") == "route"
    intent = humanize(prediction.get("intent", "Unknown"))
    decision = "Ready to route" if is_route else "Human review requested"
    detail = (
        "Confidence meets the validation-selected routing threshold."
        if is_route
        else "Confidence is below the routing threshold. Ask a reviewer to verify the intent."
    )
    st.markdown(
        f'<div class="sd-result {"" if is_route else "review"}">'
        f'<div class="label">{decision}</div><div class="intent">{html.escape(intent)}</div>'
        f'<div class="detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )
    metric_row([
        ("Model confidence", pct(prediction.get("confidence")), "Calibrated class probability; it is not a guarantee of correctness."),
        ("Routing threshold", pct(model.get("threshold")), "Selected on policy-validation data, never on the test set."),
        ("Inference latency", f'{prediction.get("latency_ms", 0):.1f} ms', "Service-reported prediction latency. This varies by hardware and request size."),
    ])
    st.write("")
    tabs = st.tabs(["Alternative intents", "Similar examples", "Feature evidence", "Review feedback"])
    with tabs[0]:
        alternatives = prediction.get("top_intents", [])
        if alternatives:
            frame = pd.DataFrame(alternatives)
            frame["Intent"] = frame["intent"].map(humanize)
            figure = go.Figure(go.Bar(
                x=frame["confidence"], y=frame["Intent"], orientation="h",
                marker_color=COLORS["mint"], text=frame["confidence"].map(pct),
                textposition="auto", hovertemplate="%{y}<br>%{x:.1%}<extra></extra>",
            ))
            figure.update_xaxes(range=[0, 1], tickformat=".0%", title="Calibrated probability")
            figure.update_yaxes(autorange="reversed")
            st.plotly_chart(chart_layout(figure, height=250), use_container_width=True)
    with tabs[1]:
        st.caption("Nearest training examples by TF–IDF cosine similarity. These are reference cases, not generated answers.")
        for example in prediction.get("similar_tickets", []):
            with st.container(border=True):
                st.markdown(f'**{humanize(example.get("intent", ""))}** · similarity {float(example.get("similarity", 0)):.2f}')
                st.write(example.get("text", ""))
        if not prediction.get("similar_tickets"):
            st.info("No similar examples were returned for this message.")
    with tabs[2]:
        st.caption("Positive feature contributions to the underlying linear classifier's predicted-class score. These explain the raw score, not the calibrated probability or a causal relationship.")
        evidence = prediction.get("explanations", [])
        if evidence:
            frame = pd.DataFrame(evidence)
            st.dataframe(frame.rename(columns={"feature": "Text feature", "contribution": "Score contribution"}), hide_index=True, width="stretch")
        else:
            st.info("No positive feature contributions are available for this prediction.")
    with tabs[3]:
        st.caption("Record a verified intent for this prediction. Feedback is stored for review; it does not automatically retrain the model.")
        classes = model.get("classes", [])
        if classes:
            current = prediction.get("intent")
            index = classes.index(current) if current in classes else 0
            correct_intent = st.selectbox("Verified intent", classes, index=index, format_func=humanize, key=f'feedback_{prediction.get("prediction_id")}')
            if st.button("Save verified label", key=f'feedback_save_{prediction.get("prediction_id")}'):
                try:
                    api_request("POST", "/feedback", {"prediction_id": prediction["prediction_id"], "correct_intent": correct_intent})
                    st.success("Verified label recorded.")
                except APIError as exc:
                    show_api_error(exc)
    st.caption(f'Prediction {prediction.get("prediction_id", "—")} · model {prediction.get("model_version", "—")}')


def workspace(model: dict) -> None:
    header("01 / Inference workspace", "Understand the ticket. Decide the next step.", "Classify a support message, inspect the evidence, and keep uncertain cases in the review queue.")
    with st.container(border=True):
        st.markdown("#### Incoming message")
        st.text_area("Support message", key="ticket_text", height=140, max_chars=4000, placeholder="Describe the customer's issue in English…", label_visibility="collapsed")
        col_submit, col_note = st.columns([1, 2.5], vertical_alignment="center")
        analyze = col_submit.button("Analyze message →", type="primary", width="stretch")
        col_note.caption("3–4,000 characters · runs locally · no paid API")
    st.caption("Try a reference scenario")
    for col, (label, text) in zip(st.columns(4), EXAMPLES.items()):
        col.button(label, on_click=set_example, args=(text,), width="stretch")
    st.markdown('<div class="sd-step"><span>01</span> Text features &nbsp; → &nbsp; <span>02</span> Calibrated classifier &nbsp; → &nbsp; <span>03</span> Route or review &nbsp; → &nbsp; <span>04</span> Monitor</div>', unsafe_allow_html=True)
    if analyze:
        text = st.session_state.get("ticket_text", "").strip()
        if len(text) < 3:
            st.warning("Enter a support message with at least three characters.")
        else:
            try:
                with st.spinner("Scoring the ticket and retrieving similar examples…"):
                    st.session_state["single_prediction"] = api_request("POST", "/predict", {"text": text})
                    st.session_state["predicted_text"] = text
            except APIError as exc:
                show_api_error(exc)
    prediction = st.session_state.get("single_prediction")
    if (
        prediction
        and prediction.get("model_version") == model.get("model_version")
        and st.session_state.get("predicted_text") == st.session_state.get("ticket_text", "").strip()
    ):
        prediction_view(prediction, model)
    else:
        st.write("")
        cols = st.columns([2, 1])
        with cols[0]:
            st.markdown("#### A routing assistant you can inspect")
            st.write("SignalDesk assigns one of 77 banking support intents. A calibrated probability and validation-selected threshold control whether a ticket can be routed or needs a reviewer.")
            st.caption("Confidence-based review is a heuristic. Out-of-domain messages may still receive high confidence; the example above is a useful stress test.")
        with cols[1]:
            with st.container(border=True):
                st.markdown("**Privacy by default**")
                st.caption("Prediction logs retain intent, confidence, and timing. The service does not retain the submitted message text.")


def safe_csv(frame: pd.DataFrame) -> bytes:
    """Prevent spreadsheet applications from interpreting untrusted text as formulas."""
    def escape(value):
        if isinstance(value, str) and (value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))):
            return "'" + value
        return value
    return frame.map(escape).to_csv(index=False).encode("utf-8")


def batch_lab(model: dict) -> None:
    header("02 / Batch lab", "One file. A clear review queue.", "Score up to 100 messages and download the results for inspection.")
    left, right = st.columns([2.4, 1])
    with left:
        upload = st.file_uploader("Upload a CSV with a text column", type=["csv"], help="Each row must have a nonempty text value of 3–4,000 characters. Maximum 100 rows.")
    with right:
        st.download_button("Download example CSV", data=safe_csv(pd.DataFrame({"text": list(EXAMPLES.values())})), file_name="signaldesk-example.csv", mime="text/csv", width="stretch")
        st.caption("The input file stays in this local app session. Downloads include the messages you supplied.")
    if upload is None:
        st.info("Upload a CSV to preview your messages before running inference.")
        return
    try:
        if upload.size > 2_000_000:
            raise ValueError("The file must be smaller than 2 MB.")
        frame = pd.read_csv(io.BytesIO(upload.getvalue()), dtype=str, keep_default_na=False)
        if "text" not in frame.columns:
            raise ValueError("Add a column named text to your CSV.")
        if not 1 <= len(frame) <= 100:
            raise ValueError("The CSV must contain between 1 and 100 messages.")
        frame["text"] = frame["text"].str.strip()
        invalid = ~frame["text"].str.len().between(3, 4000)
        if invalid.any():
            rows = ", ".join(str(i + 2) for i in frame.index[invalid][:8])
            raise ValueError(f"Each message must contain 3–4,000 characters. Check CSV line(s): {rows}.")
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Cannot read this CSV: {exc}")
        return
    st.dataframe(frame[["text"]], hide_index=True, width="stretch", height=220)
    fingerprint = upload.getvalue()
    if st.button(f"Analyze {len(frame)} messages →", type="primary"):
        try:
            with st.spinner("Processing the batch…"):
                response = api_request("POST", "/predict/batch", {"texts": frame["text"].tolist()})
            rows = [{
                "text": text, "intent": prediction.get("intent"), "confidence": prediction.get("confidence"),
                "decision": prediction.get("decision"), "prediction_id": prediction.get("prediction_id"),
                "model_version": prediction.get("model_version", response.get("model_version")),
            } for text, prediction in zip(frame["text"], response["predictions"])]
            st.session_state["batch_result"] = pd.DataFrame(rows)
            st.session_state["batch_source"] = fingerprint
        except APIError as exc:
            show_api_error(exc)
    result = st.session_state.get("batch_result")
    if result is not None and st.session_state.get("batch_source") == fingerprint:
        st.write("")
        metric_row([
            ("Messages scored", len(result), "Number of messages in this batch."),
            ("Ready to route", int((result["decision"] == "route").sum()), "Predictions meeting the routing threshold."),
            ("Needs review", int((result["decision"] == "review").sum()), "Predictions below the routing threshold."),
            ("Mean confidence", pct(result["confidence"].mean()), "Average calibrated confidence across this batch."),
        ])
        st.write("")
        review_only = st.toggle("Show review queue only")
        display = result[result["decision"] == "review"] if review_only else result
        st.dataframe(display, hide_index=True, width="stretch", column_config={"confidence": st.column_config.NumberColumn("Confidence", format="%.3f")})
        st.download_button("Download scored CSV", safe_csv(result), "signaldesk-predictions.csv", "text/csv")


def evaluation(model: dict) -> None:
    header("03 / Evaluation", "Evidence before deployment.", "Measured results on the held-out official BANKING77 test set. Model and routing decisions use separate development partitions.")
    try:
        report = get_metrics(API_URL, model.get("model_version"))
    except APIError as exc:
        show_api_error(exc)
        return
    test = report.get("test", {})
    policy = report.get("policy", {})
    selective = policy.get("test", {})
    metric_row([
        ("Test accuracy", pct(test.get("accuracy")), "Accuracy over every official test example; no abstention filtering."),
        ("Macro F1", pct(test.get("macro_f1")), "Equal-weight average of per-intent F1 across 77 intents."),
        ("Top-3 accuracy", pct(test.get("top3_accuracy")), "Fraction of examples whose correct intent is in the top three predictions."),
        ("Routing coverage", pct(selective.get("coverage")), "Fraction of test examples meeting the validation-selected threshold."),
    ])
    st.write("")
    comparison_tab, reliability_tab, errors_tab, methodology_tab = st.tabs(["Model comparison", "Confidence & review", "Error analysis", "Experiment details"])
    with comparison_tab:
        st.markdown("#### Candidate selection")
        st.caption("Models are selected using development-set macro F1. These scores are distinct from the final test results above.")
        candidates = report.get("candidates", [])
        if candidates:
            rows = [{"Model": humanize(item["name"]), "Selection accuracy": item.get("selection", {}).get("accuracy"), "Selection macro F1": item.get("selection", {}).get("macro_f1"), "Fit seconds": item.get("fit_seconds"), "Selected": item["name"] == report.get("selected_model")} for item in candidates]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Selection accuracy": st.column_config.NumberColumn(format="%.4f"), "Selection macro F1": st.column_config.NumberColumn(format="%.4f"), "Fit seconds": st.column_config.NumberColumn(format="%.2f")})
        st.success(f'Selected model: {humanize(report.get("selected_model", "unknown"))}')
        st.caption("These CPU-friendly models provide a transparent, reproducible baseline. The project does not claim state-of-the-art benchmark performance.")
    with reliability_tab:
        cols = st.columns([1.4, 1])
        with cols[0]:
            st.markdown("#### Does confidence match correctness?")
            bins = [item for item in report.get("reliability_bins", []) if item.get("count", 0) > 0]
            if bins:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal calibration", line={"color": COLORS["muted"], "dash": "dot"}))
                fig.add_trace(go.Scatter(x=[row["mean_confidence"] for row in bins], y=[row["accuracy"] for row in bins], mode="lines+markers", name="Test bins", marker={"color": COLORS["mint"], "size": 9}, customdata=[row["count"] for row in bins], hovertemplate="Confidence %{x:.1%}<br>Accuracy %{y:.1%}<br>%{customdata} examples<extra></extra>"))
                fig.update_xaxes(title="Mean confidence", range=[0, 1], tickformat=".0%")
                fig.update_yaxes(title="Observed accuracy", range=[0, 1], tickformat=".0%")
                st.plotly_chart(chart_layout(fig, height=350), use_container_width=True)
            st.caption(f'Expected calibration error: {test.get("ece", 0):.4f} · multiclass log loss: {test.get("log_loss", 0):.4f}')
        with cols[1]:
            with st.container(border=True):
                st.markdown("#### Routing policy")
                st.metric("Test accuracy among routed tickets", pct(selective.get("selective_accuracy")))
                st.write(f'Threshold: **{pct(policy.get("threshold"))}**')
                st.write(f'Validation accuracy target: **{pct(policy.get("target_accuracy"))}**')
                st.write(f'Routed test cases: **{selective.get("accepted", 0):,} / {selective.get("total", test.get("n_samples", 0)):,}**')
                st.caption("The target is an empirical validation constraint. It does not guarantee this accuracy on new or out-of-domain traffic.")
                if not policy.get("target_met", True):
                    st.warning("No threshold met the validation target and sample requirement. Automatic routing is disabled.")
    with errors_tab:
        cols = st.columns([1, 1.3])
        with cols[0]:
            st.markdown("#### Most frequent confusions")
            pairs = report.get("confusion_pairs", [])
            if pairs:
                frame = pd.DataFrame(pairs)
                frame["actual"] = frame["actual"].map(humanize)
                frame["predicted"] = frame["predicted"].map(humanize)
                st.dataframe(frame.rename(columns={"actual": "Actual intent", "predicted": "Predicted intent", "count": "Cases"}), hide_index=True, width="stretch")
        with cols[1]:
            st.markdown("#### Per-intent performance")
            classes = report.get("per_class", [])
            if classes:
                frame = pd.DataFrame(classes).sort_values("f1")
                filter_text = st.text_input("Filter intents", placeholder="e.g. transfer")
                if filter_text:
                    frame = frame[frame["intent"].str.contains(filter_text, case=False, regex=False)]
                frame["intent"] = frame["intent"].map(humanize)
                st.dataframe(frame, hide_index=True, width="stretch", height=360, column_config={key: st.column_config.NumberColumn(format="%.3f") for key in ["precision", "recall", "f1"]})
    with methodology_tab:
        sizes = report.get("split_sizes", {})
        st.markdown("#### Separate data for separate decisions")
        descriptions = {"fit": "Feature vocabulary and classifier weights", "calibration": "Probability calibration", "selection": "Model selection", "policy_validation": "Routing threshold selection", "test": "Final reporting only"}
        if sizes:
            st.dataframe(pd.DataFrame([{"Partition": key.replace("_", " ").capitalize(), "Examples": value, "Purpose": descriptions.get(key, "See experiment report")} for key, value in sizes.items()]), hide_index=True, width="stretch")
        st.markdown("#### Data and leakage audit")
        st.json(report.get("leakage_audit", {}), expanded=False)
        st.markdown("#### Reproducibility")
        st.json(report.get("training", {}), expanded=False)
        st.download_button("Download full evaluation JSON", data=json.dumps(report, indent=2), file_name="signaldesk-metrics.json", mime="application/json")


def monitoring(model: dict) -> None:
    header("04 / Monitoring", "Watch the traffic, keep the context.", "Local inference telemetry and confidence-distribution drift. Submitted message text is excluded from the prediction log.")
    if st.button("Refresh telemetry", width="content"):
        pass  # Every page rerun retrieves a fresh telemetry window.
    try:
        monitor = api_request("GET", "/monitoring")
    except APIError as exc:
        show_api_error(exc)
        return
    metric_row([
        ("Predictions logged", f'{monitor.get("total_predictions", 0):,}', "All predictions in the local telemetry store."),
        ("Routing rate", pct(monitor.get("route_rate")), "Fraction of predictions meeting the routing threshold in the current window."),
        ("Mean confidence", pct(monitor.get("mean_confidence")), "Mean top-class probability in the current window."),
        ("Confidence drift · JS", "—" if monitor.get("confidence_drift_js") is None else f'{monitor["confidence_drift_js"]:.4f}', "Jensen–Shannon divergence of current and reference confidence histograms. It does not measure prediction accuracy."),
    ])
    st.write("")
    status = monitor.get("drift_status", "insufficient_data")
    note = monitor.get("drift_note", "")
    if status in {"drift", "elevated", "warning", "shift_detected"}:
        st.warning(note or "The current confidence distribution differs from the reference.")
    elif status in {"insufficient_data", "insufficient", "warming_up"}:
        st.info(f'Collect at least {monitor.get("minimum_window", 100)} predictions before interpreting drift. {note}')
    else:
        st.info(note or "Confidence distribution telemetry is available. Interpret alongside verified feedback.")
    cols = st.columns([1.2, 1])
    with cols[0]:
        st.markdown("#### Confidence distribution")
        hist = monitor.get("confidence_histogram", {})
        reference, current = hist.get("reference", []), hist.get("current", [])
        # The API may supply proportions or raw histogram counts. Compare shares.
        reference = [value / sum(reference) for value in reference] if sum(reference) else reference
        current = [value / sum(current) for value in current] if sum(current) else current
        if reference:
            bins = hist.get("bins", [i / len(reference) for i in range(len(reference) + 1)])
            labels = [f"{float(bins[i]):.0%}–{float(bins[i+1]):.0%}" for i in range(len(reference))]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=reference, name="Reference", marker_color=COLORS["blue"]))
            if current:
                fig.add_trace(go.Bar(x=labels, y=current, name="Current window", marker_color=COLORS["mint"]))
            fig.update_layout(barmode="group")
            fig.update_yaxes(tickformat=".0%", title="Share of predictions")
            st.plotly_chart(chart_layout(fig, height=300), use_container_width=True)
        st.caption(f'Current window: {monitor.get("window_size", 0):,} predictions. Confidence shift is a monitoring signal, not evidence of an accuracy change.')
    with cols[1]:
        st.markdown("#### Most predicted intents")
        rows = monitor.get("intent_distribution", [])
        if rows:
            frame = pd.DataFrame(rows).sort_values("count", ascending=False).head(7)
            fig = go.Figure(go.Bar(x=frame["count"], y=frame["intent"].map(humanize), orientation="h", marker_color=COLORS["mint"], hovertemplate="%{y}<br>%{x} predictions<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(chart_layout(fig, height=300), use_container_width=True)
        else:
            st.info("Analyze a message to start collecting local telemetry.")
    st.markdown("#### Recent predictions")
    recent = monitor.get("recent_predictions", [])
    if recent:
        frame = pd.DataFrame(recent)
        if "intent" in frame:
            frame["intent"] = frame["intent"].map(humanize)
        st.dataframe(frame, hide_index=True, width="stretch")
    st.caption("Feedback is recorded separately for future offline evaluation. The model is never silently updated by incoming traffic.")
    if monitor.get("feedback_count", 0):
        st.caption(f'Verified labels: {monitor["feedback_count"]:,} · accuracy on labeled predictions: {pct(monitor.get("feedback_accuracy"))}. Voluntary feedback may be selection-biased.')


def main() -> None:
    st.set_page_config(page_title="SignalDesk · Support intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    apply_theme()
    with st.sidebar:
        st.markdown('<div class="sd-brand">◈ Signal<span>Desk</span></div><div class="sd-sidebar-caption">Support intelligence, in focus.</div>', unsafe_allow_html=True)
        page = st.radio("WORKSPACE", ["Workspace", "Batch lab", "Evaluation", "Monitor"], label_visibility="collapsed")
        st.divider()
        st.markdown("**MODEL ENVIRONMENT**")
        try:
            health = api_request("GET", "/health")
            loaded = bool(health.get("model_loaded"))
            st.markdown(f'<span class="sd-pill">● {"Model ready" if loaded else "Service online"}</span>', unsafe_allow_html=True)
        except APIError:
            health, loaded = {}, False
            st.caption("○ Service offline")
        st.write("")
        st.caption("BANKING77 · 77 support intents")
        st.caption("Local inference · CPU runtime")
        st.caption("Python · scikit-learn · FastAPI")
        st.divider()
        st.markdown('<div class="sd-footnote">A reproducible ML engineering project.<br>Inspect predictions. Measure errors.<br>Keep a human in the loop.</div>', unsafe_allow_html=True)
    if not loaded:
        header("Local ML workspace", "Your inference workspace is almost ready.", "Start the API with a trained model, then refresh this page.")
        st.info("The dashboard is running. The model service is not ready yet.")
        st.code("python -m signaldesk.data\npython -m signaldesk.train\nuvicorn signaldesk.api:app --host 127.0.0.1 --port 8000", language="bash")
        st.caption(f"Expected API: {API_URL}. Keep the API running in a separate terminal.")
        if st.button("Check connection", type="primary"):
            get_model.clear()
            st.rerun()
        return
    try:
        model = get_model(API_URL, health.get("model_version"))
    except APIError as exc:
        show_api_error(exc)
        return
    with st.sidebar:
        st.caption(f'Model version: {model.get("model_version", health.get("model_version", "unknown"))}')
        st.caption(f'Route threshold: {pct(model.get("threshold"))}')
    {"Workspace": workspace, "Batch lab": batch_lab, "Evaluation": evaluation, "Monitor": monitoring}[page](model)


if __name__ == "__main__":
    main()
