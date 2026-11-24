"""Exercise the Streamlit client against the real API without network ports.

AppTest does not support the file-uploader widget yet, so uploaded bytes are
injected at that widget boundary. Inference, telemetry, feedback, and request
validation still execute in the actual FastAPI application with the trained
model and an isolated database.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd
import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from signaldesk.api import create_app


PROJECT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT / "app" / "dashboard.py"
ROUTE_TEXT = "The ATM kept my card and will not give it back."
REVIEW_TEXT = "xyzzy quux blorp"


class UploadedCSV(io.BytesIO):
    @property
    def size(self):
        return len(self.getvalue())


@pytest.fixture(scope="module")
def trained_artifacts():
    directory = Path(os.getenv("SIGNALDESK_TEST_ARTIFACT_DIR", PROJECT / "artifacts"))
    if not (directory / "model.joblib").exists():
        pytest.skip("Train a model first; set SIGNALDESK_TEST_ARTIFACT_DIR to test another artifact directory")
    return directory


@pytest.fixture
def dashboard(tmp_path, monkeypatch, trained_artifacts):
    monkeypatch.syspath_prepend(str(PROJECT / "app"))
    # A distinct URL prevents Streamlit's data cache crossing test boundaries.
    base_url = f"http://dashboard-{uuid4().hex}.test"
    monkeypatch.setenv("SIGNALDESK_API_URL", base_url)
    calls, downloads = [], {}
    upload = {"file": None}
    failures = {}
    health_override = {}
    original_download_button = st.download_button

    def capture_download(*args, **kwargs):
        label = args[0] if args else kwargs["label"]
        data = args[1] if len(args) > 1 else kwargs.get("data")
        downloads[label] = data
        return original_download_button(*args, **kwargs)

    with TestClient(create_app(trained_artifacts, tmp_path / "dashboard.sqlite3")) as client:
        def bridge(method, url, **kwargs):
            endpoint = urlsplit(url).path
            payload = kwargs.get("json")
            calls.append((method, endpoint, payload))
            if endpoint in failures:
                failure = failures[endpoint]
                if isinstance(failure, Exception):
                    raise failure
                response = requests.Response()
                response.status_code = failure
                response.reason = "Service Unavailable"
                response._content = b'{"detail":"Temporarily unavailable"}'
                return response
            actual = client.request(method, endpoint, json=payload)
            response = requests.Response()
            response.status_code = actual.status_code
            response.reason = actual.reason_phrase
            response._content = actual.content
            if endpoint == "/health" and health_override:
                response._content = json.dumps(health_override).encode()
            return response

        monkeypatch.setattr(requests, "request", bridge)
        monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload["file"])
        monkeypatch.setattr(st, "download_button", capture_download)
        yield SimpleNamespace(
            at=AppTest.from_file(str(DASHBOARD), default_timeout=30),
            client=client,
            calls=calls,
            downloads=downloads,
            upload=upload,
            failures=failures,
            health_override=health_override,
        )


def run(at):
    at.run()
    assert not at.exception, [error.value for error in at.exception]
    return at


def click(at, label):
    next(button for button in at.button if button.label == label).click()
    return run(at)


def navigate(at, page):
    at.radio[0].set_value(page)
    return run(at)


def metrics(at):
    return {metric.label: metric.value for metric in at.metric}


def markdown(at):
    return "\n".join(item.value for item in at.markdown)


@pytest.mark.integration
def test_route_review_examples_feedback_and_stale_result(dashboard):
    at = run(dashboard.at)
    click(at, "Card swallowed")
    assert at.text_area[0].value == "The ATM kept my card and won't give it back."
    click(at, "Analyze message →")
    assert "Ready to route" in markdown(at)
    assert "Model confidence" in metrics(at)
    assert len(at.selectbox) == 1
    click(at, "Save verified label")
    assert any("Verified label recorded" in message.value for message in at.success)
    assert dashboard.client.get("/monitoring").json()["feedback_count"] == 1

    # Editing input must not continue displaying a result for the old ticket.
    at.text_area[0].set_value(REVIEW_TEXT)
    run(at)
    assert "Ready to route" not in markdown(at)
    click(at, "Analyze message →")
    assert "Human review requested" in markdown(at)
    click(at, "Outside the domain")
    assert "recommend a good movie" in at.text_area[0].value
    assert "Human review requested" not in markdown(at)
    assert dashboard.client.get("/monitoring").json()["total_predictions"] == 2


@pytest.mark.integration
@pytest.mark.parametrize("text", ["", "  ", "ab"])
def test_short_input_is_rejected_before_inference(dashboard, text):
    at = run(dashboard.at)
    at.text_area[0].set_value(text)
    click(at, "Analyze message →")
    assert any("at least three characters" in warning.value for warning in at.warning)
    assert not any(endpoint == "/predict" for _, endpoint, _ in dashboard.calls)
    assert dashboard.client.get("/monitoring").json()["total_predictions"] == 0


@pytest.mark.integration
def test_batch_review_filter_and_formula_safe_download(dashboard):
    source = pd.DataFrame({"text": [ROUTE_TEXT, REVIEW_TEXT, '=HYPERLINK("https://example.com")']})
    dashboard.upload["file"] = UploadedCSV(source.to_csv(index=False).encode())
    at = run(dashboard.at)
    navigate(at, "Batch lab")
    click(at, "Analyze 3 messages →")
    assert metrics(at)["Messages scored"] == "3"
    assert metrics(at)["Ready to route"] == "1"
    assert metrics(at)["Needs review"] == "2"
    exported = pd.read_csv(io.BytesIO(dashboard.downloads["Download scored CSV"]))
    assert len(exported) == 3
    assert exported.loc[2, "text"].startswith("'=HYPERLINK")
    assert exported["prediction_id"].nunique() == 3
    at.toggle[0].set_value(True)
    run(at)
    filtered = at.dataframe[-1].value
    assert len(filtered) == 2 and set(filtered["decision"]) == {"review"}
    # The download intentionally includes the complete scored batch.
    assert len(pd.read_csv(io.BytesIO(dashboard.downloads["Download scored CSV"]))) == 3
    dashboard.upload["file"] = UploadedCSV(b"text\nA completely different ticket\n")
    run(at)
    assert "Messages scored" not in metrics(at)
    assert dashboard.client.get("/monitoring").json()["total_predictions"] == 3


@pytest.mark.integration
@pytest.mark.parametrize("contents,expected", [
    (b"", "Cannot read this CSV"),
    (b"message\nWhere is my card?\n", "column named text"),
    (b"text\n", "between 1 and 100"),
    (b"text\n" + b"Where is my card?\n" * 101, "between 1 and 100"),
    (b"text\nab\n", "3–4,000 characters"),
    (b'text\n"unfinished', "Cannot read this CSV"),
    (b"text\n\xff\n", "Cannot read this CSV"),
    (b"text\n" + b"x" * 4001, "3–4,000 characters"),
    (b"text\n" + b"x" * 2_000_001, "smaller than 2 MB"),
])
def test_bad_csv_is_rejected_without_partial_predictions(dashboard, contents, expected):
    dashboard.upload["file"] = UploadedCSV(contents)
    at = run(dashboard.at)
    navigate(at, "Batch lab")
    assert any(expected in error.value for error in at.error)
    assert not any(endpoint == "/predict/batch" for _, endpoint, _ in dashboard.calls)
    assert dashboard.client.get("/monitoring").json()["total_predictions"] == 0


@pytest.mark.integration
def test_evaluation_displays_service_metrics_and_downloads_full_report(dashboard):
    report = dashboard.client.get("/metrics").json()
    at = run(dashboard.at)
    navigate(at, "Evaluation")
    observed = metrics(at)
    assert observed["Test accuracy"] == f'{report["test"]["accuracy"]:.1%}'
    assert observed["Macro F1"] == f'{report["test"]["macro_f1"]:.1%}'
    assert observed["Top-3 accuracy"] == f'{report["test"]["top3_accuracy"]:.1%}'
    assert observed["Routing coverage"] == f'{report["policy"]["test"]["coverage"]:.1%}'
    assert observed["Test accuracy among routed tickets"] == f'{report["policy"]["test"]["selective_accuracy"]:.1%}'
    assert json.loads(dashboard.downloads["Download full evaluation JSON"]) == report
    at.text_input[0].set_value("transfer")
    run(at)
    per_class = at.dataframe[-2].value
    assert len(per_class) and per_class["intent"].str.contains("transfer", case=False).all()


@pytest.mark.integration
def test_metadata_cache_refreshes_when_service_reports_new_model_version(dashboard):
    at = run(dashboard.at)
    assert sum(endpoint == "/model" for _, endpoint, _ in dashboard.calls) == 1
    # Restarting the API with another model must immediately refresh the UI's
    # routing threshold, rather than showing the old model for the cache TTL.
    dashboard.health_override.update({"model_loaded": True, "model_version": "reloaded-model"})
    run(at)
    assert sum(endpoint == "/model" for _, endpoint, _ in dashboard.calls) == 2


@pytest.mark.integration
def test_monitoring_empty_then_populated_and_verified_feedback(dashboard):
    at = run(dashboard.at)
    navigate(at, "Monitor")
    assert metrics(at)["Predictions logged"] == "0"
    assert any("at least 100" in message.value for message in at.info)
    result = dashboard.client.post("/predict", json={"text": ROUTE_TEXT}).json()
    dashboard.client.post("/feedback", json={
        "prediction_id": result["prediction_id"], "correct_intent": result["intent"],
    })
    click(at, "Refresh telemetry")
    assert metrics(at)["Predictions logged"] == "1"
    assert len(at.dataframe[-1].value) == 1
    assert any("Verified labels: 1" in caption.value for caption in at.caption)


@pytest.mark.integration
@pytest.mark.parametrize("model_online", [False, True])
def test_unavailable_service_and_missing_model_show_recovery_instructions(dashboard, model_online):
    if model_online:
        dashboard.health_override.update({"model_loaded": False, "status": "degraded"})
    else:
        dashboard.failures["/health"] = requests.ConnectionError("Service offline")
    at = run(dashboard.at)
    assert any("not ready yet" in message.value for message in at.info)
    assert any("uvicorn signaldesk.api:app" in code.value for code in at.code)
    assert not at.text_area
    dashboard.failures.clear()
    dashboard.health_override.clear()
    click(at, "Check connection")
    assert len(at.text_area) == 1


@pytest.mark.integration
@pytest.mark.parametrize("page,endpoint", [("Workspace", "/predict"), ("Evaluation", "/metrics"), ("Monitor", "/monitoring")])
def test_api_failure_is_displayed_without_crashing(dashboard, page, endpoint):
    at = run(dashboard.at)
    dashboard.failures[endpoint] = 503
    if page == "Workspace":
        at.text_area[0].set_value(ROUTE_TEXT)
        click(at, "Analyze message →")
    else:
        navigate(at, page)
    assert any("503" in error.value and "Temporarily unavailable" in error.value for error in at.error)


def test_csv_export_escapes_formula_prefixes_without_changing_plain_text(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "app"))
    spec = importlib.util.spec_from_file_location("signaldesk_dashboard_helpers", DASHBOARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = ["=1+1", "+SUM(A1)", "-SUM(A1)", "@example", "  =1", "\ttext", "\rtext", "\ntext", "Normal text", "", None]
    exported = pd.read_csv(io.BytesIO(module.safe_csv(pd.DataFrame({"text": values}))), keep_default_na=False)
    assert exported["text"].tolist()[:8] == ["'" + text for text in values[:8]]
    assert exported["text"].tolist()[8:] == ["Normal text", "", ""]
