"""Exercise the actual distributed model rather than substituting a fake predictor."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from signaldesk.api import create_app
from signaldesk.data import read_rows
from signaldesk.modeling import load_bundle, predict
from signaldesk.policy import policy_metrics
from signaldesk.train import score_probabilities

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def trained_bundle():
    return load_bundle(ROOT / "artifacts")


def test_actual_artifact_has_normalized_class_aligned_probabilities(trained_bundle):
    texts = ["The ATM swallowed my card.", "I forgot the PIN for my card.", "My transfer is pending."]
    probabilities = trained_bundle["classifier"].predict_proba(trained_bundle["vectorizer"].transform(texts))
    assert probabilities.shape == (3, 77)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert trained_bundle["classes"] == list(trained_bundle["classifier"].classes_)
    direct = predict(trained_bundle, texts)
    classes = np.asarray(trained_bundle["classes"])
    assert [row["intent"] for row in direct] == classes[probabilities.argmax(axis=1)].tolist()


def test_official_test_report_matches_distributed_artifact(trained_bundle):
    rows = read_rows(ROOT / "data/raw/test.csv")
    labels = np.asarray([row["category"] for row in rows])
    features = trained_bundle["vectorizer"].transform([row["text"] for row in rows])
    probabilities = trained_bundle["classifier"].predict_proba(features)
    report = json.loads((ROOT / "artifacts/metrics.json").read_text())
    actual = score_probabilities(probabilities, labels, trained_bundle["classes"])
    for metric, value in actual.items():
        assert value == pytest.approx(report["test"][metric], abs=1e-12)
    correct = np.asarray(trained_bundle["classes"])[probabilities.argmax(axis=1)] == labels
    routed = policy_metrics(probabilities.max(axis=1), correct, trained_bundle["threshold"])
    assert routed == report["policy"]["test"]


def test_live_model_api_single_batch_and_direct_predictions_agree(trained_bundle, tmp_path):
    texts = ["The ATM swallowed my card.", "I ordered pizza and it arrived cold."]
    direct = predict(trained_bundle, texts)
    with TestClient(create_app(ROOT / "artifacts", tmp_path / "events.sqlite3")) as client:
        first = client.post("/predict", json={"text": texts[0]}).json()
        response = client.post("/predict/batch", json={"texts": texts})
        assert response.status_code == 200
        batch = response.json()["predictions"]
        for index, expected in enumerate(direct):
            for field in ("intent", "confidence", "decision", "top_intents", "similar_tickets", "explanations"):
                assert batch[index][field] == expected[field]
                if index == 0:
                    assert first[field] == expected[field]
        summary = client.get("/monitoring").json()
        assert summary["total_predictions"] == 3
        assert summary["model_version"] == trained_bundle["metadata"]["model_version"]
        assert summary["text_stored"] is False
