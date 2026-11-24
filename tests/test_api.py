import importlib
import json

import pytest
from fastapi.testclient import TestClient

api = importlib.import_module("signaldesk.api")


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake_bundle = {
        "metadata": {"model_version": "test-v1", "model_name": "test"},
        "threshold": 0.8, "classes": ["card_arrival", "lost_or_stolen_card"],
        "reference": {"confidence_histogram": [0.1] * 10}, "metrics": {"accuracy": 0.9},
    }
    monkeypatch.setattr(api, "load_bundle", lambda _: fake_bundle)

    def fake_predict(bundle, texts):
        return [{"intent": "card_arrival", "confidence": 0.9, "decision": "route",
                 "top_intents": [{"intent": "card_arrival", "confidence": 0.9}],
                 "similar_tickets": [], "explanations": []} for _ in texts]

    monkeypatch.setattr(api, "predict", fake_predict)
    with TestClient(api.create_app(tmp_path, tmp_path / "events.db")) as session:
        yield session


def test_lifecycle_prediction_batch_and_feedback(client):
    assert client.get("/readyz").status_code == 200
    assert client.get("/health").json()["model_loaded"]
    result = client.post("/predict", json={"text": "Where is my card?"})
    assert result.status_code == 200
    row = result.json()
    assert row["model_version"] == "test-v1" and row["latency_ms"] >= 0
    result = client.post("/predict/batch", json={"texts": ["First ticket", "Second ticket"]})
    assert result.status_code == 200 and result.json()["count"] == 2
    assert len({p["prediction_id"] for p in result.json()["predictions"]}) == 2
    assert client.get("/monitoring").json()["total_predictions"] == 3
    assert client.post("/feedback", json={"prediction_id": row["prediction_id"],
                                         "correct_intent": "card_arrival"}).status_code == 200
    assert client.get("/monitoring").json()["feedback_accuracy"] == 1
    assert client.post("/feedback", json={"prediction_id": "missing",
                                         "correct_intent": "card_arrival"}).status_code == 404
    assert client.post("/feedback", json={"prediction_id": row["prediction_id"],
                                         "correct_intent": "invalid"}).status_code == 422


@pytest.mark.parametrize("body", [{"text": " "}, {"text": "ab"}, {"text": "a" * 4001},
                                        {"text": 123}, {"text": "abc", "other": True}, {}])
def test_prediction_validation(client, body):
    assert client.post("/predict", json=body).status_code == 422
    assert client.get("/monitoring").json()["total_predictions"] == 0


@pytest.mark.parametrize("texts", [[], ["valid"] * 101, ["valid", " "]])
def test_batch_validation_atomic(client, texts):
    assert client.post("/predict/batch", json={"texts": texts}).status_code == 422
    assert client.get("/monitoring").json()["total_predictions"] == 0


def test_body_size_limit(client):
    response = client.post("/predict", content=json.dumps({"text": "a" * 2_000_001}),
                           headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_unavailable_model_health_is_distinct_from_readiness(tmp_path, monkeypatch):
    def missing(_):
        raise FileNotFoundError("missing model")
    monkeypatch.setattr(api, "load_bundle", missing)
    with TestClient(api.create_app(tmp_path, tmp_path / "events.db")) as client:
        assert client.get("/health").status_code == 200
        assert not client.get("/health").json()["model_loaded"]
        assert client.get("/readyz").status_code == 503
        assert client.post("/predict", json={"text": "Where is my card?"}).status_code == 503


@pytest.mark.parametrize("metrics", ["{broken", "[]"])
def test_invalid_metrics_never_marks_partially_loaded_model_ready(tmp_path, monkeypatch, metrics):
    monkeypatch.setattr(api, "load_bundle", lambda _: {"metadata": {"model_version": "test-v1"}})
    (tmp_path / "metrics.json").write_text(metrics)
    with TestClient(api.create_app(tmp_path, tmp_path / "events.db")) as client:
        assert client.get("/health").json()["model_loaded"] is False
        for endpoint in ("/readyz", "/metrics", "/model", "/monitoring"):
            assert client.get(endpoint).status_code == 503
        assert client.post("/predict", json={"text": "Where is my card?"}).status_code == 503
