import hashlib
import json

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from signaldesk.modeling import load_bundle, predict
from signaldesk.retrieval import build_reference
from signaldesk.train import make_vectorizer, score_probabilities


@pytest.fixture
def tiny_bundle():
    texts = ["card lost stolen", "lost stolen bank card", "cash transfer pending", "bank transfer cash pending", "phone password reset", "reset password phone"]
    labels = ["card", "card", "transfer", "transfer", "password", "password"]
    vectorizer = make_vectorizer()
    features = vectorizer.fit_transform(texts)
    raw = OneVsRestClassifier(LogisticRegression(solver="liblinear")).fit(features, labels)
    classifier = CalibratedClassifierCV(FrozenEstimator(raw), cv=2).fit(features, labels)
    return {"schema_version": 1, "vectorizer": vectorizer, "raw_classifier": raw, "classifier": classifier, "threshold": 0.0, "feature_names": vectorizer.get_feature_names_out(), "train_texts": texts, "train_labels": labels, "retrieval_matrix": build_reference(vectorizer, texts)}


def test_predictions_retrieval_and_explanations_share_trained_artifact(tiny_bundle):
    result = predict(tiny_bundle, ["my lost stolen card"])[0]
    assert result["intent"] == "card"
    assert result["decision"] == "route"
    assert result["top_intents"][0]["confidence"] == result["confidence"]
    assert all(item["text"] in tiny_bundle["train_texts"] for item in result["similar_tickets"])
    assert result["explanations"]
    assert all(item["contribution"] > 0 for item in result["explanations"])
    assert predict(tiny_bundle, ["☃☃☃"])[0]["decision"] == "review"
    assert predict(tiny_bundle, []) == []
    with pytest.raises(ValueError):
        predict(tiny_bundle, [" "])


def test_artifact_roundtrip_and_corruption_guard(tiny_bundle, tmp_path):
    model = tmp_path / "model.joblib"
    joblib.dump(tiny_bundle, model)
    manifest = {"files": {"model.joblib": {"sha256": hashlib.sha256(model.read_bytes()).hexdigest()}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert predict(load_bundle(tmp_path), ["card stolen"]) == predict(tiny_bundle, ["card stolen"])
    model.write_bytes(model.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(model)


def test_artifact_sidecars_are_verified_before_deserialization(tiny_bundle, tmp_path):
    model = tmp_path / "model.joblib"
    metrics = tmp_path / "metrics.json"
    joblib.dump(tiny_bundle, model)
    metrics.write_text('{"test": {"accuracy": 0.9}}')
    manifest = {"files": {
        path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (model, metrics)
    }}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert load_bundle(model)["schema_version"] == 1
    metrics.write_text('{"test": {"accuracy": 1.0}}')
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bundle(model)
    metrics.unlink()
    with pytest.raises(FileNotFoundError):
        load_bundle(model)


@pytest.mark.parametrize("manifest", [[], {}, {"files": []}, {"files": {"model.joblib": {}}},
                                        {"files": {"../model.joblib": {"sha256": "0" * 64}}}])
def test_invalid_artifact_manifest_has_a_clear_error(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest|checksum"):
        load_bundle(tmp_path / "model.joblib")


def test_metrics_use_probability_class_order():
    scores = score_probabilities(np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]), ["a", "b"], ["a", "b", "c"])
    assert scores["accuracy"] == 0.5
    assert scores["top3_accuracy"] == 1.0
    assert scores["ece"] == pytest.approx(0.45)
