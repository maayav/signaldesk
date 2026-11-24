"""Shared artifact loading and inference for API, batch jobs, and experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from signaldesk.retrieval import similar_tickets


def load_bundle(path: Path | str) -> dict:
    """Load trusted local joblib after verifying every declared artifact file.

    A checksum detects corruption, not a malicious artifact whose manifest was
    also replaced. Never load artifacts uploaded by an untrusted user.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "model.joblib"
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"Invalid artifact manifest: {manifest_path}")
    declared_files = manifest["files"]
    if path.name not in declared_files:
        raise ValueError(f"Model file is absent from artifact manifest: {path.name}")
    for filename, specification in declared_files.items():
        if (not isinstance(filename, str) or Path(filename).name != filename
                or filename in {".", ".."} or not isinstance(specification, dict)):
            raise ValueError(f"Invalid artifact file entry in {manifest_path}")
        expected = specification.get("sha256")
        if (not isinstance(expected, str) or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)):
            raise ValueError(f"Invalid artifact checksum for {filename}")
        artifact_path = path.parent / filename
        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {artifact_path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported artifact schema version")
    return bundle


def explain_features(bundle: dict, vector, intent: str, k: int = 8) -> list[dict]:
    """Positive contributions to the raw linear margin (not calibrated probability)."""
    classifier = bundle["raw_classifier"]
    classes = list(classifier.classes_)
    class_index = classes.index(intent)
    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[class_index]
    else:
        coefficients = classifier.estimators_[class_index].coef_[0]
    if hasattr(coefficients, "toarray"):
        coefficients = coefficients.toarray().ravel()
    contributions = vector.data * np.asarray(coefficients).ravel()[vector.indices]
    feature_names = bundle["feature_names"]
    order = np.argsort(-contributions, kind="stable")[:k]
    return [
        {"feature": str(feature_names[vector.indices[int(index)]]), "contribution": float(contributions[index])}
        for index in order if contributions[index] > 0
    ]


def predict(bundle: dict, texts: list[str]) -> list[dict]:
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text must be a non-empty string")
    features = bundle["vectorizer"].transform(texts).tocsr()
    probabilities = bundle["classifier"].predict_proba(features)
    classes = np.asarray(bundle["classifier"].classes_)
    results = []
    for index, (text, scores) in enumerate(zip(texts, probabilities, strict=True)):
        top = np.argsort(-scores, kind="stable")[:3]
        intent, confidence = str(classes[top[0]]), float(scores[top[0]])
        # Featureless inputs should never route, even if calibration assigns a prior.
        route = confidence >= bundle["threshold"] and features[index].nnz > 0
        results.append({
            "intent": intent,
            "confidence": confidence,
            "decision": "route" if route else "review",
            "top_intents": [{"intent": str(classes[j]), "confidence": float(scores[j])} for j in top],
            "similar_tickets": similar_tickets(bundle, text),
            "explanations": explain_features(bundle, features[index], intent),
        })
    return results
