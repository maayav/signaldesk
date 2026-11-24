"""Reproduce feature engineering, calibrated candidate selection, and final evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from signaldesk.data import FILES, SOURCE_COMMIT, SOURCE_URL, clean_training_rows, normalized_text, read_rows, sha256, split_training_rows
from signaldesk.policy import policy_metrics, select_threshold
from signaldesk.retrieval import build_reference


def reliability_bins(probabilities, labels, classes, n_bins: int = 10) -> list[dict]:
    confidence = probabilities.max(axis=1)
    correct = np.asarray(classes)[probabilities.argmax(axis=1)] == np.asarray(labels)
    assignment = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    return [
        {
            "lower": index / n_bins,
            "upper": (index + 1) / n_bins,
            "count": int((assignment == index).sum()),
            "mean_confidence": float(confidence[assignment == index].mean()) if (assignment == index).any() else None,
            "accuracy": float(correct[assignment == index].mean()) if (assignment == index).any() else None,
        }
        for index in range(n_bins)
    ]


def score_probabilities(probabilities, labels, classes) -> dict:
    classes, labels = np.asarray(classes), np.asarray(labels)
    predicted = classes[probabilities.argmax(axis=1)]
    bins = reliability_bins(probabilities, labels, classes)
    top_three = classes[np.argsort(-probabilities, axis=1)[:, :3]]
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predicted, labels=classes, average="weighted", zero_division=0)),
        "top3_accuracy": float(np.any(top_three == labels[:, None], axis=1).mean()),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "ece": float(sum(item["count"] / len(labels) * abs(item["accuracy"] - item["mean_confidence"]) for item in bins if item["count"])),
        "n_samples": int(len(labels)),
    }


def make_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25_000, sublinear_tf=True, strip_accents="unicode")),
            ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40_000, sublinear_tf=True)),
        ],
        transformer_weights={"word": 1.0, "character": 0.7},
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run_training(data_dir: Path, artifact_dir: Path, seed: int = 42) -> dict:
    data_dir, artifact_dir = Path(data_dir), Path(artifact_dir)
    started = time.perf_counter()
    original_train = read_rows(data_dir / "train.csv")
    test = read_rows(data_dir / "test.csv")
    train, audit = clean_training_rows(original_train, test)
    splits = split_training_rows(train, seed)
    splits["test"] = test
    texts = {name: [row["text"] for row in rows] for name, rows in splits.items()}
    labels = {name: np.asarray([row["category"] for row in rows]) for name, rows in splits.items()}
    print("Fitting features on fit partition only...", flush=True)
    vectorizer = make_vectorizer()
    features = {"fit": vectorizer.fit_transform(texts["fit"]).tocsr()}
    for name in splits:
        if name != "fit":
            features[name] = vectorizer.transform(texts[name]).tocsr()
    estimators = {
        "logistic_regression": OneVsRestClassifier(LogisticRegression(C=4.0, solver="liblinear", max_iter=600, random_state=seed), n_jobs=1),
        "linear_svc": LinearSVC(C=1.5, dual="auto", max_iter=3000, random_state=seed),
    }
    candidates, fitted = [], {}
    # Keep CPU/memory use predictable across laptops and CI runners.
    with threadpool_limits(limits=2):
        for name, raw in estimators.items():
            candidate_start = time.perf_counter()
            print(f"Training and calibrating {name}...", flush=True)
            raw.fit(features["fit"], labels["fit"])
            calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method="sigmoid")
            calibrated.fit(features["calibration"], labels["calibration"])
            selection_probabilities = calibrated.predict_proba(features["selection"])
            scores = score_probabilities(selection_probabilities, labels["selection"], calibrated.classes_)
            candidates.append({"name": name, "selection": scores, "fit_seconds": round(time.perf_counter() - candidate_start, 3)})
            fitted[name] = (raw, calibrated)
            print(f"{name}: selection macro F1 = {scores['macro_f1']:.4f}", flush=True)
    # Tie-break by candidate order only; test results never enter this choice.
    winner = max(candidates, key=lambda item: item["selection"]["macro_f1"])["name"]
    raw, calibrated = fitted[winner]
    classes = np.asarray(calibrated.classes_)
    policy_probabilities = calibrated.predict_proba(features["policy_validation"])
    policy_correct = classes[policy_probabilities.argmax(axis=1)] == labels["policy_validation"]
    policy = select_threshold(policy_probabilities.max(axis=1), policy_correct)
    print(f"Selected {winner}; threshold = {policy['threshold']:.4f}. Evaluating untouched official test...", flush=True)
    test_probabilities = calibrated.predict_proba(features["test"])
    test_predictions = classes[test_probabilities.argmax(axis=1)]
    test_scores = score_probabilities(test_probabilities, labels["test"], classes)
    policy["test"] = policy_metrics(test_probabilities.max(axis=1), test_predictions == labels["test"], policy["threshold"])
    report = classification_report(labels["test"], test_predictions, labels=classes, output_dict=True, zero_division=0)
    confusion: dict[tuple[str, str], int] = {}
    for actual, predicted in zip(labels["test"], test_predictions, strict=True):
        if actual != predicted:
            pair = (str(actual), str(predicted))
            confusion[pair] = confusion.get(pair, 0) + 1
    reference_predictions = classes[policy_probabilities.argmax(axis=1)]
    reference = {
        "source": "policy_validation",
        "confidence_histogram": (np.histogram(policy_probabilities.max(axis=1), bins=np.linspace(0, 1, 11))[0] / len(policy_probabilities)).tolist(),
        "intent_distribution": {str(intent): float((reference_predictions == intent).mean()) for intent in classes},
    }
    code_digest = hashlib.sha256()
    code_hashes = {}
    for filename in ("data.py", "modeling.py", "policy.py", "retrieval.py", "train.py"):
        content = (Path(__file__).parent / filename).read_bytes()
        code_hashes[filename] = hashlib.sha256(content).hexdigest()
        code_digest.update(content)
    config = {"seed": seed, "source_commit": SOURCE_COMMIT, "sklearn": sklearn.__version__, "code_sha256": code_digest.hexdigest()}
    version = "banking77-" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    metrics = {
        "schema_version": 1,
        "model_version": version,
        "dataset": {"name": "BANKING77", "source_url": SOURCE_URL, "original_train_size": len(original_train), "official_test_size": len(test), "n_classes": len(classes)},
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "leakage_audit": audit,
        "selected_model": winner,
        "candidates": candidates,
        "test": test_scores,
        "policy": policy,
        "per_class": [{"intent": str(intent), "precision": report[intent]["precision"], "recall": report[intent]["recall"], "f1": report[intent]["f1-score"], "support": int(report[intent]["support"])} for intent in classes],
        "confusion_pairs": [{"actual": actual, "predicted": predicted, "count": count} for (actual, predicted), count in sorted(confusion.items(), key=lambda item: (-item[1], item[0]))[:30]],
        "reliability_bins": reliability_bins(test_probabilities, labels["test"], classes),
        "training": {"seed": seed, "duration_seconds": round(time.perf_counter() - started, 3), "python": platform.python_version(), "sklearn": sklearn.__version__, "feature_count": int(features["fit"].shape[1]), "code_hashes": code_hashes},
        "reference": reference,
        "methodology": {
            "selection": "Highest selection-partition macro F1 after independent sigmoid calibration.",
            "policy": "Maximum policy-validation coverage satisfying empirical accuracy >=0.95 and >=100 accepted; all confidence ties accepted together.",
            "evaluation": "One final evaluation on all 3080 official test rows; no refit or threshold changes afterward.",
            "explanations": "Positive feature contributions to the selected class raw linear margin; not probability attribution.",
            "retrieval": "Word TF-IDF cosine similarity over the fit partition only; not neural semantic retrieval.",
        },
    }
    metadata = {"model_version": version, "model_name": winner, "threshold": policy["threshold"], "classes": classes.tolist(), "created_at": datetime.now(timezone.utc).isoformat(), "source_commit": SOURCE_COMMIT, "sklearn_version": sklearn.__version__}
    bundle = {
        "schema_version": 1,
        "vectorizer": vectorizer,
        "classifier": calibrated,
        "raw_classifier": raw,
        "threshold": policy["threshold"],
        "classes": classes.tolist(),
        "feature_names": vectorizer.get_feature_names_out(),
        "train_texts": texts["fit"],
        "train_labels": labels["fit"].tolist(),
        "retrieval_matrix": build_reference(vectorizer, texts["fit"]),
        "metadata": metadata,
        "reference": reference,
        "metrics": metrics,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    metrics["training"]["model_size_bytes"] = model_path.stat().st_size
    _write_json(artifact_dir / "metrics.json", metrics)
    split_manifest = {
        "seed": seed,
        "description": "SHA256 of normalized input texts, in partition order; no official test label used for model/policy selection.",
        "partitions": {name: [hashlib.sha256(normalized_text(row["text"]).encode()).hexdigest() for row in rows] for name, rows in splits.items()},
    }
    _write_json(artifact_dir / "splits.json", split_manifest)
    manifest = {
        "schema_version": 1,
        "model_version": version,
        "created_at": metadata["created_at"],
        "data_source_commit": SOURCE_COMMIT,
        "data_sha256": {filename: digest for filename, (_, digest) in FILES.items()},
        "files": {filename: {"sha256": sha256(artifact_dir / filename), "size_bytes": (artifact_dir / filename).stat().st_size} for filename in ("model.joblib", "metrics.json", "splits.json")},
        "warning": "Joblib is executable serialization: only load trusted local artifacts. Checksums detect corruption, not malicious replacement.",
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(json.dumps({"selected_model": winner, "test": test_scores, "policy": policy, "model_version": version}, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_training(args.data_dir, args.artifact_dir, args.seed)


if __name__ == "__main__":
    main()
