import sqlite3

import numpy as np
import pytest

from signaldesk.monitoring import EventStore, jensen_shannon


def event(i, version="v1", confidence=0.8):
    return {"prediction_id": str(i), "model_version": version, "intent": "card_arrival",
            "confidence": confidence, "decision": "route", "latency_ms": 1.5,
            "text": "SENSITIVE INPUT MUST NEVER BE STORED"}


def test_divergence_bounds_and_zero_bins():
    assert jensen_shannon([1, 0], [1, 0]) == 0
    assert jensen_shannon([1, 0], [0, 1]) == 1
    assert jensen_shannon([0, 0], [0, 1]) == 0
    with pytest.raises(ValueError):
        jensen_shannon([-1, 2], [1, 0])


def test_storage_allowlist_feedback_and_model_isolation(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    store.record([event(1), event(2, "other")])
    assert store.feedback("missing", "card_arrival") is False
    assert store.feedback("1", "lost_or_stolen_card")
    assert store.feedback("1", "card_arrival")  # Correction updates, not duplicated labels.
    summary = store.summary("v1")
    assert summary["total_predictions"] == summary["window_size"] == summary["feedback_count"] == 1
    assert summary["feedback_accuracy"] == 1
    assert summary["drift_status"] == "insufficient_data"
    with sqlite3.connect(store.path) as con:
        assert "SENSITIVE" not in " ".join(con.iterdump())


def test_window_and_drift_minimum(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    store.record([event(i, confidence=0.99) for i in range(120)])
    ref = {"bins": np.linspace(0, 1, 11), "counts": [1] + [0] * 9}
    result = store.summary("v1", ref, window=100)
    assert result["total_predictions"] == 120
    assert result["window_size"] == 100
    assert result["drift_status"] == "shift_detected"
    assert result["confidence_drift_js"] == 1
