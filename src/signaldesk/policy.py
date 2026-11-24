"""An empirical precision/coverage policy, fitted only on policy validation."""

from __future__ import annotations

import numpy as np


def policy_metrics(confidences, correct, threshold: float) -> dict:
    confidences, correct = np.asarray(confidences), np.asarray(correct, dtype=bool)
    accepted = confidences >= threshold
    count = int(accepted.sum())
    return {
        "coverage": float(accepted.mean()) if len(accepted) else 0.0,
        "selective_accuracy": float(correct[accepted].mean()) if count else None,
        "accepted": count,
        "reviewed": int(len(correct) - count),
        "total": int(len(correct)),
    }


def select_threshold(confidences, correct, target_accuracy: float = 0.95, min_accepted: int = 100) -> dict:
    """Maximize validation coverage subject to empirical accuracy and sample count.

    Tied scores must be accepted together. If no threshold meets the target, use
    a value above 1 so all inputs are reviewed. This is not an OOD detector or a
    statistical accuracy guarantee.
    """
    confidences, correct = np.asarray(confidences, dtype=float), np.asarray(correct, dtype=bool)
    if confidences.ndim != 1 or correct.shape != confidences.shape or not len(confidences):
        raise ValueError("Expected non-empty, equally sized one-dimensional inputs")
    if not np.isfinite(confidences).all() or np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("Confidences must be finite probabilities")
    if not 0 < target_accuracy <= 1 or min_accepted < 1:
        raise ValueError("Invalid policy target or minimum sample count")
    threshold = 1.000001
    for candidate in np.unique(confidences):
        accepted = confidences >= candidate
        if accepted.sum() >= min_accepted and correct[accepted].mean() >= target_accuracy:
            threshold = float(candidate)
            break
    return {
        "threshold": threshold,
        "target_accuracy": target_accuracy,
        "min_accepted": min_accepted,
        "target_met": bool(threshold <= 1),
        "validation": policy_metrics(confidences, correct, threshold),
        "interpretation": "Maximum coverage meeting empirical policy-validation accuracy; no production guarantee.",
    }
