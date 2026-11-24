"""Privacy-conscious local event storage and distribution monitoring.

Confidence distribution drift is a proxy for changed inputs. It cannot establish
accuracy drift without representative labels. No raw input text is persisted.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def jensen_shannon(p, q) -> float:
    """Base-2 Jensen-Shannon divergence, bounded in [0, 1]."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 1 or np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must have matching shapes and nonnegative values")
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    midpoint = (p + q) / 2

    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / midpoint[mask])))

    return float(np.clip((kl(p) + kl(q)) / 2, 0, 1))


class EventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('route', 'review')),
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS prediction_window
                    ON predictions(model_version, created_at DESC);
                CREATE TABLE IF NOT EXISTS feedback (
                    prediction_id TEXT PRIMARY KEY REFERENCES predictions(prediction_id),
                    correct_intent TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
            """)

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    def record(self, predictions: list[dict]):
        # Explicit allowlist: input text, neighbors and explanations never enter SQLite.
        columns = ("prediction_id", "model_version", "intent", "confidence", "decision", "latency_ms")
        with self.connection() as con:
            con.executemany(
                "INSERT INTO predictions (prediction_id,model_version,intent,confidence,decision,latency_ms) "
                "VALUES (?,?,?,?,?,?)",
                [tuple(row[k] for k in columns) for row in predictions],
            )

    def feedback(self, prediction_id: str, correct_intent: str) -> bool:
        with self.connection() as con:
            if not con.execute("SELECT 1 FROM predictions WHERE prediction_id=?", (prediction_id,)).fetchone():
                return False
            con.execute(
                "INSERT INTO feedback (prediction_id, correct_intent) VALUES (?,?) "
                "ON CONFLICT(prediction_id) DO UPDATE SET correct_intent=excluded.correct_intent, "
                "created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (prediction_id, correct_intent),
            )
        return True

    def summary(self, model_version: str, reference: dict | None = None, window: int = 1000) -> dict:
        reference = reference or {}
        bins = np.asarray(reference.get("bins", np.linspace(0, 1, 11)), dtype=float)
        reference_counts = np.asarray(reference.get("counts", np.zeros(len(bins) - 1)), dtype=float)
        with self.connection() as con:
            rows = [dict(r) for r in con.execute(
                "SELECT * FROM predictions WHERE model_version=? ORDER BY rowid DESC LIMIT ?",
                (model_version, window),
            )]
            total = con.execute(
                "SELECT COUNT(*) FROM predictions WHERE model_version=?", (model_version,)
            ).fetchone()[0]
            feedback_row = con.execute(
                "SELECT COUNT(*) AS n, AVG(CASE WHEN p.intent=f.correct_intent THEN 1.0 ELSE 0.0 END) AS accuracy "
                "FROM feedback f JOIN predictions p USING(prediction_id) WHERE p.model_version=?",
                (model_version,),
            ).fetchone()
        confidences = [r["confidence"] for r in rows]
        counts = np.histogram(confidences, bins=bins)[0]
        divergence = jensen_shannon(reference_counts, counts)
        distribution = {}
        for row in rows:
            distribution[row["intent"]] = distribution.get(row["intent"], 0) + 1
        minimum_window = 100
        status = "insufficient_data" if len(rows) < minimum_window else (
            "shift_detected" if divergence >= 0.10 else "stable"
        )
        route_rate = sum(r["decision"] == "route" for r in rows) / len(rows) if rows else 0.0
        return {
            "model_version": model_version,
            "total_predictions": total,
            "window_size": len(rows), "window_limit": window, "minimum_window": minimum_window,
            "route_rate": route_rate, "review_rate": 1 - route_rate if rows else 0.0,
            "mean_confidence": float(np.mean(confidences)) if rows else 0.0,
            "confidence_drift_js": divergence, "drift_status": status,
            "drift_threshold": 0.10,
            "drift_note": "Heuristic confidence-distribution shift versus policy validation; not an accuracy or OOD guarantee.",
            "confidence_histogram": {"bins": bins.tolist(), "reference": reference_counts.tolist(), "current": counts.tolist()},
            "intent_distribution": [{"intent": k, "count": v} for k, v in sorted(distribution.items(), key=lambda x: -x[1])],
            "recent_predictions": rows[:20],
            "feedback_count": feedback_row["n"], "feedback_accuracy": feedback_row["accuracy"],
            "feedback_note": "Accuracy on voluntarily labeled predictions; may be selection-biased.",
            "text_stored": False,
        }
