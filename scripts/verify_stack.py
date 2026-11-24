"""Exercise a running SignalDesk API and dashboard using only the standard library.

This submits public demo tickets and one synthetic out-of-domain message. It
checks the service contract and metadata persistence, and records optional JSON
evidence. Run it against an isolated demo stack, not production traffic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def verify_stack(api_url: str, dashboard_url: str) -> dict:
    api_url, dashboard_url = api_url.rstrip("/"), dashboard_url.rstrip("/")
    checks = []
    started = perf_counter()

    def call(method, endpoint, payload=None, expected=200):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(api_url + endpoint, data=data, method=method,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:
                status, body = response.status, response.read()
        except HTTPError as error:
            status, body = error.code, error.read()
        assert status == expected, f"{method} {endpoint}: expected {expected}, received {status}"
        return json.loads(body)

    health = call("GET", "/health")
    assert health["model_loaded"] is True
    assert call("GET", "/readyz")["status"] == "ready"
    model = call("GET", "/model")
    metrics = call("GET", "/metrics")
    assert len(model["classes"]) == metrics["dataset"]["n_classes"] == 77
    assert model["model_version"] == metrics["model_version"] == health["model_version"]
    assert metrics["test"]["n_samples"] == 3080
    checks.append("readiness, model identity, and evaluation report")
    baseline = call("GET", "/monitoring")

    for payload in ({"text": " "}, {"text": "a" * 4001}, {"text": "valid", "unexpected": True}):
        call("POST", "/predict", payload, expected=422)
    for texts in ([], ["valid"] * 101, ["valid", " "]):
        call("POST", "/predict/batch", {"texts": texts}, expected=422)
    assert call("GET", "/monitoring")["total_predictions"] == baseline["total_predictions"]
    checks.append("invalid single and batch requests rejected without storing events")

    text = "The ATM swallowed my card."
    first = call("POST", "/predict", {"text": text})
    assert first["intent"] == "card_swallowed" and first["decision"] == "route"
    assert first["confidence"] >= model["threshold"]
    assert len(first["top_intents"]) == 3 and first["similar_tickets"] and first["explanations"]
    assert first["top_intents"][0]["intent"] == first["intent"]
    unknown = call("POST", "/predict", {"text": "I ordered pizza and it arrived cold."})
    assert unknown["decision"] == "review"
    checks.append("trained single-ticket routing, evidence, and a fixed review example")

    samples_path = Path(__file__).resolve().parents[1] / "examples/tickets.csv"
    with samples_path.open(newline="") as stream:
        texts = [text] + [row["text"] for row in csv.DictReader(stream)]
    batch = call("POST", "/predict/batch", {"texts": texts})
    assert batch["count"] == len(texts) == len(batch["predictions"])
    assert len({row["prediction_id"] for row in batch["predictions"]}) == len(texts)
    for field in ("intent", "confidence", "decision", "top_intents", "similar_tickets", "explanations"):
        assert batch["predictions"][0][field] == first[field], field
    assert all(row["model_version"] == model["model_version"] for row in batch["predictions"])
    checks.append("sample CSV batch, unique IDs, and single/batch prediction agreement")

    feedback = {"prediction_id": first["prediction_id"], "correct_intent": "card_swallowed"}
    assert call("POST", "/feedback", feedback)["status"] == "recorded"
    assert call("POST", "/feedback", feedback)["status"] == "recorded"
    call("POST", "/feedback", {**feedback, "correct_intent": "unknown_category"}, expected=422)
    call("POST", "/feedback", {**feedback, "prediction_id": "missing-prediction"}, expected=404)
    monitoring = call("GET", "/monitoring")
    expected_total = baseline["total_predictions"] + len(texts) + 2
    assert monitoring["total_predictions"] == expected_total
    assert monitoring["feedback_count"] == baseline["feedback_count"] + 1
    assert monitoring["text_stored"] is False
    assert all("text" not in row for row in monitoring["recent_predictions"])
    assert 0 <= monitoring["confidence_drift_js"] <= 1
    checks.append("feedback correction, exact event counts, and text-free monitoring")

    with urlopen(dashboard_url + "/_stcore/health", timeout=15) as response:
        assert response.status == 200 and response.read() == b"ok"
    with urlopen(dashboard_url, timeout=15) as response:
        assert response.status == 200 and b"streamlit" in response.read().lower()
    checks.append("Streamlit health and application HTTP entrypoint")
    return {
        "status": "passed", "model_version": model["model_version"],
        "checks": checks, "check_count": len(checks),
        "duration_seconds": round(perf_counter() - started, 3),
        "total_predictions_after": expected_total,
        "feedback_count_after": monitoring["feedback_count"],
        "note": "Known-example review is a regression check, not a general OOD detection claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8501")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_stack(args.api_url, args.dashboard_url)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
