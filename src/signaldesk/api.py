"""FastAPI serving layer; training and artifact verification stay outside requests."""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from signaldesk.modeling import load_bundle, predict
from signaldesk.monitoring import EventStore
from signaldesk.schemas import (
    BatchPrediction,
    BatchRequest,
    FeedbackRequest,
    Prediction,
    PredictRequest,
)

logger = logging.getLogger(__name__)


def create_app(artifact_dir: str | Path | None = None, db_path: str | Path | None = None) -> FastAPI:
    artifact_path = Path(artifact_dir or os.getenv("SIGNALDESK_ARTIFACT_DIR", "artifacts"))
    database = Path(db_path or os.getenv("SIGNALDESK_DB_PATH", "var/events.sqlite3"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = None
        app.state.metrics = {}
        app.state.store = EventStore(database)
        try:
            candidate_bundle = load_bundle(artifact_path)
            metrics_path = artifact_path / "metrics.json"
            candidate_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else (
                candidate_bundle.get("metrics", {})
            )
            if not isinstance(candidate_metrics, dict):
                raise ValueError("Artifact metrics must be a JSON object")
            app.state.metrics = candidate_metrics
            app.state.bundle = candidate_bundle
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logger.error("Model unavailable: %s", exc)
        yield

    app = FastAPI(
        title="SignalDesk API", version="1.0.0", lifespan=lifespan,
        description="Calibrated support intent predictions, a human-review policy, and local monitoring.",
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        max_bytes = 2_000_000
        try:
            if int(request.headers.get("content-length", 0)) > max_bytes:
                return JSONResponse({"detail": "Request body exceeds 2 MB"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        if request.method in {"POST", "PUT", "PATCH"}:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > max_bytes:
                    return JSONResponse({"detail": "Request body exceeds 2 MB"}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def bundle():
        value = app.state.bundle
        if value is None:
            raise HTTPException(503, "Model unavailable. Run python -m signaldesk.train and restart the API.")
        return value

    def infer(texts: list[str]) -> list[dict]:
        model = bundle()
        started = perf_counter()
        results = predict(model, texts)
        elapsed = round((perf_counter() - started) * 1000 / len(texts), 3)
        version = model["metadata"]["model_version"]
        enriched = [
            {**row, "prediction_id": str(uuid4()), "model_version": version, "latency_ms": elapsed}
            for row in results
        ]
        app.state.store.record(enriched)
        return enriched

    @app.get("/health")
    def health():
        model = app.state.bundle
        return {"status": "ok" if model is not None else "degraded", "model_loaded": model is not None,
                "model_version": model["metadata"]["model_version"] if model is not None else None}

    @app.get("/readyz")
    def ready():
        bundle()
        return {"status": "ready"}

    @app.get("/model")
    def model_info():
        model = bundle()
        return {**model["metadata"], "dataset": "BANKING77", "classes": list(model["classes"]),
                "threshold": model["threshold"], "metrics": app.state.metrics}

    @app.get("/metrics")
    def metrics():
        bundle()
        return app.state.metrics

    @app.post("/predict", response_model=Prediction)
    def predict_one(payload: PredictRequest):
        return infer([payload.text])[0]

    @app.post("/predict/batch", response_model=BatchPrediction)
    def predict_batch(payload: BatchRequest):
        predictions = infer(payload.texts)
        return {"model_version": bundle()["metadata"]["model_version"],
                "count": len(predictions), "predictions": predictions}

    @app.get("/monitoring")
    def monitoring():
        model = bundle()
        hist = model.get("reference", {}).get("confidence_histogram", [0] * 10)
        reference = {"bins": np.linspace(0, 1, len(hist) + 1).tolist(), "counts": hist}
        return app.state.store.summary(model["metadata"]["model_version"], reference)

    @app.post("/feedback")
    def feedback(payload: FeedbackRequest):
        if payload.correct_intent not in bundle()["classes"]:
            raise HTTPException(422, "correct_intent must be one of the model's 77 intent labels")
        if not app.state.store.feedback(payload.prediction_id, payload.correct_intent):
            raise HTTPException(404, "Prediction ID not found")
        return {"status": "recorded"}

    return app


app = create_app()
