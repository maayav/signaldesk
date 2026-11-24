"""Validated public API contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

TicketText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=4000)]


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictRequest(StrictRequest):
    text: TicketText


class BatchRequest(StrictRequest):
    texts: list[TicketText] = Field(min_length=1, max_length=100)


class IntentScore(BaseModel):
    intent: str
    confidence: float


class SimilarTicket(BaseModel):
    text: str
    intent: str
    similarity: float


class Explanation(BaseModel):
    feature: str
    contribution: float


class Prediction(BaseModel):
    prediction_id: str
    model_version: str
    intent: str
    confidence: float
    decision: Literal["route", "review"]
    top_intents: list[IntentScore]
    similar_tickets: list[SimilarTicket]
    explanations: list[Explanation]
    latency_ms: float


class BatchPrediction(BaseModel):
    model_version: str
    predictions: list[Prediction]
    count: int


class FeedbackRequest(StrictRequest):
    prediction_id: str = Field(min_length=1, max_length=64)
    correct_intent: str = Field(min_length=1, max_length=100)
