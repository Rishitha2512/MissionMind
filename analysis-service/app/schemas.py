"""
MissionMind Analysis Service — Pydantic schemas for request / response.
Matches the shared API contract exactly.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ── Request ──────────────────────────────────────────────────────────────────

class TimeSeriesPoint(BaseModel):
    t: str = Field(..., description="ISO-8601 timestamp or sol+LTST string")
    value: float


class AnalyzeRequest(BaseModel):
    mission: str = Field(default="mars2020", description="Mission identifier")
    instrument: str = Field(default="meda", description="Instrument identifier")
    description: str = Field(..., description="Free-text description of the observation")
    series: Optional[List[TimeSeriesPoint]] = Field(
        default=None,
        description="Optional time-series data (e.g., temp readings)"
    )
    sol: Optional[int] = Field(default=None, description="Martian Sol (day)")
    ltst: Optional[float] = Field(default=None, description="Local True Solar Time (0-24)")
    ls: Optional[float] = Field(default=None, description="Solar longitude (0-360)")
    t_air: Optional[float] = Field(default=None, description="Air temperature in Celsius")
    pressure: Optional[float] = Field(default=None, description="Pressure in Pascals")


# ── Response sub-models ──────────────────────────────────────────────────────

class BaselineMetric(BaseModel):
    mean: float
    std: float
    measured: float
    z_score: float
    delta: float
    unit: str


class AnomalyResult(BaseModel):
    detected: bool
    score: float = Field(..., description="Anomaly score (z-score magnitude)")
    method: str = Field(default="seasonal_zscore")
    reasons: List[str]
    variable_scores: Optional[dict[str, float]] = None
    baselines: Optional[dict[str, BaselineMetric]] = None
    season_name: Optional[str] = None
    time_of_sol: Optional[str] = None
    severity: Optional[str] = None


class SimilarEvent(BaseModel):
    event_id: str
    similarity: float = Field(..., description="Cosine similarity 0-1")
    summary: str


class EvidenceChunk(BaseModel):
    chunk_id: str
    doc_title: str
    source_url: str
    snippet: str
    score: float = Field(..., description="Retrieval relevance score")


class AnalyzeResponse(BaseModel):
    anomaly: AnomalyResult
    similar_events: List[SimilarEvent]
    evidence: List[EvidenceChunk]
    explanation: str = Field(
        ..., description="LLM-generated explanation citing chunk_ids"
    )
    status: Literal["ok", "insufficient_evidence"]


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    data_loaded: bool
    docs_loaded: int
    events_loaded: int
