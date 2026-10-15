"""
MissionMind Analysis Service — FastAPI Application.

POST /analyze — the core endpoint.
GET  /health  — service health check.

All numerical scores are computed by code.
The LLM only explains structured results and must cite chunk IDs.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    AnalyzeRequest, AnalyzeResponse, HealthResponse,
    AnomalyResult, SimilarEvent, EvidenceChunk,
)
from app.data.loader import MEDADataLoader
from app.anomaly.zscore import SeasonalZScoreDetector
from app.similarity.cosine import EventSimilarityEngine
from app.rag.chunker import load_chunks
from app.rag.retriever import HybridRetriever
from app.rag.generator import RAGGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("missionmind.analysis")

# ── Global state (loaded at startup) ─────────────────────────────────────────
data_loader = MEDADataLoader(settings.meda_data_path)
detector = SeasonalZScoreDetector(
    time_bin_hours=settings.time_bin_hours,
    threshold=settings.zscore_threshold,
)
similarity_engine = EventSimilarityEngine(top_k=settings.similarity_top_k)
retriever = HybridRetriever(
    embedding_model_name=settings.embedding_model_name,
    rrf_k=settings.hybrid_rrf_k,
    top_k=settings.retrieval_top_k,
)
generator = RAGGenerator(
    api_key=settings.openai_api_key,
    model=settings.llm_model,
    min_evidence_score=settings.min_evidence_score,
)

_service_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data, fit models, and build indices at startup."""
    global _service_ready

    logger.info("Loading MEDA data...")
    try:
        df = data_loader.load()
        logger.info(f"  Loaded {len(df)} rows, sols {df['sol'].min()}-{df['sol'].max()}")
    except FileNotFoundError as e:
        logger.warning(f"  {e}")
        logger.warning("  Run: python scripts/generate_meda_data.py")
        yield
        return

    logger.info("Fitting seasonal z-score detector...")
    detector.fit(df)
    baseline_path = settings.processed_dir / "baselines.json"
    detector.export_baselines(baseline_path)
    logger.info(f"  Baselines exported to {baseline_path}")

    logger.info("Loading historical events...")
    events_path = settings.events_path
    if events_path.exists():
        similarity_engine.load_events(events_path)
        logger.info(f"  Loaded {len(similarity_engine.events)} events")
    else:
        logger.warning(f"  Events file not found: {events_path}")
        logger.warning("  Run: python scripts/seed_events.py")

    logger.info("Loading document chunks and building retrieval index...")
    chunks_path = settings.chunks_path
    if chunks_path.exists():
        chunks = load_chunks(chunks_path)
        retriever.index(chunks)
        logger.info(f"  Indexed {len(chunks)} chunks")
    else:
        logger.warning(f"  Chunks file not found: {chunks_path}")
        logger.warning("  Run: python scripts/ingest_docs.py")

    _service_ready = True
    logger.info("✓ Analysis service ready.")
    yield
    logger.info("Shutting down analysis service.")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MissionMind Analysis Service",
    description="Anomaly detection, similarity search, and RAG for Mars 2020 MEDA data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if _service_ready else "initializing",
        version="1.0.0",
        data_loaded=data_loader.is_loaded,
        docs_loaded=len(retriever.chunks),
        events_loaded=len(similarity_engine.events),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Core analysis endpoint.

    Pipeline:
      1. Anomaly detection (z-score)
      2. Similar event search (cosine similarity)
      3. Evidence retrieval (hybrid BM25+vector)
      4. LLM explanation (with citation enforcement)
    """
    if not _service_ready:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Data may not be loaded."
        )

    # ── Step 1: Anomaly Detection ────────────────────────────────────────
    if req.t_air is not None or req.pressure is not None:
        ltst = req.ltst if req.ltst is not None else 12.0
        ls = req.ls if req.ls is not None else (data_loader.estimate_ls(req.sol) if req.sol is not None else 30.0)
        vals = {}
        if req.t_air is not None:
            vals["t_air"] = req.t_air
        if req.pressure is not None:
            vals["pressure"] = req.pressure
        anomaly_report = detector.score_observation(ltst=ltst, ls=ls, values=vals)
    elif req.series and len(req.series) > 0:
        # Use provided time-series data
        series_data = [{"t": p.t, "value": p.value} for p in req.series]
        # Estimate Ls from first point or use mid-range
        ls_estimate = _estimate_ls_from_description(req.description)
        anomaly_report = detector.score_series(series_data, ls=ls_estimate)
        ltst = _parse_description(req.description)["ltst"]
    else:
        # Parse what we can from the description
        parsed = _parse_description(req.description)
        ltst = parsed["ltst"]
        anomaly_report = detector.score_observation(
            ltst=parsed["ltst"],
            ls=parsed["ls"],
            values=parsed["values"],
        )

    logger.info(
        f"Anomaly: detected={anomaly_report.detected}, "
        f"score={anomaly_report.score}"
    )

    # ── Step 2: Similar Events ───────────────────────────────────────────
    similar_matches = similarity_engine.find_similar_from_description(
        anomaly_score=anomaly_report.score,
        ltst=ltst,
        duration_hours=1.0,
    )

    similar_events = [
        SimilarEvent(
            event_id=m.event_id,
            similarity=m.similarity,
            summary=m.summary,
        )
        for m in similar_matches
    ]

    # ── Step 3: Evidence Retrieval ───────────────────────────────────────
    query = _build_retrieval_query(req.description, anomaly_report.reasons)
    retrieved = retriever.retrieve(query, method="hybrid")

    evidence = [
        EvidenceChunk(
            chunk_id=r.chunk_id,
            doc_title=r.doc_title,
            source_url=r.source_url,
            snippet=r.snippet,
            score=r.score,
        )
        for r in retrieved
    ]

    # ── Step 4: LLM Explanation ──────────────────────────────────────────
    explanation, status = generator.generate(
        description=req.description,
        anomaly_detected=anomaly_report.detected,
        anomaly_score=anomaly_report.score,
        anomaly_reasons=anomaly_report.reasons,
        similar_events=[
            {"event_id": e.event_id, "similarity": e.similarity, "summary": e.summary}
            for e in similar_events
        ],
        evidence_chunks=[
            {"chunk_id": e.chunk_id, "doc_title": e.doc_title,
             "source_url": e.source_url, "snippet": e.snippet, "score": e.score}
            for e in evidence
        ],
    )

    return AnalyzeResponse(
        anomaly=AnomalyResult(
            detected=anomaly_report.detected,
            score=anomaly_report.score,
            method=anomaly_report.method,
            reasons=anomaly_report.reasons,
            variable_scores=anomaly_report.variable_scores,
            baselines=anomaly_report.baselines,
            season_name=anomaly_report.season_name,
            time_of_sol=anomaly_report.time_of_sol,
            severity=anomaly_report.severity,
        ),
        similar_events=similar_events,
        evidence=evidence,
        explanation=explanation,
        status=status,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_description(description: str) -> dict:
    """
    Extract observation parameters from free-text description.
    Returns defaults when values can't be parsed.
    """
    import re

    result = {
        "ltst": 12.0,
        "ls": 30.0,
        "values": {},
    }

    # Try to parse temperature
    temp_match = re.search(
        r'(?:temperature|temp|air temp)[:\s]*([+-]?\d+\.?\d*)\s*°?C?',
        description, re.IGNORECASE,
    )
    if temp_match:
        result["values"]["t_air"] = float(temp_match.group(1))

    # Try to parse pressure
    press_match = re.search(
        r'(?:pressure|press)[:\s]*(\d+\.?\d*)\s*(?:Pa)?',
        description, re.IGNORECASE,
    )
    if press_match:
        result["values"]["pressure"] = float(press_match.group(1))

    # Try to parse LTST
    ltst_match = re.search(
        r'(?:LTST|local time|time)[:\s]*(\d+\.?\d*)',
        description, re.IGNORECASE,
    )
    if ltst_match:
        result["values"].setdefault("t_air", -30.0)  # default temp
        result["ltst"] = float(ltst_match.group(1))

    # Try to parse sol
    sol_match = re.search(r'sol[:\s]*(\d+)', description, re.IGNORECASE)
    if sol_match:
        sol = int(sol_match.group(1))
        result["ls"] = data_loader.estimate_ls(sol)

    # If no values parsed, use defaults for a "generic" query
    if not result["values"]:
        result["values"] = {"t_air": -30.0, "pressure": 750.0}

    return result


def _estimate_ls_from_description(description: str) -> float:
    """Estimate Ls from sol number in description, or return default."""
    import re
    sol_match = re.search(r'sol[:\s]*(\d+)', description, re.IGNORECASE)
    if sol_match:
        return data_loader.estimate_ls(int(sol_match.group(1)))
    return 30.0  # default: early northern spring


def _build_retrieval_query(description: str, reasons: list[str]) -> str:
    """Build a retrieval query combining user description and anomaly reasons."""
    parts = [description]
    for r in reasons[:2]:  # max 2 reasons to avoid query bloat
        parts.append(r)
    return " ".join(parts)
