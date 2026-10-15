"""
MissionMind — RAG Generator (LLM explanation with citations).

Rules enforced:
  1. The LLM receives ONLY the structured analysis results + retrieved chunks.
  2. The LLM must cite chunk_ids in its explanation.
  3. If evidence is weak → status = "insufficient_evidence".
  4. The LLM never assigns scores — only explains them.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from openai import OpenAI

from app.rag.retriever import RetrievalResult


SYSTEM_PROMPT = """You are MissionMind, a scientific assistant for Mars 2020 / Perseverance MEDA data analysis. Your role is to EXPLAIN the structured analysis results provided to you.

STRICT RULES:
1. You must ONLY explain the anomaly scores and similar events that have already been computed. NEVER assign your own scores or change them.
2. You MUST cite evidence using the chunk IDs provided in square brackets, e.g., [chunk_abc123].
3. Every factual claim you make must be supported by at least one cited chunk.
4. If the retrieved evidence is insufficient or not relevant to the observation, say so explicitly.
5. Use precise scientific language appropriate for a planetary scientist audience.
6. Structure your response as:
   - Brief anomaly interpretation (referencing the computed scores)
   - Relevant context from the evidence chunks (with citations)
   - Comparison to similar historical events if available
   - Any caveats or limitations

CHUNK FORMAT:
Each evidence chunk has:
  - chunk_id: unique identifier to cite as [chunk_id]
  - doc_title: source document name
  - snippet: the actual text content

When citing, always use the format [chunk_id] inline within your sentences."""


def build_user_prompt(
    description: str,
    anomaly_detected: bool,
    anomaly_score: float,
    anomaly_reasons: List[str],
    similar_events: List[dict],
    evidence_chunks: List[dict],
) -> str:
    """Build the user prompt with structured results and evidence."""

    prompt_parts = [
        "## Observation",
        f"Description: {description}",
        "",
        "## Computed Anomaly Analysis (DO NOT modify these values)",
        f"- Anomaly detected: {anomaly_detected}",
        f"- Anomaly score (z-score): {anomaly_score}",
        f"- Method: seasonal_zscore",
        f"- Reasons:",
    ]
    for r in anomaly_reasons:
        prompt_parts.append(f"  • {r}")

    if similar_events:
        prompt_parts.extend(["", "## Similar Historical Events"])
        for evt in similar_events:
            prompt_parts.append(
                f"- {evt['event_id']}: {evt['summary']} "
                f"(similarity: {evt['similarity']:.2%})"
            )

    if evidence_chunks:
        prompt_parts.extend(["", "## Retrieved Evidence Chunks (cite these by chunk_id)"])
        for chunk in evidence_chunks:
            prompt_parts.append(
                f"\n[{chunk['chunk_id']}] from \"{chunk['doc_title']}\":\n"
                f"{chunk['snippet']}\n"
            )
    else:
        prompt_parts.extend([
            "",
            "## No evidence chunks retrieved.",
            "State that there is insufficient evidence to provide context.",
        ])

    prompt_parts.extend([
        "",
        "## Task",
        "Provide a scientific explanation of this observation based ONLY on "
        "the computed results and the evidence chunks above. Cite chunk IDs "
        "inline using [chunk_id] format.",
    ])

    return "\n".join(prompt_parts)


class RAGGenerator:
    """Generates LLM explanations with chunk-level citations."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        min_evidence_score: float = 0.25,
    ):
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.min_evidence_score = min_evidence_score

    def generate(
        self,
        description: str,
        anomaly_detected: bool,
        anomaly_score: float,
        anomaly_reasons: List[str],
        similar_events: List[dict],
        evidence_chunks: List[dict],
    ) -> tuple[str, str]:
        """
        Generate an explanation with citations.

        Returns:
            (explanation: str, status: "ok" | "insufficient_evidence")
        """
        # Check evidence quality
        if not evidence_chunks:
            status = "insufficient_evidence"
        else:
            avg_score = sum(c.get("score", 0) for c in evidence_chunks) / len(evidence_chunks)
            status = "ok" if avg_score >= self.min_evidence_score else "insufficient_evidence"

        # If no API key, return a structured fallback
        if not self.client:
            return self._fallback_explanation(
                anomaly_detected, anomaly_score, anomaly_reasons,
                similar_events, evidence_chunks, status
            ), status

        user_prompt = build_user_prompt(
            description=description,
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            anomaly_reasons=anomaly_reasons,
            similar_events=similar_events,
            evidence_chunks=evidence_chunks,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            explanation = response.choices[0].message.content or ""

            # Verify citations exist
            cited_chunks = set(re.findall(r'\[([a-f0-9]{12})\]', explanation))
            available_chunks = {c["chunk_id"] for c in evidence_chunks}

            if evidence_chunks and not cited_chunks.intersection(available_chunks):
                # LLM failed to cite — append warning
                explanation += (
                    "\n\n⚠️ Note: The explanation above could not be fully "
                    "grounded in the retrieved evidence."
                )
                status = "insufficient_evidence"

            return explanation, status

        except Exception:
            # Keep provider details (including credential-related responses) out
            # of the user-facing API. Preserve a useful local explanation when
            # the model service is unavailable or rejects its configured key.
            fallback = self._fallback_explanation(
                anomaly_detected, anomaly_score, anomaly_reasons,
                similar_events, evidence_chunks, "insufficient_evidence"
            )
            return (
                "AI explanation is temporarily unavailable. Showing the local "
                "evidence-based summary instead.\n\n" + fallback,
                "insufficient_evidence"
            )

    def _fallback_explanation(
        self,
        anomaly_detected: bool,
        anomaly_score: float,
        anomaly_reasons: List[str],
        similar_events: List[dict],
        evidence_chunks: List[dict],
        status: str,
    ) -> str:
        """Template-based fallback when no LLM API key is available."""
        parts = []

        if anomaly_detected:
            parts.append(
                f"**Anomaly Detected** — The seasonal-baseline z-score analysis "
                f"computed an anomaly score of {anomaly_score:.2f}, exceeding the "
                f"detection threshold."
            )
        else:
            parts.append(
                f"**No Anomaly Detected** — The observation falls within the "
                f"seasonal baseline norms (score: {anomaly_score:.2f})."
            )

        if anomaly_reasons:
            parts.append("\n**Analysis Details:**")
            for r in anomaly_reasons:
                parts.append(f"• {r}")

        if similar_events:
            parts.append("\n**Similar Historical Events:**")
            for evt in similar_events[:3]:
                parts.append(
                    f"• {evt['event_id']}: {evt['summary']} "
                    f"(similarity: {evt['similarity']:.2%})"
                )

        if evidence_chunks:
            parts.append("\n**Supporting Evidence:**")
            for chunk in evidence_chunks[:3]:
                parts.append(
                    f"• [{chunk['chunk_id']}] ({chunk['doc_title']}): "
                    f"{chunk['snippet'][:150]}..."
                )
        else:
            parts.append(
                "\n⚠️ **Insufficient Evidence**: No relevant documents were "
                "retrieved to provide additional context."
            )

        parts.append(
            "\n*Note: This is a template-based explanation. Configure "
            "OPENAI_API_KEY for LLM-generated scientific explanations.*"
        )

        return "\n".join(parts)
