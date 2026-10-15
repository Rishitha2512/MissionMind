"""
MissionMind — Cosine Similarity for Historical Event Matching.

Featurize events as [magnitude, local_time_hour, duration_hours] (normalized),
then compute cosine similarity to find the closest historical matches.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EventRecord:
    event_id: str
    summary: str
    sol: int
    ltst: float                # local time (0-24)
    ls: float                  # solar longitude (0-360)
    magnitude: float           # e.g., z-score or absolute deviation
    duration_hours: float      # duration of event
    variable: str              # which variable (t_air, pressure)
    features: Optional[np.ndarray] = None


@dataclass
class SimilarityMatch:
    event_id: str
    similarity: float
    summary: str


class EventSimilarityEngine:
    """Finds historically similar events using cosine similarity."""

    FEATURE_NAMES = ["magnitude", "ltst", "duration_hours"]

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.events: List[EventRecord] = []
        self.scaler = StandardScaler()
        self._feature_matrix: Optional[np.ndarray] = None
        self._fitted = False

    def load_events(self, path: Path) -> None:
        """Load event records from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.events = []
        for e in data:
            self.events.append(EventRecord(
                event_id=e["event_id"],
                summary=e["summary"],
                sol=e["sol"],
                ltst=e["ltst"],
                ls=e["ls"],
                magnitude=e["magnitude"],
                duration_hours=e["duration_hours"],
                variable=e.get("variable", "t_air"),
            ))

        self._fit_features()

    def _fit_features(self) -> None:
        """Extract and normalize feature vectors."""
        if not self.events:
            return

        raw = np.array([
            [e.magnitude, e.ltst, e.duration_hours]
            for e in self.events
        ])

        self._feature_matrix = self.scaler.fit_transform(raw)
        for i, event in enumerate(self.events):
            event.features = self._feature_matrix[i]

        self._fitted = True

    def find_similar(
        self,
        magnitude: float,
        ltst: float,
        duration_hours: float,
        exclude_event_id: Optional[str] = None,
    ) -> List[SimilarityMatch]:
        """
        Find the top-k most similar historical events.

        Args:
            magnitude: anomaly magnitude (e.g., z-score)
            ltst: local time of observation
            duration_hours: duration of the anomalous event
            exclude_event_id: optional event to exclude (avoid self-match)

        Returns:
            Sorted list of SimilarityMatch (highest similarity first)
        """
        if not self._fitted or self._feature_matrix is None:
            return []

        # Build query vector and transform using fitted scaler
        query = np.array([[magnitude, ltst, duration_hours]])
        query_scaled = self.scaler.transform(query)

        # Cosine similarity against all events
        sims = cosine_similarity(query_scaled, self._feature_matrix)[0]

        # Build results
        results: List[SimilarityMatch] = []
        for i, event in enumerate(self.events):
            if exclude_event_id and event.event_id == exclude_event_id:
                continue
            results.append(SimilarityMatch(
                event_id=event.event_id,
                similarity=round(float(sims[i]), 4),
                summary=event.summary,
            ))

        # Sort descending by similarity
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:self.top_k]

    def find_similar_from_description(
        self,
        anomaly_score: float,
        ltst: float = 12.0,
        duration_hours: float = 1.0,
    ) -> List[SimilarityMatch]:
        """
        Convenience method when only a text description is available.
        Uses anomaly score as magnitude proxy.
        """
        return self.find_similar(
            magnitude=anomaly_score,
            ltst=ltst,
            duration_hours=duration_hours,
        )
