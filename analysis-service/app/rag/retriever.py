"""
MissionMind — Hybrid Retriever (BM25 + Vector + RRF).

Retrieval pipeline:
  1. BM25 (keyword) via rank_bm25
  2. Dense vector via sentence-transformers
  3. Hybrid = Reciprocal Rank Fusion of both lists
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from app.rag.chunker import DocumentChunk


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_title: str
    source_url: str
    snippet: str
    score: float
    method: str  # "bm25", "vector", "hybrid"


class HybridRetriever:
    """BM25 + Dense Vector hybrid retriever with RRF merging."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        top_k: int = 10,
    ):
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.embedder = None
        self.chunk_embeddings: Optional[np.ndarray] = None
        self._model_name = embedding_model_name
        self._ready = False

    def index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 and vector indices from document chunks."""
        self.chunks = chunks
        if not chunks:
            return

        # BM25 index
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        # Vector index
        if SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer(self._model_name)
                texts = [c.text for c in chunks]
                self.chunk_embeddings = self.embedder.encode(
                    texts, show_progress_bar=False, normalize_embeddings=True
                )
            except Exception:
                self.embedder = None

        self._ready = True

    def retrieve_bm25(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Keyword retrieval using BM25."""
        if not self._ready or self.bm25 is None:
            return []
        k = top_k or self.top_k
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            c = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=c.chunk_id,
                doc_title=c.doc_title,
                source_url=c.source_url,
                snippet=c.text[:300],
                score=round(float(scores[idx]), 4),
                method="bm25",
            ))
        return results

    def retrieve_vector(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Dense vector retrieval using sentence-transformers."""
        if not self._ready or self.embedder is None or self.chunk_embeddings is None:
            return []
        k = top_k or self.top_k

        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        )
        sims = (self.chunk_embeddings @ q_emb.T).squeeze()
        top_indices = np.argsort(sims)[::-1][:k]

        results = []
        for idx in top_indices:
            if sims[idx] <= 0:
                continue
            c = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=c.chunk_id,
                doc_title=c.doc_title,
                source_url=c.source_url,
                snippet=c.text[:300],
                score=round(float(sims[idx]), 4),
                method="vector",
            ))
        return results

    def retrieve_hybrid(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Hybrid retrieval: Reciprocal Rank Fusion (RRF) of BM25 + vector.

        RRF score = Σ 1 / (k + rank_i) for each retriever.
        """
        k = top_k or self.top_k
        bm25_results = self.retrieve_bm25(query, top_k=k * 2)
        vector_results = self.retrieve_vector(query, top_k=k * 2)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}

        for rank, r in enumerate(bm25_results):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + \
                1.0 / (self.rrf_k + rank + 1)
            chunk_map[r.chunk_id] = r

        for rank, r in enumerate(vector_results):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + \
                1.0 / (self.rrf_k + rank + 1)
            if r.chunk_id not in chunk_map:
                chunk_map[r.chunk_id] = r

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for cid in sorted_ids[:k]:
            r = chunk_map[cid]
            results.append(RetrievalResult(
                chunk_id=r.chunk_id,
                doc_title=r.doc_title,
                source_url=r.source_url,
                snippet=r.snippet,
                score=round(rrf_scores[cid], 4),
                method="hybrid",
            ))
        return results

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Unified retrieve method."""
        if method == "bm25":
            return self.retrieve_bm25(query, top_k)
        elif method == "vector":
            return self.retrieve_vector(query, top_k)
        else:
            return self.retrieve_hybrid(query, top_k)
