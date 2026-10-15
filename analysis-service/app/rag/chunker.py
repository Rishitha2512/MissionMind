"""
MissionMind — Document Chunker.

Splits documents into fixed-size chunks with overlap for RAG retrieval.
Each chunk gets a unique chunk_id for citation tracking.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source_url: str
    chunk_index: int
    text: str
    token_count: int  # approximate


def _approx_token_count(text: str) -> int:
    """Rough token count: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences at period/newline boundaries."""
    parts = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    doc_id: str,
    doc_title: str,
    source_url: str,
    text: str,
    chunk_size_tokens: int = 300,
    chunk_overlap_tokens: int = 50,
) -> List[DocumentChunk]:
    """
    Split a document into overlapping chunks.

    Uses sentence-boundary splitting to avoid breaking mid-sentence.
    """
    sentences = _split_into_sentences(text)
    chunks: List[DocumentChunk] = []

    current_sentences: List[str] = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sent_tokens = _approx_token_count(sentence)

        if current_tokens + sent_tokens > chunk_size_tokens and current_sentences:
            # Emit chunk
            chunk_text = " ".join(current_sentences)
            chunk_id_raw = f"{doc_id}_chunk_{chunk_index}"
            chunk_id = hashlib.md5(chunk_id_raw.encode()).hexdigest()[:12]

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                source_url=source_url,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=_approx_token_count(chunk_text),
            ))
            chunk_index += 1

            # Overlap: keep last few sentences
            overlap_tokens = 0
            overlap_start = len(current_sentences)
            for i in range(len(current_sentences) - 1, -1, -1):
                st = _approx_token_count(current_sentences[i])
                if overlap_tokens + st > chunk_overlap_tokens:
                    break
                overlap_tokens += st
                overlap_start = i

            current_sentences = current_sentences[overlap_start:]
            current_tokens = sum(_approx_token_count(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_id_raw = f"{doc_id}_chunk_{chunk_index}"
        chunk_id = hashlib.md5(chunk_id_raw.encode()).hexdigest()[:12]
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            doc_title=doc_title,
            source_url=source_url,
            chunk_index=chunk_index,
            text=chunk_text,
            token_count=_approx_token_count(chunk_text),
        ))

    return chunks


def load_chunks(path: Path) -> List[DocumentChunk]:
    """Load pre-computed chunks from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [DocumentChunk(**c) for c in data]


def save_chunks(chunks: List[DocumentChunk], path: Path) -> None:
    """Save chunks to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
