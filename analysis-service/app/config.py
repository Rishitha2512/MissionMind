"""
MissionMind Analysis Service — Configuration.
Loads settings from environment / .env file.
"""
from pydantic_settings import BaseSettings
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SERVICE_ROOT.parent


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Paths
    data_dir: Path = _SERVICE_ROOT.parent / "data"
    processed_dir: Path = _SERVICE_ROOT.parent / "data" / "processed"
    docs_dir: Path = _SERVICE_ROOT.parent / "data" / "docs"
    chunks_path: Path = _SERVICE_ROOT.parent / "data" / "processed" / "chunks.json"
    events_path: Path = _SERVICE_ROOT.parent / "data" / "processed" / "events.json"
    meda_data_path: Path = _SERVICE_ROOT.parent / "data" / "processed" / "meda_clean.parquet"

    # Anomaly detection
    zscore_threshold: float = 2.5
    time_bin_hours: int = 1          # 1-hour LTST bins
    season_bins: int = 4             # 4 Ls quadrants

    # Similarity
    similarity_top_k: int = 5

    # RAG
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 10
    hybrid_rrf_k: int = 60           # RRF constant
    min_evidence_score: float = 0.25  # below this → insufficient_evidence

    class Config:
        env_file = str(_PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
