"""
core/config.py — Centralized configuration for DocSage
All settings via environment variables with sensible defaults.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "DocSage"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── API ───────────────────────────────────────────────────────────────────
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]
    max_upload_size_mb: int = 50
    request_timeout_seconds: int = 120

    # ── Auth (JWT) ────────────────────────────────────────────────────────────
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_32_CHARS_MIN"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    documents_dir: Path = Path("./data/documents")
    index_dir: Path = Path("./data/indices")
    model_cache_dir: Path = Path("./data/model_cache")
    database_url: str = "sqlite+aiosqlite:///./data/docsage.db"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 60 * 60 * 4  # 4 hours

    # ── Models ────────────────────────────────────────────────────────────────
    # Reader model (extractive QA)
    reader_model_name: str = "deepset/roberta-base-squad2"
    reader_model_device: str = "cpu"  # "cuda" for GPU

    # Embedder (for dense retrieval)
    embedder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_batch_size: int = 64
    embedding_dim: int = 384

    # Reranker (cross-encoder)
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Robustness model (ELECTRA)
    robustness_model_name: str = "google/electra-base-discriminator"
    robustness_threshold: float = 0.7  # confidence below which we flag uncertainty

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size_tokens: int = 400        # within BERT limit with overlap buffer
    chunk_overlap_tokens: int = 80      # sliding window overlap
    max_chunks_per_doc: int = 2000
    long_doc_threshold_tokens: int = 512  # trigger hierarchical mode above this

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 20           # candidates before reranking
    rerank_top_k: int = 5               # passages sent to reader
    dense_weight: float = 0.6           # hybrid fusion weight for dense score
    sparse_weight: float = 0.3          # weight for BM25
    kg_weight: float = 0.1              # weight for KG-based score
    faiss_nprobe: int = 16              # FAISS IVF probe count

    # ── Conversational ────────────────────────────────────────────────────────
    max_history_turns: int = 10
    history_summary_threshold: int = 6  # summarize older turns beyond this

    # ── Adversarial Robustness ────────────────────────────────────────────────
    enable_adversarial_filter: bool = True
    adversarial_score_penalty: float = 0.15  # reduce confidence for flagged answers

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def create_dirs(self) -> None:
        """Ensure all required directories exist."""
        for d in [self.data_dir, self.documents_dir, self.index_dir, self.model_cache_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()