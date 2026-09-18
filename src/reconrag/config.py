"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings configurable through `RECONRAG_` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RECONRAG_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_name: str = "ReconRAG"
    data_dir: Path = Field(default=Path("data"))
    papers_dir: Path = Field(default=Path("papers"))
    qdrant_path: Path = Field(default=Path("data/qdrant"))
    database_path: Path = Field(default=Path("data/reconrag.db"))
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    generation_model: str = "gemma4:e2b"
    ollama_host: str = "http://localhost:11434"
    generation_max_tokens: int = Field(
        default=500,
        ge=64,
        le=2048,
    )
    chunk_target_tokens: int = Field(default=350, ge=64, le=510)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=128)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
