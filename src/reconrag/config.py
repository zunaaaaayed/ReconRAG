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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
