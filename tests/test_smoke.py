"""Foundation-level smoke tests."""

from reconrag import __version__
from reconrag.config import Settings


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_default_settings() -> None:
    settings = Settings()

    assert settings.app_name == "ReconRAG"
    assert settings.retrieval_top_k == 5
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
