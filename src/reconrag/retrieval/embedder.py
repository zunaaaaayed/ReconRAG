"""Local text-embedding models used by retrieval."""

from typing import Any, Protocol

import numpy as np


class TextEmbedder(Protocol):
    """Interface implemented by text-embedding backends."""

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Encode texts into one vector per input text."""
        ...


class SentenceTransformerEmbedder:
    """Generate normalized embeddings with Sentence Transformers."""

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """Load the model only when embeddings are first requested."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device="cpu",
            )

        return self._model

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Return normalized float32 embeddings."""
        if not texts:
            raise ValueError("At least one text is required for embedding.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )
