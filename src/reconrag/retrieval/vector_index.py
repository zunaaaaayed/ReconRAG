"""In-memory cosine-similarity index for paper chunks."""

from dataclasses import dataclass

import numpy as np

from reconrag.models import Chunk, Document, SearchResult
from reconrag.retrieval.embedder import TextEmbedder


@dataclass(frozen=True)
class _IndexedChunk:
    """Metadata associated with one embedding row."""

    chunk: Chunk
    document: Document


class InMemoryVectorIndex:
    """Search normalized chunk embeddings with cosine similarity."""

    def __init__(
        self,
        embedder: TextEmbedder,
    ) -> None:
        self.embedder = embedder
        self._records: list[_IndexedChunk] = []
        self._embeddings: np.ndarray | None = None

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._records)

    def build(
        self,
        entries: list[tuple[Document, list[Chunk]]],
    ) -> None:
        """Replace the index with the supplied documents and chunks."""
        records = [
            _IndexedChunk(
                chunk=chunk,
                document=document,
            )
            for document, chunks in entries
            for chunk in chunks
        ]

        if not records:
            self._records = []
            self._embeddings = None
            return

        texts = [self._embedding_text(record) for record in records]

        embeddings = np.asarray(
            self.embedder.encode(texts),
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError("The embedder must return a two-dimensional array.")

        if embeddings.shape[0] != len(records):
            raise ValueError(
                "The number of embeddings does not match the number of chunks."
            )

        self._records = records
        self._embeddings = self._normalize_rows(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
        section_heading: str | None = None,
    ) -> list[SearchResult]:
        """Return the chunks most similar to a query."""
        clean_query = query.strip()

        if not clean_query:
            return []

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        if not self._records or self._embeddings is None:
            return []

        query_embeddings = np.asarray(
            self.embedder.encode([clean_query]),
            dtype=np.float32,
        )

        if query_embeddings.ndim != 2 or query_embeddings.shape[0] != 1:
            raise ValueError("The embedder must return one vector for the query.")

        query_vector = query_embeddings[0]
        query_norm = float(np.linalg.norm(query_vector))

        if query_norm == 0.0:
            return []

        query_vector = query_vector / query_norm

        if query_vector.shape[0] != self._embeddings.shape[1]:
            raise ValueError("Query and document embedding dimensions do not match.")

        eligible_indices = [
            index
            for index, record in enumerate(self._records)
            if self._matches_filters(
                record=record,
                document_ids=document_ids,
                section_heading=section_heading,
            )
        ]

        if not eligible_indices:
            return []

        scores = self._embeddings @ query_vector

        ranked_indices = sorted(
            eligible_indices,
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                chunk=self._records[index].chunk,
                document=self._records[index].document,
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]

    @staticmethod
    def _embedding_text(
        record: _IndexedChunk,
    ) -> str:
        """Combine text with useful retrieval metadata."""
        parts = [
            f"Paper: {record.document.title}",
        ]

        if record.chunk.section_heading:
            parts.append(f"Section: {record.chunk.section_heading}")

        parts.append(record.chunk.text)

        return "\n".join(parts)

    @staticmethod
    def _matches_filters(
        record: _IndexedChunk,
        document_ids: set[str] | None,
        section_heading: str | None,
    ) -> bool:
        if document_ids is not None and record.document.id not in document_ids:
            return False

        if section_heading is None:
            return True

        record_section = record.chunk.section_heading

        return (
            record_section is not None
            and record_section.casefold() == section_heading.casefold()
        )

    @staticmethod
    def _normalize_rows(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Normalize every embedding for cosine similarity."""
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0.0):
            raise ValueError("The embedder returned a zero-length vector.")

        return embeddings / norms
