"""Tests for local semantic retrieval."""

from datetime import UTC, datetime

import numpy as np

from reconrag.models import Chunk, Document
from reconrag.retrieval.vector_index import (
    InMemoryVectorIndex,
)

VOCABULARY = (
    "gaussian",
    "uncertainty",
    "tomography",
    "segmentation",
)


class FakeEmbedder:
    """Deterministic test embedder requiring no model download."""

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        vectors = [
            [float(text.casefold().count(term)) for term in VOCABULARY]
            for text in texts
        ]

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


def _document(
    document_id: str,
    title: str,
) -> Document:
    return Document(
        id=document_id,
        title=title,
        filename=f"{document_id}.pdf",
        checksum=document_id,
        ingested_at=datetime.now(UTC),
    )


def _chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    section: str,
    index: int,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=index,
        page_start=index + 1,
        page_end=index + 1,
        section_heading=section,
        token_count=5,
    )


def test_search_ranks_semantically_matching_chunk_first() -> None:
    document = _document(
        "paper-one",
        "Gaussian Reconstruction",
    )

    chunks = [
        _chunk(
            "chunk-one",
            document.id,
            "Gaussian tomography reconstruction.",
            "Method",
            0,
        ),
        _chunk(
            "chunk-two",
            document.id,
            "Uncertainty estimation and calibration.",
            "Uncertainty",
            1,
        ),
    ]

    index = InMemoryVectorIndex(FakeEmbedder())
    index.build([(document, chunks)])

    results = index.search(
        "How is uncertainty estimated?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.id == "chunk-two"
    assert results[0].score > results[1].score


def test_search_filters_by_section() -> None:
    document = _document(
        "paper-one",
        "Gaussian Reconstruction",
    )

    chunks = [
        _chunk(
            "chunk-one",
            document.id,
            "Gaussian tomography method.",
            "Method",
            0,
        ),
        _chunk(
            "chunk-two",
            document.id,
            "Gaussian uncertainty analysis.",
            "Uncertainty",
            1,
        ),
    ]

    index = InMemoryVectorIndex(FakeEmbedder())
    index.build([(document, chunks)])

    results = index.search(
        "Gaussian",
        section_heading="Uncertainty",
    )

    assert len(results) == 1
    assert results[0].chunk.id == "chunk-two"


def test_empty_index_returns_no_results() -> None:
    index = InMemoryVectorIndex(FakeEmbedder())

    assert index.search("Gaussian tomography") == []


def test_snapshot_restores_searchable_index() -> None:
    document = _document(
        "paper-one",
        "Gaussian Reconstruction",
    )

    chunks = [
        _chunk(
            "chunk-one",
            document.id,
            "Gaussian tomography reconstruction.",
            "Method",
            0,
        ),
        _chunk(
            "chunk-two",
            document.id,
            "Uncertainty estimation and calibration.",
            "Uncertainty",
            1,
        ),
    ]

    original_index = InMemoryVectorIndex(FakeEmbedder())
    original_index.build([(document, chunks)])

    snapshot = original_index.snapshot()

    restored_index = InMemoryVectorIndex(FakeEmbedder())
    restored_index.load(snapshot)

    results = restored_index.search(
        "uncertainty",
        top_k=1,
    )

    assert restored_index.size == 2
    assert len(snapshot) == 2
    assert results[0].chunk.id == "chunk-two"
