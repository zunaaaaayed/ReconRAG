"""Tests for dependency-free BM25 retrieval."""

from datetime import UTC, datetime

import pytest

from reconrag.models import Chunk, Document
from reconrag.retrieval import BM25Index


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
    document_id: str,
    chunk_id: str,
    index: int,
    text: str,
    section: str,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=index,
        page_start=index + 1,
        page_end=index + 1,
        section_heading=section,
        token_count=max(
            1,
            len(text.split()),
        ),
    )


def _entries() -> list[tuple[Document, list[Chunk]]]:
    uncertainty_paper = _document(
        "paper-1",
        "Uncertainty in Sparse-View CT",
    )
    gaussian_paper = _document(
        "paper-2",
        "Radiative Gaussian Reconstruction",
    )

    return [
        (
            uncertainty_paper,
            [
                _chunk(
                    "paper-1",
                    "chunk-1",
                    0,
                    (
                        "Temperature calibration improves "
                        "interval coverage but cannot "
                        "change spatial ranking."
                    ),
                    "Calibration",
                ),
                _chunk(
                    "paper-1",
                    "chunk-2",
                    1,
                    ("Projection geometry and ray coverage affect CT reconstruction."),
                    "Methods",
                ),
            ],
        ),
        (
            gaussian_paper,
            [
                _chunk(
                    "paper-2",
                    "chunk-3",
                    0,
                    (
                        "Gaussian splatting represents "
                        "attenuation for efficient "
                        "tomographic reconstruction."
                    ),
                    "Methods",
                )
            ],
        ),
    ]


def test_bm25_ranks_exact_technical_terms() -> None:
    index = BM25Index()
    index.build(_entries())

    results = index.search(
        "interval temperature calibration",
        top_k=3,
    )

    assert index.size == 3
    assert results[0].chunk.id == "chunk-1"
    assert results[0].score > 0


def test_bm25_applies_document_filter() -> None:
    index = BM25Index()
    index.build(_entries())

    results = index.search(
        "reconstruction",
        document_ids={"paper-2"},
    )

    assert [result.chunk.id for result in results] == ["chunk-3"]


def test_bm25_applies_case_insensitive_section_filter() -> None:
    index = BM25Index()
    index.build(_entries())

    results = index.search(
        "calibration",
        section_heading="calibration",
    )

    assert [result.chunk.id for result in results] == ["chunk-1"]


def test_bm25_handles_empty_and_unmatched_queries() -> None:
    index = BM25Index()
    index.build(_entries())

    assert index.search("   ") == []
    assert index.search("nonexistentvocabulary") == []


def test_bm25_validates_parameters_and_rebuilds() -> None:
    with pytest.raises(
        ValueError,
        match="k1 must be positive",
    ):
        BM25Index(k1=0)

    with pytest.raises(
        ValueError,
        match="b must be between",
    ):
        BM25Index(b=1.5)

    index = BM25Index()
    index.build(_entries())
    index.build([])

    assert index.size == 0
    assert index.search("calibration") == []

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        index.search(
            "calibration",
            top_k=0,
        )
