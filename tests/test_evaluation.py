"""Tests for retrieval evaluation metrics and orchestration."""

from datetime import UTC, datetime

import pytest

from reconrag.evaluation import (
    RetrievalCase,
    RetrievalEvaluator,
    recall_at_k,
    reciprocal_rank,
)
from reconrag.models import (
    Chunk,
    Document,
    SearchResult,
)


class FakeSearchIndex:
    """Return a fixed ranking for evaluator tests."""

    def __init__(
        self,
        results: list[SearchResult],
    ) -> None:
        self.results = results

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
        section_heading: str | None = None,
    ) -> list[SearchResult]:
        del query
        del document_ids
        del section_heading

        return self.results[:top_k]


def _document() -> Document:
    return Document(
        id="paper-1",
        title="Test Paper",
        filename="test.pdf",
        checksum="checksum-1",
        ingested_at=datetime.now(UTC),
    )


def _result(
    chunk_id: str,
    chunk_index: int,
    score: float,
) -> SearchResult:
    document = _document()

    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            document_id=document.id,
            text=f"Evidence from {chunk_id}.",
            chunk_index=chunk_index,
            page_start=chunk_index + 1,
            page_end=chunk_index + 1,
            section_heading="Methods",
            token_count=4,
        ),
        document=document,
        score=score,
    )


def test_recall_at_k_counts_unique_relevant_chunks() -> None:
    score = recall_at_k(
        retrieved_ids=[
            "chunk-1",
            "chunk-1",
            "chunk-2",
        ],
        relevant_ids={
            "chunk-1",
            "chunk-2",
        },
        k=3,
    )

    assert score == 1.0


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    score = reciprocal_rank(
        retrieved_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        relevant_ids={"chunk-2"},
        k=3,
    )

    assert score == 0.5


def test_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(
        ValueError,
        match="k must be positive",
    ):
        recall_at_k(
            retrieved_ids=[],
            relevant_ids={"chunk-1"},
            k=0,
        )

    with pytest.raises(
        ValueError,
        match="At least one relevant",
    ):
        reciprocal_rank(
            retrieved_ids=[],
            relevant_ids=set(),
            k=5,
        )


def test_retrieval_case_validates_required_fields() -> None:
    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        RetrievalCase(
            id="case-1",
            question=" ",
            relevant_chunk_ids=frozenset({"chunk-1"}),
        )

    with pytest.raises(
        ValueError,
        match="relevant chunk ID",
    ):
        RetrievalCase(
            id="case-1",
            question="What is reconstructed?",
            relevant_chunk_ids=frozenset(),
        )


def test_evaluator_calculates_case_metrics() -> None:
    index = FakeSearchIndex(
        [
            _result("chunk-1", 0, 0.9),
            _result("chunk-2", 1, 0.8),
            _result("chunk-3", 2, 0.7),
        ]
    )
    evaluator = RetrievalEvaluator(index)

    result = evaluator.evaluate_case(
        RetrievalCase(
            id="case-1",
            question="How is uncertainty measured?",
            relevant_chunk_ids=frozenset(
                {
                    "chunk-2",
                    "chunk-3",
                }
            ),
            top_k=2,
        )
    )

    assert result.retrieved_chunk_ids == (
        "chunk-1",
        "chunk-2",
    )
    assert result.recall_at_k == 0.5
    assert result.reciprocal_rank == 0.5
    assert result.hit


def test_report_aggregates_all_cases() -> None:
    index = FakeSearchIndex(
        [
            _result("chunk-1", 0, 0.9),
            _result("chunk-2", 1, 0.8),
            _result("chunk-3", 2, 0.7),
        ]
    )
    evaluator = RetrievalEvaluator(index)

    report = evaluator.evaluate(
        [
            RetrievalCase(
                id="case-1",
                question="First question",
                relevant_chunk_ids=frozenset({"chunk-2"}),
                top_k=2,
            ),
            RetrievalCase(
                id="case-2",
                question="Second question",
                relevant_chunk_ids=frozenset({"chunk-3"}),
                top_k=2,
            ),
        ]
    )

    assert report.case_count == 2
    assert report.mean_recall_at_k == 0.5
    assert report.mean_reciprocal_rank == 0.25
    assert report.hit_rate == 0.5
