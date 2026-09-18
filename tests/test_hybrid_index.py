"""Tests for reciprocal-rank-fused retrieval."""

from datetime import UTC, datetime

import pytest

from reconrag.models import (
    Chunk,
    Document,
    SearchResult,
)
from reconrag.retrieval import (
    HybridSearchIndex,
)


class FakeSearchIndex:
    """Return a fixed ranking and record search calls."""

    def __init__(
        self,
        results: list[SearchResult],
    ) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    @property
    def size(self) -> int:
        return len(self.results)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
        section_heading: str | None = None,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "document_ids": document_ids,
                "section_heading": (section_heading),
            }
        )

        return self.results[:top_k]


def _result(
    chunk_id: str,
    chunk_index: int,
) -> SearchResult:
    document = Document(
        id="paper-1",
        title="Test Paper",
        filename="test.pdf",
        checksum="checksum-1",
        ingested_at=datetime.now(UTC),
    )

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
        score=1.0,
    )


def test_hybrid_fuses_semantic_and_lexical_rankings() -> None:
    semantic = FakeSearchIndex(
        [
            _result("chunk-a", 0),
            _result("chunk-b", 1),
            _result("chunk-c", 2),
        ]
    )
    lexical = FakeSearchIndex(
        [
            _result("chunk-b", 1),
            _result("chunk-d", 3),
            _result("chunk-a", 0),
        ]
    )
    hybrid = HybridSearchIndex(
        semantic,
        lexical,
    )

    results = hybrid.search(
        "reconstruction uncertainty",
        top_k=4,
    )

    assert [result.chunk.id for result in results] == [
        "chunk-b",
        "chunk-a",
        "chunk-d",
        "chunk-c",
    ]


def test_hybrid_deduplicates_shared_results() -> None:
    semantic = FakeSearchIndex([_result("chunk-a", 0)])
    lexical = FakeSearchIndex([_result("chunk-a", 0)])
    hybrid = HybridSearchIndex(
        semantic,
        lexical,
    )

    results = hybrid.search("shared result")

    assert len(results) == 1
    assert results[0].chunk.id == "chunk-a"


def test_hybrid_propagates_filters_and_candidate_count() -> None:
    semantic = FakeSearchIndex([_result("chunk-a", 0)])
    lexical = FakeSearchIndex([_result("chunk-b", 1)])
    hybrid = HybridSearchIndex(
        semantic,
        lexical,
        candidate_multiplier=3,
    )

    hybrid.search(
        query="filtered query",
        top_k=2,
        document_ids={"paper-1"},
        section_heading="Methods",
    )

    expected_call = {
        "query": "filtered query",
        "top_k": 6,
        "document_ids": {"paper-1"},
        "section_heading": "Methods",
    }

    assert semantic.calls == [expected_call]
    assert lexical.calls == [expected_call]


def test_hybrid_falls_back_to_available_ranking() -> None:
    semantic = FakeSearchIndex(
        [
            _result("chunk-a", 0),
            _result("chunk-b", 1),
        ]
    )
    lexical = FakeSearchIndex([])
    hybrid = HybridSearchIndex(
        semantic,
        lexical,
    )

    results = hybrid.search(
        "semantic only",
        top_k=2,
    )

    assert [result.chunk.id for result in results] == [
        "chunk-a",
        "chunk-b",
    ]


def test_hybrid_validates_configuration() -> None:
    empty_index = FakeSearchIndex([])

    with pytest.raises(
        ValueError,
        match="rrf_k must be positive",
    ):
        HybridSearchIndex(
            empty_index,
            empty_index,
            rrf_k=0,
        )

    with pytest.raises(
        ValueError,
        match="candidate_multiplier",
    ):
        HybridSearchIndex(
            empty_index,
            empty_index,
            candidate_multiplier=0,
        )

    with pytest.raises(
        ValueError,
        match="At least one retrieval weight",
    ):
        HybridSearchIndex(
            empty_index,
            empty_index,
            semantic_weight=0,
            lexical_weight=0,
        )

    hybrid = HybridSearchIndex(
        empty_index,
        empty_index,
    )

    assert hybrid.search("   ") == []

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        hybrid.search(
            "query",
            top_k=0,
        )
