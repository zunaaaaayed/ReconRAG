"""Evaluation contracts and orchestration for semantic retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from reconrag.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
)
from reconrag.models import SearchResult


class SearchIndex(Protocol):
    """Search behavior required by the retrieval evaluator."""

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
        section_heading: str | None = None,
    ) -> list[SearchResult]:
        """Return ranked retrieval results."""


@dataclass(frozen=True)
class RetrievalCase:
    """One benchmark question with known relevant chunks."""

    id: str
    question: str
    relevant_chunk_ids: frozenset[str]
    top_k: int = 5
    document_ids: frozenset[str] | None = None
    section_heading: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Evaluation case ID cannot be empty.")

        if not self.question.strip():
            raise ValueError("Evaluation question cannot be empty.")

        if not self.relevant_chunk_ids:
            raise ValueError("At least one relevant chunk ID is required.")

        if self.top_k < 1:
            raise ValueError("top_k must be positive.")


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Metrics and rankings produced for one benchmark case."""

    case_id: str
    question: str
    retrieved_chunk_ids: tuple[str, ...]
    relevant_chunk_ids: frozenset[str]
    recall_at_k: float
    reciprocal_rank: float

    @property
    def hit(self) -> bool:
        """Return whether at least one relevant chunk was retrieved."""
        return self.reciprocal_rank > 0.0


@dataclass(frozen=True)
class RetrievalReport:
    """Aggregate retrieval results for a benchmark."""

    results: tuple[RetrievalEvaluation, ...]

    @property
    def case_count(self) -> int:
        """Return the number of evaluated questions."""
        return len(self.results)

    @property
    def mean_recall_at_k(self) -> float:
        """Return mean Recall@k across all cases."""
        if not self.results:
            return 0.0

        return sum(result.recall_at_k for result in self.results) / len(self.results)

    @property
    def mean_reciprocal_rank(self) -> float:
        """Return mean reciprocal rank across all cases."""
        if not self.results:
            return 0.0

        return sum(result.reciprocal_rank for result in self.results) / len(
            self.results
        )

    @property
    def hit_rate(self) -> float:
        """Return the fraction of cases with a relevant result."""
        if not self.results:
            return 0.0

        return sum(result.hit for result in self.results) / len(self.results)


class RetrievalEvaluator:
    """Evaluate a search index against labelled questions."""

    def __init__(
        self,
        index: SearchIndex,
    ) -> None:
        self.index = index

    def evaluate_case(
        self,
        case: RetrievalCase,
    ) -> RetrievalEvaluation:
        """Evaluate one retrieval case."""
        results = self.index.search(
            query=case.question,
            top_k=case.top_k,
            document_ids=(
                set(case.document_ids) if case.document_ids is not None else None
            ),
            section_heading=case.section_heading,
        )

        retrieved_chunk_ids = tuple(result.chunk.id for result in results)

        return RetrievalEvaluation(
            case_id=case.id,
            question=case.question,
            retrieved_chunk_ids=(retrieved_chunk_ids),
            relevant_chunk_ids=(case.relevant_chunk_ids),
            recall_at_k=recall_at_k(
                retrieved_ids=(retrieved_chunk_ids),
                relevant_ids=(case.relevant_chunk_ids),
                k=case.top_k,
            ),
            reciprocal_rank=reciprocal_rank(
                retrieved_ids=(retrieved_chunk_ids),
                relevant_ids=(case.relevant_chunk_ids),
                k=case.top_k,
            ),
        )

    def evaluate(
        self,
        cases: Sequence[RetrievalCase],
    ) -> RetrievalReport:
        """Evaluate all supplied benchmark cases."""
        return RetrievalReport(
            results=tuple(self.evaluate_case(case) for case in cases)
        )
