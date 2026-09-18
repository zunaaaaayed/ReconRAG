"""Hybrid semantic and lexical retrieval using rank fusion."""

from typing import Protocol

from reconrag.models import SearchResult


class RankedSearchIndex(Protocol):
    """Search behavior required by hybrid retrieval."""

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
        section_heading: str | None = None,
    ) -> list[SearchResult]:
        """Return ranked search results."""


class HybridSearchIndex:
    """Fuse semantic and BM25 rankings with weighted RRF."""

    def __init__(
        self,
        semantic_index: RankedSearchIndex,
        lexical_index: RankedSearchIndex,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive.")

        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive.")

        if semantic_weight < 0:
            raise ValueError("semantic_weight cannot be negative.")

        if lexical_weight < 0:
            raise ValueError("lexical_weight cannot be negative.")

        if semantic_weight == 0 and lexical_weight == 0:
            raise ValueError("At least one retrieval weight must be positive.")

        self.semantic_index = semantic_index
        self.lexical_index = lexical_index
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

    @property
    def size(self) -> int:
        """Return the largest underlying index size."""
        return max(
            self.semantic_index.size,
            self.lexical_index.size,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
        section_heading: str | None = None,
    ) -> list[SearchResult]:
        """Return reciprocal-rank-fused results."""
        clean_query = query.strip()

        if not clean_query:
            return []

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        candidate_k = top_k * self.candidate_multiplier

        semantic_results = self.semantic_index.search(
            query=clean_query,
            top_k=candidate_k,
            document_ids=document_ids,
            section_heading=section_heading,
        )
        lexical_results = self.lexical_index.search(
            query=clean_query,
            top_k=candidate_k,
            document_ids=document_ids,
            section_heading=section_heading,
        )

        fused_scores: dict[str, float] = {}
        results_by_chunk_id: dict[
            str,
            SearchResult,
        ] = {}

        self._add_ranking(
            results=semantic_results,
            weight=self.semantic_weight,
            fused_scores=fused_scores,
            results_by_chunk_id=(results_by_chunk_id),
        )
        self._add_ranking(
            results=lexical_results,
            weight=self.lexical_weight,
            fused_scores=fused_scores,
            results_by_chunk_id=(results_by_chunk_id),
        )

        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                results_by_chunk_id[chunk_id].chunk.chunk_index,
                chunk_id,
            ),
        )[:top_k]

        return [
            SearchResult(
                chunk=results_by_chunk_id[chunk_id].chunk,
                document=results_by_chunk_id[chunk_id].document,
                score=fused_scores[chunk_id],
            )
            for chunk_id in ranked_chunk_ids
        ]

    def _add_ranking(
        self,
        results: list[SearchResult],
        weight: float,
        fused_scores: dict[str, float],
        results_by_chunk_id: dict[
            str,
            SearchResult,
        ],
    ) -> None:
        """Add one weighted ranking to the fused score."""
        if weight == 0:
            return

        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = result.chunk.id

            results_by_chunk_id.setdefault(
                chunk_id,
                result,
            )

            fused_scores[chunk_id] = fused_scores.get(
                chunk_id,
                0.0,
            ) + weight / (self.rrf_k + rank)
