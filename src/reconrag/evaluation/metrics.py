"""Metrics for evaluating retrieval quality."""

from collections.abc import Sequence, Set


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Return the fraction of relevant chunks retrieved in the top k."""
    if k < 1:
        raise ValueError("k must be positive.")

    if not relevant_ids:
        raise ValueError("At least one relevant chunk ID is required.")

    retrieved_at_k = set(retrieved_ids[:k])

    return len(retrieved_at_k.intersection(relevant_ids)) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Return the reciprocal rank of the first relevant top-k result."""
    if k < 1:
        raise ValueError("k must be positive.")

    if not relevant_ids:
        raise ValueError("At least one relevant chunk ID is required.")

    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0
