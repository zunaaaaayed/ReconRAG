"""Tests for JSON retrieval benchmarks."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reconrag.evaluation import (
    RetrievalBenchmark,
    load_benchmark,
)


def test_load_benchmark_converts_cases(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "name": "Test benchmark",
                "description": ("Small retrieval benchmark."),
                "cases": [
                    {
                        "id": "case-1",
                        "question": ("How is geometry optimized?"),
                        "relevant_chunk_ids": [
                            "chunk-1",
                            "chunk-2",
                        ],
                        "top_k": 3,
                        "document_ids": ["paper-1"],
                        "section_heading": ("Methods"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    benchmark = load_benchmark(benchmark_path)
    cases = benchmark.to_retrieval_cases()

    assert benchmark.name == "Test benchmark"
    assert len(cases) == 1
    assert cases[0].id == "case-1"
    assert cases[0].top_k == 3
    assert cases[0].relevant_chunk_ids == (
        frozenset(
            {
                "chunk-1",
                "chunk-2",
            }
        )
    )
    assert cases[0].document_ids == (frozenset({"paper-1"}))
    assert cases[0].section_heading == "Methods"


def test_benchmark_rejects_duplicate_case_ids() -> None:
    with pytest.raises(
        ValidationError,
        match="case IDs must be unique",
    ):
        RetrievalBenchmark.model_validate(
            {
                "name": "Duplicates",
                "cases": [
                    {
                        "id": "same-id",
                        "question": "Question one",
                        "relevant_chunk_ids": ["chunk-1"],
                    },
                    {
                        "id": "same-id",
                        "question": "Question two",
                        "relevant_chunk_ids": ["chunk-2"],
                    },
                ],
            }
        )


def test_benchmark_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RetrievalBenchmark.model_validate(
            {
                "name": "Invalid benchmark",
                "unexpected": True,
                "cases": [
                    {
                        "id": "case-1",
                        "question": "Question",
                        "relevant_chunk_ids": ["chunk-1"],
                    }
                ],
            }
        )
