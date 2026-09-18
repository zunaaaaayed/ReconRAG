"""JSON benchmark loading and validation."""

from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from reconrag.evaluation.evaluator import (
    RetrievalCase,
)


class BenchmarkCase(BaseModel):
    """Serializable retrieval question and relevance labels."""

    model_config = ConfigDict(
        extra="forbid",
    )

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_ids: list[str] | None = None
    section_heading: str | None = None

    def to_retrieval_case(self) -> RetrievalCase:
        """Convert the serialized case into evaluator input."""
        return RetrievalCase(
            id=self.id,
            question=self.question,
            relevant_chunk_ids=frozenset(self.relevant_chunk_ids),
            top_k=self.top_k,
            document_ids=(
                frozenset(self.document_ids) if self.document_ids is not None else None
            ),
            section_heading=self.section_heading,
        )


class RetrievalBenchmark(BaseModel):
    """A named collection of retrieval questions."""

    model_config = ConfigDict(
        extra="forbid",
    )

    name: str = Field(min_length=1)
    description: str | None = None
    cases: list[BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> Self:
        """Reject ambiguous duplicate case identifiers."""
        case_ids = [case.id for case in self.cases]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Benchmark case IDs must be unique.")

        return self

    def to_retrieval_cases(
        self,
    ) -> list[RetrievalCase]:
        """Convert all serialized cases to evaluator input."""
        return [case.to_retrieval_case() for case in self.cases]


def load_benchmark(
    path: Path,
) -> RetrievalBenchmark:
    """Load and validate a retrieval benchmark JSON file."""
    return RetrievalBenchmark.model_validate_json(path.read_text(encoding="utf-8"))
