"""Tests for evidence-grounded answer generation."""

from datetime import UTC, datetime

import pytest

from reconrag.generation.generator import (
    OllamaAnswerGenerator,
)
from reconrag.generation.prompt import (
    SYSTEM_PROMPT,
    build_grounded_prompt,
)
from reconrag.models import Chunk, Document, SearchResult


class FakeMessage:
    """Fake Ollama response message."""

    content = "The method uses Gaussian primitives for reconstruction [1]."


class FakeResponse:
    """Fake Ollama chat response."""

    message = FakeMessage()


class FakeClient:
    """Record Ollama calls without running a model."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        **kwargs: object,
    ) -> FakeResponse:
        self.calls.append(kwargs)

        return FakeResponse()


def _evidence() -> list[SearchResult]:
    document = Document(
        id="paper-one",
        title="Gaussian Reconstruction",
        filename="gaussian.pdf",
        checksum="paper-one",
        ingested_at=datetime.now(UTC),
    )

    chunk = Chunk(
        id="chunk-one",
        document_id=document.id,
        text=("The method represents the volume with Gaussian primitives."),
        chunk_index=0,
        page_start=3,
        page_end=4,
        section_heading="Method",
        token_count=10,
    )

    return [
        SearchResult(
            chunk=chunk,
            document=document,
            score=0.91,
        )
    ]


def test_prompt_contains_numbered_evidence_and_metadata() -> None:
    prompt = build_grounded_prompt(
        question=("How is the volume represented?"),
        evidence=_evidence(),
    )

    assert "[1]" in prompt
    assert "Gaussian Reconstruction" in prompt
    assert "Section: Method" in prompt
    assert "Pages: 3–4" in prompt
    assert "Gaussian primitives" in prompt


def test_generator_returns_answer_with_evidence() -> None:
    client = FakeClient()

    generator = OllamaAnswerGenerator(
        model_name="test-model",
        client=client,
    )

    answer = generator.generate(
        question=("How is the volume represented?"),
        evidence=_evidence(),
    )

    assert answer.text.endswith("[1].")
    assert answer.model_name == "test-model"
    assert len(answer.evidence) == 1
    assert answer.latency_seconds >= 0
    assert len(client.calls) == 1

    call = client.calls[0]

    assert call["model"] == "test-model"
    assert call["stream"] is False
    assert call["think"] is False


def test_prompt_requires_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="evidence passage",
    ):
        build_grounded_prompt(
            question="What is the method?",
            evidence=[],
        )


class ThinkingMessage:
    """Fake response containing an exposed reasoning trace."""

    content = (
        "This reasoning must remain hidden."
        "\n</think>\n\n"
        "The grounded answer is visible [1]."
    )


class ThinkingResponse:
    """Fake Ollama response with reasoning content."""

    message = ThinkingMessage()


class ThinkingClient:
    """Return an answer containing a reasoning trace."""

    def chat(
        self,
        **kwargs: object,
    ) -> ThinkingResponse:
        return ThinkingResponse()


def test_generator_removes_reasoning_trace() -> None:
    generator = OllamaAnswerGenerator(
        model_name="test-model",
        client=ThinkingClient(),
    )

    answer = generator.generate(
        question=("How is the volume represented?"),
        evidence=_evidence(),
    )

    assert answer.text == ("The grounded answer is visible [1].")
    assert "reasoning" not in answer.text
    assert "</think>" not in answer.text


def test_system_prompt_requests_final_answer_only() -> None:
    assert "Return only the final answer" in SYSTEM_PROMPT
    assert "Never reveal planning" in SYSTEM_PROMPT
