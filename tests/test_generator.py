"""Tests for evidence-grounded answer generation."""

from datetime import UTC, datetime

import pytest

from reconrag.generation.generator import (
    OllamaAnswerGenerator,
)
from reconrag.generation.prompt import (
    SYSTEM_PROMPT,
    build_citation_repair_prompt,
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
    assert "Do not reproduce bibliography" in SYSTEM_PROMPT
    assert "Preserve numerical values" in SYSTEM_PROMPT


class SequenceMessage:
    """Fake Ollama message with configurable content."""

    def __init__(
        self,
        content: str,
    ) -> None:
        self.content = content


class SequenceResponse:
    """Fake Ollama response with configurable content."""

    def __init__(
        self,
        content: str,
    ) -> None:
        self.message = SequenceMessage(content)


class SequenceClient:
    """Return configured responses in sequence."""

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        **kwargs: object,
    ) -> SequenceResponse:
        self.calls.append(kwargs)
        response_index = len(self.calls) - 1

        return SequenceResponse(self.responses[response_index])


def test_repair_prompt_limits_allowed_citations() -> None:
    prompt = build_citation_repair_prompt(
        evidence_count=3,
        invalid_citations=(12,),
    )

    assert "between [1] and [3]" in prompt
    assert "[12]" in prompt
    assert "bibliography" in prompt
    assert "Do not add new claims" in prompt
    assert "complete corrected answer" in prompt


def test_generator_repairs_missing_citations() -> None:
    client = SequenceClient(
        [
            ("The method represents the volume with Gaussian primitives."),
            ("The method represents the volume with Gaussian primitives [1]."),
        ]
    )
    generator = OllamaAnswerGenerator(
        model_name="test-model",
        client=client,
    )

    answer = generator.generate(
        question="How is the volume represented?",
        evidence=_evidence(),
    )

    assert answer.text.endswith("[1].")
    assert len(client.calls) == 2

    repair_call = client.calls[1]
    messages = repair_call["messages"]

    assert isinstance(messages, list)
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    assert "citations are valid" in (messages[-1]["content"])


def test_generator_attempts_only_one_repair() -> None:
    client = SequenceClient(
        [
            "Initial answer with invalid citation [2].",
            "Repaired answer is still invalid [3].",
        ]
    )
    generator = OllamaAnswerGenerator(
        model_name="test-model",
        client=client,
    )

    answer = generator.generate(
        question="How is the volume represented?",
        evidence=_evidence(),
    )

    assert answer.text == ("Repaired answer is still invalid [3].")
    assert len(client.calls) == 2


def test_repair_prompt_requires_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="evidence passage",
    ):
        build_citation_repair_prompt(evidence_count=0)


def test_prompt_normalizes_spaced_decimals() -> None:
    evidence = _evidence()
    original_result = evidence[0]

    normalized_chunk = original_result.chunk.model_copy(
        update={
            "text": ("Foreground is defined as values above 0 . 05 times the peak.")
        }
    )

    evidence[0] = original_result.model_copy(update={"chunk": normalized_chunk})

    prompt = build_grounded_prompt(
        question="How is foreground defined?",
        evidence=evidence,
    )

    assert "0.05 times the peak" in prompt
    assert "0 . 05" not in prompt


def test_generator_repairs_copied_bibliography_citation() -> None:
    client = SequenceClient(
        [
            ("AUSE [12] evaluates error removal using the supplied evidence [1]."),
            ("AUSE evaluates error removal using the supplied evidence [1]."),
        ]
    )
    generator = OllamaAnswerGenerator(
        model_name="test-model",
        client=client,
    )

    answer = generator.generate(
        question="Which metrics are used?",
        evidence=_evidence(),
    )

    assert "[12]" not in answer.text
    assert "[1]" in answer.text
    assert len(client.calls) == 2
