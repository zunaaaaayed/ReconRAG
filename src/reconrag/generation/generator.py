"""Local answer generation through Ollama."""

import re
from time import perf_counter
from typing import Any, Protocol

from ollama import Client

from reconrag.generation.prompt import (
    SYSTEM_PROMPT,
    build_grounded_prompt,
)
from reconrag.models import Answer, SearchResult

THINKING_BLOCK = re.compile(
    r"<think>.*?</think>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _visible_response_text(
    response_text: str,
) -> str:
    """Remove model reasoning traces from visible output."""
    clean_text = THINKING_BLOCK.sub(
        "",
        response_text,
    ).strip()

    closing_tag = "</think>"
    closing_index = clean_text.lower().rfind(closing_tag)

    if closing_index != -1:
        clean_text = clean_text[closing_index + len(closing_tag) :].strip()

    return clean_text


class AnswerGenerator(Protocol):
    """Interface implemented by answer-generation backends."""

    def generate(
        self,
        question: str,
        evidence: list[SearchResult],
    ) -> Answer:
        """Generate an answer grounded in retrieved evidence."""
        ...


class OllamaAnswerGenerator:
    """Generate grounded answers with a local Ollama model."""

    def __init__(
        self,
        model_name: str,
        host: str = "http://localhost:11434",
        max_tokens: int = 500,
        client: Any | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.model_name = model_name
        self.host = host
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self) -> Any:
        """Create the Ollama client only when first required."""
        if self._client is None:
            self._client = Client(
                host=self.host,
            )

        return self._client

    def generate(
        self,
        question: str,
        evidence: list[SearchResult],
    ) -> Answer:
        """Generate an answer using only retrieved evidence."""
        user_prompt = build_grounded_prompt(
            question=question,
            evidence=evidence,
        )

        started_at = perf_counter()

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            stream=False,
            think=False,
            options={
                "temperature": 0.1,
                "num_predict": self.max_tokens,
            },
        )

        latency_seconds = perf_counter() - started_at

        raw_text = response.message.content

        if not isinstance(raw_text, str):
            raise RuntimeError("Ollama returned an invalid answer.")

        text = _visible_response_text(raw_text)

        if not text:
            raise RuntimeError("Ollama returned an empty answer.")

        return Answer(
            text=text.strip(),
            evidence=evidence,
            model_name=self.model_name,
            latency_seconds=latency_seconds,
        )
