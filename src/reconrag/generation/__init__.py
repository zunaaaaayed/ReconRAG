"""Evidence-grounded answer generation."""

from reconrag.generation.generator import (
    AnswerGenerator,
    OllamaAnswerGenerator,
)
from reconrag.generation.prompt import (
    SYSTEM_PROMPT,
    build_grounded_prompt,
)

__all__ = [
    "AnswerGenerator",
    "OllamaAnswerGenerator",
    "SYSTEM_PROMPT",
    "build_grounded_prompt",
]
