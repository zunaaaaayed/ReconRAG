"""Evidence-grounded answer generation."""

from reconrag.generation.citations import (
    CitationValidation,
    CitationValidator,
)
from reconrag.generation.generator import (
    AnswerGenerator,
    OllamaAnswerGenerator,
)
from reconrag.generation.prompt import (
    SYSTEM_PROMPT,
    build_citation_repair_prompt,
    build_grounded_prompt,
)

__all__ = [
    "AnswerGenerator",
    "CitationValidation",
    "CitationValidator",
    "OllamaAnswerGenerator",
    "SYSTEM_PROMPT",
    "build_citation_repair_prompt",
    "build_grounded_prompt",
]
