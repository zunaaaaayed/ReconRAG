"""Deterministic validation of generated evidence citations."""

import re
from dataclasses import dataclass

BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)\]")
CITATION_GROUP_PATTERN = re.compile(r"\s*\d+(?:\s*,\s*\d+)*\s*")
NUMBER_PATTERN = re.compile(r"\d+")


@dataclass(frozen=True)
class CitationValidation:
    """Citation integrity metrics for one generated answer."""

    citation_occurrences: tuple[int, ...]
    valid_citations: tuple[int, ...]
    invalid_citations: tuple[int, ...]
    malformed_citations: tuple[str, ...]
    evidence_count: int

    @property
    def has_citations(self) -> bool:
        """Return whether any numeric citation was found."""
        return bool(self.citation_occurrences)

    @property
    def all_citations_valid(self) -> bool:
        """Return whether no invalid citation was found."""
        return not self.invalid_citations and not self.malformed_citations

    @property
    def is_valid(self) -> bool:
        """Return whether the answer has only valid citations."""
        return bool(self.valid_citations) and self.all_citations_valid

    @property
    def cited_evidence_count(self) -> int:
        """Return the number of distinct evidence items cited."""
        return len(self.valid_citations)

    @property
    def evidence_coverage(self) -> float:
        """Return the fraction of supplied evidence cited."""
        if self.evidence_count == 0:
            return 0.0

        return self.cited_evidence_count / self.evidence_count

    @property
    def unused_evidence(self) -> tuple[int, ...]:
        """Return evidence numbers not cited by the answer."""
        cited = set(self.valid_citations)

        return tuple(
            citation_number
            for citation_number in range(
                1,
                self.evidence_count + 1,
            )
            if citation_number not in cited
        )


class CitationValidator:
    """Validate bracketed citations against supplied evidence."""

    def validate(
        self,
        text: str,
        evidence_count: int,
    ) -> CitationValidation:
        """Parse and validate citations in generated text."""
        if evidence_count < 0:
            raise ValueError("evidence_count cannot be negative.")

        occurrences: list[int] = []
        malformed: list[str] = []

        for match in BRACKET_PATTERN.finditer(text):
            raw_citation = match.group(0)
            content = match.group(1)

            if CITATION_GROUP_PATTERN.fullmatch(content):
                occurrences.extend(
                    int(number) for number in NUMBER_PATTERN.findall(content)
                )
                continue

            if content.lstrip()[:1].isdigit():
                malformed.append(raw_citation)

        valid_occurrences = [
            citation for citation in occurrences if 1 <= citation <= evidence_count
        ]
        invalid_occurrences = [
            citation
            for citation in occurrences
            if citation < 1 or citation > evidence_count
        ]

        return CitationValidation(
            citation_occurrences=tuple(occurrences),
            valid_citations=self._unique(valid_occurrences),
            invalid_citations=self._unique(invalid_occurrences),
            malformed_citations=tuple(malformed),
            evidence_count=evidence_count,
        )

    @staticmethod
    def _unique(
        values: list[int],
    ) -> tuple[int, ...]:
        """Return unique integers in first-seen order."""
        return tuple(dict.fromkeys(values))
