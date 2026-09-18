"""Tests for generated-answer citation validation."""

import pytest

from reconrag.generation import CitationValidator


def test_validator_accepts_individual_and_grouped_citations() -> None:
    validation = CitationValidator().validate(
        (
            "Calibration changes coverage [1]. "
            "Ranking requires separate analysis "
            "[2, 3]. The first result is also "
            "relevant [1]."
        ),
        evidence_count=4,
    )

    assert validation.citation_occurrences == (
        1,
        2,
        3,
        1,
    )
    assert validation.valid_citations == (
        1,
        2,
        3,
    )
    assert validation.invalid_citations == ()
    assert validation.malformed_citations == ()
    assert validation.is_valid
    assert validation.evidence_coverage == 0.75
    assert validation.unused_evidence == (4,)


def test_validator_rejects_out_of_range_citations() -> None:
    validation = CitationValidator().validate(
        "Supported claim [2], bad claims [0, 4].",
        evidence_count=3,
    )

    assert validation.valid_citations == (2,)
    assert validation.invalid_citations == (
        0,
        4,
    )
    assert validation.has_citations
    assert not validation.all_citations_valid
    assert not validation.is_valid


def test_validator_detects_malformed_numeric_citations() -> None:
    validation = CitationValidator().validate(
        ("Unsupported ranges [1-3] and mixed citations [2; 3] are invalid."),
        evidence_count=3,
    )

    assert validation.citation_occurrences == ()
    assert validation.malformed_citations == (
        "[1-3]",
        "[2; 3]",
    )
    assert not validation.is_valid


def test_validator_ignores_non_citation_brackets() -> None:
    validation = CitationValidator().validate(
        (
            "The appendix [Figure A] provides "
            "additional context, but this answer "
            "contains no evidence citation."
        ),
        evidence_count=2,
    )

    assert not validation.has_citations
    assert validation.all_citations_valid
    assert not validation.is_valid
    assert validation.evidence_coverage == 0.0
    assert validation.unused_evidence == (
        1,
        2,
    )


def test_validator_handles_zero_evidence() -> None:
    validation = CitationValidator().validate(
        "An unsupported answer [1].",
        evidence_count=0,
    )

    assert validation.invalid_citations == (1,)
    assert validation.evidence_coverage == 0.0
    assert not validation.is_valid


def test_validator_rejects_negative_evidence_count() -> None:
    with pytest.raises(
        ValueError,
        match="evidence_count cannot be negative",
    ):
        CitationValidator().validate(
            "Answer [1].",
            evidence_count=-1,
        )
