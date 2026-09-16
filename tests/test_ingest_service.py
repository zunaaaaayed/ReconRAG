"""Tests for safe local PDF storage."""

from reconrag.services.ingest_service import store_pdf


def test_store_pdf_reuses_identical_file(tmp_path) -> None:
    content = b"%PDF-1.4\nexample"

    first = store_pdf(
        content,
        "paper.pdf",
        tmp_path,
    )
    second = store_pdf(
        content,
        "paper.pdf",
        tmp_path,
    )

    assert first.created is True
    assert second.created is False
    assert first.path == second.path


def test_store_pdf_renames_different_content(tmp_path) -> None:
    first = store_pdf(
        b"%PDF-1.4\nfirst",
        "paper.pdf",
        tmp_path,
    )
    second = store_pdf(
        b"%PDF-1.4\nsecond",
        "paper.pdf",
        tmp_path,
    )

    assert first.path.name == "paper.pdf"
    assert second.path.name.startswith("paper-")
    assert second.path.suffix == ".pdf"


def test_store_pdf_rejects_non_pdf_content(tmp_path) -> None:
    try:
        store_pdf(
            b"not a PDF",
            "paper.pdf",
            tmp_path,
        )
    except ValueError as exc:
        assert "valid PDF" in str(exc)
    else:
        raise AssertionError("Invalid PDF content should be rejected")
