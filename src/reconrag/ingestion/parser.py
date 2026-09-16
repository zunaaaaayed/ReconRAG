"""Structured PDF parsing with Docling."""

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel

from reconrag.models import ParsedBlock, ParsedPaper


class PdfParsingError(RuntimeError):
    """Raised when Docling cannot parse a PDF."""


class PdfParser:
    """Convert PDFs into structured, citation-aware paper data."""

    def __init__(self, converter: Any | None = None) -> None:
        self._converter = converter

    @property
    def converter(self) -> Any:
        """Create the expensive Docling converter only when first needed."""
        if self._converter is None:
            self._converter = DocumentConverter()
        return self._converter

    def parse(self, pdf_path: Path) -> ParsedPaper:
        """Parse a PDF while preserving labels, sections, and page numbers."""
        try:
            result = self.converter.convert(pdf_path)
            document = result.document
            blocks = list(_extract_blocks(document))
            markdown = document.export_to_markdown()
        except Exception as exc:
            raise PdfParsingError(f"Could not parse {pdf_path.name}: {exc}") from exc

        title = next(
            (block.text for block in blocks if block.label == DocItemLabel.TITLE.value),
            pdf_path.stem,
        )
        pages = getattr(document, "pages", {})

        return ParsedPaper(
            title=title,
            filename=pdf_path.name,
            checksum=_sha256_file(pdf_path),
            page_count=len(pages),
            blocks=blocks,
            markdown=markdown,
        )


def _extract_blocks(document: Any) -> Iterator[ParsedBlock]:
    """Yield text-bearing Docling items with their provenance."""
    current_section: str | None = None

    for item, _level in document.iterate_items():
        text = getattr(item, "text", "").strip()
        if not text:
            continue

        label = _label_value(getattr(item, "label", "text"))

        if label == DocItemLabel.SECTION_HEADER.value:
            current_section = text

        page_numbers = sorted(
            {
                int(provenance.page_no)
                for provenance in (getattr(item, "prov", None) or [])
                if getattr(provenance, "page_no", None) is not None
            }
        )

        yield ParsedBlock(
            text=text,
            label=label,
            page_numbers=page_numbers,
            section_heading=current_section,
        )


def _label_value(label: Any) -> str:
    """Normalize Docling enum labels and test doubles to plain strings."""
    return str(getattr(label, "value", label))


def _sha256_file(path: Path) -> str:
    """Calculate a checksum without loading the entire PDF into memory."""
    digest = sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()
