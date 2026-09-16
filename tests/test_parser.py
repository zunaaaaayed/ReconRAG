"""Tests for Docling-to-ReconRAG conversion."""

from dataclasses import dataclass

from reconrag.ingestion.parser import PdfParser


@dataclass
class FakeLabel:
    value: str


@dataclass
class FakeProvenance:
    page_no: int


@dataclass
class FakeItem:
    text: str
    label: FakeLabel
    prov: list[FakeProvenance]


class FakeDocument:
    pages = {
        1: object(),
        2: object(),
    }

    def iterate_items(self):
        yield (
            FakeItem(
                "A CT Reconstruction Paper",
                FakeLabel("title"),
                [FakeProvenance(1)],
            ),
            0,
        )

        yield (
            FakeItem(
                "Methods",
                FakeLabel("section_header"),
                [FakeProvenance(2)],
            ),
            0,
        )

        yield (
            FakeItem(
                "We reconstruct sparse projections.",
                FakeLabel("text"),
                [FakeProvenance(2)],
            ),
            1,
        )

    def export_to_markdown(self) -> str:
        return "# A CT Reconstruction Paper\n\n## Methods"


@dataclass
class FakeResult:
    document: FakeDocument


class FakeConverter:
    def convert(self, _path):
        return FakeResult(document=FakeDocument())


def test_parser_preserves_title_section_and_page(
    tmp_path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\ntest")

    parser = PdfParser(converter=FakeConverter())
    paper = parser.parse(pdf_path)

    assert paper.title == ("A CT Reconstruction Paper")
    assert paper.page_count == 2
    assert paper.blocks[-1].page_numbers == [2]
    assert paper.blocks[-1].section_heading == "Methods"
