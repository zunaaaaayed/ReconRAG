"""Tests for section-aware token chunking."""

from reconrag.ingestion.chunker import (
    SectionAwareChunker,
)
from reconrag.models import (
    ParsedBlock,
    ParsedPaper,
)


class FakeTokenizer:
    """A deterministic tokenizer for unit tests."""

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._reverse_vocabulary: dict[int, str] = {}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens

        token_ids: list[int] = []

        for word in text.split():
            if word not in self._vocabulary:
                token_id = len(self._vocabulary) + 1

                self._vocabulary[word] = token_id
                self._reverse_vocabulary[token_id] = word

            token_ids.append(self._vocabulary[word])

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens

        return " ".join(self._reverse_vocabulary[token_id] for token_id in token_ids)


def _paper() -> ParsedPaper:
    return ParsedPaper(
        title="Test Paper",
        filename="test.pdf",
        checksum="abc123",
        page_count=3,
        markdown="# Test Paper",
        blocks=[
            ParsedBlock(
                text="Introduction",
                label="section_header",
                page_numbers=[1],
                section_heading="Introduction",
            ),
            ParsedBlock(
                text=("one two three four. Five six seven eight."),
                label="text",
                page_numbers=[1, 2],
                section_heading="Introduction",
            ),
            ParsedBlock(
                text="Methods",
                label="section_header",
                page_numbers=[2],
                section_heading="Methods",
            ),
            ParsedBlock(
                text=("nine ten eleven twelve. Thirteen fourteen fifteen sixteen."),
                label="text",
                page_numbers=[2, 3],
                section_heading="Methods",
            ),
        ],
    )


def _chunker() -> SectionAwareChunker:
    return SectionAwareChunker(
        model_name="fake-model",
        target_tokens=8,
        overlap_tokens=2,
        tokenizer=FakeTokenizer(),
    )


def test_chunks_respect_token_limit_and_sections() -> None:
    chunks = _chunker().chunk(_paper())

    assert chunks

    assert all(chunk.token_count <= 8 for chunk in chunks)

    assert all(
        not ("Introduction" in chunk.text and "Methods" in chunk.text)
        for chunk in chunks
    )


def test_chunks_preserve_page_ranges() -> None:
    chunks = _chunker().chunk(_paper())

    introduction_chunks = [
        chunk for chunk in chunks if chunk.section_heading == "Introduction"
    ]

    assert introduction_chunks[0].page_start == 1

    assert max(chunk.page_end or 0 for chunk in introduction_chunks) == 2


def test_chunk_ids_are_deterministic() -> None:
    first = _chunker().chunk(_paper())
    second = _chunker().chunk(_paper())

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


def test_section_headings_are_metadata_not_standalone_chunks() -> None:
    chunks = _chunker().chunk(_paper())

    assert {chunk.section_heading for chunk in chunks} == {
        "Introduction",
        "Methods",
    }
    assert all(chunk.text not in {"Introduction", "Methods"} for chunk in chunks)
