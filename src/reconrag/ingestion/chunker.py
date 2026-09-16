"""Section-aware token chunking with citation provenance."""

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from transformers import AutoTokenizer

from reconrag.models import Chunk, ParsedBlock, ParsedPaper

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


@dataclass(frozen=True)
class _Segment:
    text: str
    page_numbers: tuple[int, ...]
    section_heading: str | None


class SectionAwareChunker:
    """Create bounded chunks without crossing detected sections."""

    def __init__(
        self,
        model_name: str,
        target_tokens: int = 350,
        overlap_tokens: int = 50,
        tokenizer: Any | None = None,
    ) -> None:
        if target_tokens < 1:
            raise ValueError("target_tokens must be positive.")

        if overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative.")

        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens.")

        self.model_name = model_name
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Any:
        """Load the embedding tokenizer only when required."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return self._tokenizer

    def chunk(
        self,
        paper: ParsedPaper,
    ) -> list[Chunk]:
        """Convert a paper into provenance-aware chunks."""
        output: list[Chunk] = []
        current: list[_Segment] = []
        current_section: str | None = None

        for block in paper.blocks:
            block_section = block.section_heading

            if current and block_section != current_section:
                output.append(
                    self._build_chunk(
                        paper,
                        current,
                        len(output),
                    )
                )
                current = []

            current_section = block_section

            if block.label == "section_header":
                continue

            for segment in self._segments_from_block(block):
                candidate = [*current, segment]

                if current and self._count_segments(candidate) > self.target_tokens:
                    output.append(
                        self._build_chunk(
                            paper,
                            current,
                            len(output),
                        )
                    )

                    current = self._overlap_tail(current)

                    while (
                        current
                        and self._count_segments([*current, segment])
                        > self.target_tokens
                    ):
                        current.pop(0)

                current.append(segment)

        if current:
            output.append(
                self._build_chunk(
                    paper,
                    current,
                    len(output),
                )
            )

        return output

    def _segments_from_block(
        self,
        block: ParsedBlock,
    ) -> list[_Segment]:
        """Split a block on sentence boundaries."""
        sentences = [
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY.split(block.text)
            if sentence.strip()
        ]

        segments: list[_Segment] = []

        for sentence in sentences:
            token_ids = self._encode(sentence)

            if len(token_ids) <= self.target_tokens:
                segments.append(
                    _Segment(
                        text=sentence,
                        page_numbers=tuple(block.page_numbers),
                        section_heading=(block.section_heading),
                    )
                )
                continue

            for start in range(
                0,
                len(token_ids),
                self.target_tokens,
            ):
                token_slice = token_ids[start : start + self.target_tokens]

                segments.append(
                    _Segment(
                        text=self.tokenizer.decode(
                            token_slice,
                            skip_special_tokens=True,
                        ).strip(),
                        page_numbers=tuple(block.page_numbers),
                        section_heading=(block.section_heading),
                    )
                )

        return segments

    def _overlap_tail(
        self,
        segments: list[_Segment],
    ) -> list[_Segment]:
        """Select trailing segments for the next chunk."""
        if self.overlap_tokens == 0:
            return []

        selected: list[_Segment] = []

        for segment in reversed(segments):
            selected.insert(0, segment)

            if self._count_segments(selected) >= self.overlap_tokens:
                break

        return selected

    def _build_chunk(
        self,
        paper: ParsedPaper,
        segments: list[_Segment],
        chunk_index: int,
    ) -> Chunk:
        """Build one validated chunk."""
        text = self._join_segments(segments)

        page_numbers = sorted(
            {page for segment in segments for page in segment.page_numbers}
        )

        section_heading = next(
            (
                segment.section_heading
                for segment in segments
                if segment.section_heading is not None
            ),
            None,
        )

        identity = "\0".join(
            [
                paper.checksum,
                str(chunk_index),
                section_heading or "",
                text,
            ]
        )

        return Chunk(
            id=sha256(identity.encode("utf-8")).hexdigest()[:24],
            document_id=paper.checksum,
            text=text,
            chunk_index=chunk_index,
            page_start=(min(page_numbers) if page_numbers else None),
            page_end=(max(page_numbers) if page_numbers else None),
            section_heading=section_heading,
            token_count=self._count_tokens(text),
        )

    def _count_segments(
        self,
        segments: list[_Segment],
    ) -> int:
        return self._count_tokens(self._join_segments(segments))

    def _count_tokens(
        self,
        text: str,
    ) -> int:
        return len(self._encode(text))

    def _encode(
        self,
        text: str,
    ) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    @staticmethod
    def _join_segments(
        segments: list[_Segment],
    ) -> str:
        return "\n\n".join(segment.text for segment in segments)
