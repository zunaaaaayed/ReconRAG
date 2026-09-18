"""Dependency-free BM25 retrieval for paper chunks."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from reconrag.models import (
    Chunk,
    Document,
    SearchResult,
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class _LexicalRecord:
    """One indexed chunk and its token statistics."""

    document: Document
    chunk: Chunk
    term_counts: Counter[str]
    length: int


class BM25Index:
    """Rank chunks using BM25 lexical relevance."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one.")

        self.k1 = k1
        self.b = b
        self._records: list[_LexicalRecord] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_document_length = 0.0

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._records)

    def build(
        self,
        entries: list[tuple[Document, list[Chunk]]],
    ) -> None:
        """Replace the index with supplied documents and chunks."""
        raw_records = [
            (
                document,
                chunk,
                self._tokenize(
                    self._search_text(
                        document,
                        chunk,
                    )
                ),
            )
            for document, chunks in entries
            for chunk in chunks
        ]

        if not raw_records:
            self._records = []
            self._document_frequency = Counter()
            self._average_document_length = 0.0
            return

        document_frequency: Counter[str] = Counter()

        for _, _, tokens in raw_records:
            document_frequency.update(set(tokens))

        records = [
            _LexicalRecord(
                document=document,
                chunk=chunk,
                term_counts=Counter(tokens),
                length=len(tokens),
            )
            for document, chunk, tokens in raw_records
        ]

        self._records = records
        self._document_frequency = document_frequency
        self._average_document_length = sum(record.length for record in records) / len(
            records
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
        section_heading: str | None = None,
    ) -> list[SearchResult]:
        """Return chunks ranked by BM25 score."""
        clean_query = query.strip()

        if not clean_query:
            return []

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        if not self._records:
            return []

        query_terms = set(self._tokenize(clean_query))

        if not query_terms:
            return []

        scored_records: list[tuple[float, _LexicalRecord]] = []

        for record in self._records:
            if not self._matches_filters(
                record=record,
                document_ids=document_ids,
                section_heading=section_heading,
            ):
                continue

            score = sum(
                self._term_score(
                    term=term,
                    record=record,
                )
                for term in query_terms
            )

            if score > 0:
                scored_records.append(
                    (
                        score,
                        record,
                    )
                )

        ranked_records = sorted(
            scored_records,
            key=lambda item: (
                -item[0],
                item[1].chunk.chunk_index,
                item[1].chunk.id,
            ),
        )[:top_k]

        return [
            SearchResult(
                chunk=record.chunk,
                document=record.document,
                score=score,
            )
            for score, record in ranked_records
        ]

    def _term_score(
        self,
        term: str,
        record: _LexicalRecord,
    ) -> float:
        """Calculate one BM25 term contribution."""
        term_frequency = record.term_counts.get(
            term,
            0,
        )

        if term_frequency == 0:
            return 0.0

        document_frequency = self._document_frequency[term]
        document_count = len(self._records)

        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

        length_ratio = (
            record.length / self._average_document_length
            if self._average_document_length
            else 0.0
        )

        denominator = term_frequency + self.k1 * (1 - self.b + self.b * length_ratio)

        return inverse_document_frequency * term_frequency * (self.k1 + 1) / denominator

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        """Create case-insensitive alphanumeric tokens."""
        return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]

    @staticmethod
    def _search_text(
        document: Document,
        chunk: Chunk,
    ) -> str:
        """Combine retrieval text and useful metadata."""
        parts = [
            document.title,
        ]

        if chunk.section_heading:
            parts.append(chunk.section_heading)

        parts.append(chunk.text)

        return "\n".join(parts)

    @staticmethod
    def _matches_filters(
        record: _LexicalRecord,
        document_ids: set[str] | None,
        section_heading: str | None,
    ) -> bool:
        """Apply document and section filters."""
        if document_ids is not None and record.document.id not in document_ids:
            return False

        if section_heading is None:
            return True

        record_section = record.chunk.section_heading

        return (
            record_section is not None
            and record_section.casefold() == section_heading.casefold()
        )
