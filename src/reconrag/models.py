"""Library-independent data contracts used across ReconRAG."""

from datetime import datetime

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Metadata for an ingested research paper."""

    id: str
    title: str
    filename: str
    checksum: str
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    source_url: str | None = None
    ingested_at: datetime


class Chunk(BaseModel):
    """A citation-preserving unit of paper text."""

    id: str
    document_id: str
    text: str
    chunk_index: int = Field(ge=0)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_heading: str | None = None


class SearchResult(BaseModel):
    """A retrieved chunk and its similarity score."""

    chunk: Chunk
    document: Document
    score: float


class Answer(BaseModel):
    """A generated answer with the evidence used to produce it."""

    text: str
    evidence: list[SearchResult] = Field(default_factory=list)
    model_name: str
    latency_seconds: float = Field(ge=0)
