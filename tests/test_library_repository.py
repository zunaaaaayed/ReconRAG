"""Tests for persistent paper-library storage."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from reconrag.models import Chunk, Document, ParsedPaper
from reconrag.retrieval import IndexedEmbedding, InMemoryVectorIndex
from reconrag.storage import LibraryRepository


class QueryEmbedder:
    """Return deterministic vectors for repository search tests."""

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        vectors = []

        for text in texts:
            if "geometry" in text.casefold():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


def _document() -> Document:
    return Document(
        id="paper-1",
        title="Sparse-View Reconstruction",
        filename="paper.pdf",
        checksum="checksum-1",
        ingested_at=datetime.now(UTC),
    )


def _paper() -> ParsedPaper:
    return ParsedPaper(
        title="Sparse-View Reconstruction",
        filename="paper.pdf",
        checksum="checksum-1",
        page_count=2,
        markdown="# Sparse-View Reconstruction",
    )


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="chunk-1",
            document_id="paper-1",
            text="Projection geometry affects reconstruction quality.",
            chunk_index=0,
            page_start=1,
            page_end=1,
            section_heading="Methods",
            token_count=6,
        ),
        Chunk(
            id="chunk-2",
            document_id="paper-1",
            text="Uncertainty was evaluated on held-out scans.",
            chunk_index=1,
            page_start=2,
            page_end=2,
            section_heading="Results",
            token_count=7,
        ),
    ]


def _embeddings(
    document: Document,
    chunks: list[Chunk],
) -> list[IndexedEmbedding]:
    return [
        IndexedEmbedding(
            document=document,
            chunk=chunks[0],
            vector=np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            ),
        ),
        IndexedEmbedding(
            document=document,
            chunk=chunks[1],
            vector=np.asarray(
                [0.0, 1.0],
                dtype=np.float32,
            ),
        ),
    ]


def test_repository_round_trip_restores_searchable_index(
    tmp_path: Path,
) -> None:
    repository = LibraryRepository(tmp_path / "reconrag.db")
    document = _document()
    paper = _paper()
    chunks = _chunks()

    repository.save_paper(
        document=document,
        paper=paper,
        embeddings=_embeddings(document, chunks),
        embedding_model="test-model",
    )

    snapshot = repository.load("test-model")

    assert snapshot.documents == {
        document.id: document,
    }
    assert snapshot.parsed_papers == {
        paper.checksum: paper,
    }
    assert snapshot.paper_chunks == {
        document.id: chunks,
    }
    assert not snapshot.requires_reindex

    index = InMemoryVectorIndex(QueryEmbedder())
    index.load(snapshot.embeddings)

    results = index.search("How does geometry affect reconstruction?")

    assert index.size == 2
    assert results[0].chunk.id == "chunk-1"


def test_saving_same_paper_replaces_existing_chunks(
    tmp_path: Path,
) -> None:
    repository = LibraryRepository(tmp_path / "reconrag.db")
    document = _document()
    paper = _paper()
    chunks = _chunks()

    repository.save_paper(
        document=document,
        paper=paper,
        embeddings=_embeddings(document, chunks),
        embedding_model="test-model",
    )

    updated_document = document.model_copy(update={"title": "Updated Paper"})
    updated_paper = paper.model_copy(update={"title": "Updated Paper"})

    repository.save_paper(
        document=updated_document,
        paper=updated_paper,
        embeddings=[
            IndexedEmbedding(
                document=updated_document,
                chunk=chunks[0],
                vector=np.asarray(
                    [1.0, 0.0],
                    dtype=np.float32,
                ),
            )
        ],
        embedding_model="test-model",
    )

    snapshot = repository.load("test-model")

    assert snapshot.documents[document.id].title == "Updated Paper"
    assert snapshot.parsed_papers[paper.checksum].title == ("Updated Paper")
    assert snapshot.paper_chunks[document.id] == [chunks[0]]
    assert len(snapshot.embeddings) == 1


def test_delete_document_cascades_to_chunks(
    tmp_path: Path,
) -> None:
    repository = LibraryRepository(tmp_path / "reconrag.db")
    document = _document()
    paper = _paper()
    chunks = _chunks()

    repository.save_paper(
        document=document,
        paper=paper,
        embeddings=_embeddings(document, chunks),
        embedding_model="test-model",
    )

    assert repository.delete_document(document.id)
    assert not repository.delete_document(document.id)

    snapshot = repository.load("test-model")

    assert snapshot.documents == {}
    assert snapshot.parsed_papers == {}
    assert snapshot.paper_chunks == {}
    assert snapshot.embeddings == []


def test_changed_embedding_model_requires_reindex(
    tmp_path: Path,
) -> None:
    repository = LibraryRepository(tmp_path / "reconrag.db")
    document = _document()
    paper = _paper()
    chunks = _chunks()

    repository.save_paper(
        document=document,
        paper=paper,
        embeddings=_embeddings(document, chunks),
        embedding_model="old-model",
    )

    snapshot = repository.load("new-model")

    assert snapshot.documents[document.id] == document
    assert snapshot.paper_chunks[document.id] == chunks
    assert snapshot.embeddings == []
    assert snapshot.stale_document_ids == {
        document.id,
    }
    assert snapshot.requires_reindex


def test_contains_checksum(
    tmp_path: Path,
) -> None:
    repository = LibraryRepository(tmp_path / "reconrag.db")
    document = _document()
    paper = _paper()
    chunks = _chunks()

    assert not repository.contains_checksum(document.checksum)

    repository.save_paper(
        document=document,
        paper=paper,
        embeddings=_embeddings(document, chunks),
        embedding_model="test-model",
    )

    assert repository.contains_checksum(document.checksum)
