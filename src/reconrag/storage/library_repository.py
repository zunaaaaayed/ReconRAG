"""SQLite persistence for papers, chunks, and embeddings."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reconrag.models import Chunk, Document, ParsedPaper
from reconrag.retrieval import IndexedEmbedding


@dataclass(frozen=True)
class LibrarySnapshot:
    """All persistent state required to restore the paper library."""

    documents: dict[str, Document]
    parsed_papers: dict[str, ParsedPaper]
    paper_chunks: dict[str, list[Chunk]]
    embeddings: list[IndexedEmbedding]
    stale_document_ids: set[str]

    @property
    def requires_reindex(self) -> bool:
        """Return whether any stored embeddings use another model."""
        return bool(self.stale_document_ids)


class LibraryRepository:
    """Store and restore the ReconRAG paper library using SQLite."""

    def __init__(
        self,
        database_path: Path,
    ) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize_schema()

    def save_paper(
        self,
        document: Document,
        paper: ParsedPaper,
        embeddings: list[IndexedEmbedding],
        embedding_model: str,
    ) -> None:
        """Atomically store one paper and its indexed chunks."""
        clean_model = embedding_model.strip()

        if not clean_model:
            raise ValueError("embedding_model cannot be empty.")

        if document.checksum != paper.checksum:
            raise ValueError("Document and parsed paper checksums do not match.")

        chunk_ids: set[str] = set()
        dimensions: set[int] = set()
        serialized_embeddings: list[tuple[Chunk, bytes, int]] = []

        for indexed_embedding in embeddings:
            if indexed_embedding.document.id != document.id:
                raise ValueError(
                    "Every embedding must belong to the supplied document."
                )

            chunk = indexed_embedding.chunk

            if chunk.document_id != document.id:
                raise ValueError("Every chunk must belong to the supplied document.")

            if chunk.id in chunk_ids:
                raise ValueError("Duplicate chunk IDs cannot be stored.")

            vector = np.asarray(
                indexed_embedding.vector,
                dtype=np.float32,
            )

            if vector.ndim != 1:
                raise ValueError("Stored embeddings must be one-dimensional.")

            if vector.size == 0:
                raise ValueError("Stored embeddings cannot be empty.")

            chunk_ids.add(chunk.id)
            dimensions.add(int(vector.size))

            serialized_embeddings.append(
                (
                    chunk,
                    vector.tobytes(),
                    int(vector.size),
                )
            )

        if len(dimensions) > 1:
            raise ValueError("All embeddings for a paper must have the same dimension.")

        connection = self._connect()

        try:
            with connection:
                connection.execute(
                    """
                    DELETE FROM papers
                    WHERE document_id = ? OR checksum = ?
                    """,
                    (
                        document.id,
                        document.checksum,
                    ),
                )

                connection.execute(
                    """
                    INSERT INTO papers (
                        document_id,
                        checksum,
                        document_json,
                        parsed_paper_json,
                        embedding_model
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.checksum,
                        document.model_dump_json(),
                        paper.model_dump_json(),
                        clean_model,
                    ),
                )

                connection.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id,
                        document_id,
                        chunk_index,
                        chunk_json,
                        embedding,
                        embedding_dimension
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.id,
                            document.id,
                            chunk.chunk_index,
                            chunk.model_dump_json(),
                            sqlite3.Binary(vector_bytes),
                            dimension,
                        )
                        for (
                            chunk,
                            vector_bytes,
                            dimension,
                        ) in serialized_embeddings
                    ],
                )
        finally:
            connection.close()

    def load(
        self,
        expected_embedding_model: str,
    ) -> LibrarySnapshot:
        """Load the library and compatible stored embeddings."""
        documents: dict[str, Document] = {}
        parsed_papers: dict[str, ParsedPaper] = {}
        paper_chunks: dict[str, list[Chunk]] = {}
        embeddings: list[IndexedEmbedding] = []
        stale_document_ids: set[str] = set()

        connection = self._connect()

        try:
            paper_rows = connection.execute(
                """
                SELECT
                    document_id,
                    document_json,
                    parsed_paper_json,
                    embedding_model
                FROM papers
                ORDER BY document_id
                """
            ).fetchall()

            for row in paper_rows:
                document = Document.model_validate_json(row["document_json"])
                paper = ParsedPaper.model_validate_json(row["parsed_paper_json"])

                documents[document.id] = document
                parsed_papers[paper.checksum] = paper
                paper_chunks[document.id] = []

                if row["embedding_model"] != expected_embedding_model:
                    stale_document_ids.add(document.id)

            chunk_rows = connection.execute(
                """
                SELECT
                    chunks.document_id,
                    chunks.chunk_json,
                    chunks.embedding,
                    chunks.embedding_dimension,
                    papers.embedding_model
                FROM chunks
                JOIN papers
                    ON papers.document_id = chunks.document_id
                ORDER BY
                    chunks.document_id,
                    chunks.chunk_index
                """
            ).fetchall()

            for row in chunk_rows:
                document_id = row["document_id"]
                chunk = Chunk.model_validate_json(row["chunk_json"])

                paper_chunks[document_id].append(chunk)

                if row["embedding_model"] != expected_embedding_model:
                    continue

                vector = np.frombuffer(
                    row["embedding"],
                    dtype=np.float32,
                ).copy()

                expected_dimension = row["embedding_dimension"]

                if vector.size != expected_dimension:
                    raise ValueError("A stored embedding has an invalid dimension.")

                embeddings.append(
                    IndexedEmbedding(
                        document=documents[document_id],
                        chunk=chunk,
                        vector=vector,
                    )
                )
        finally:
            connection.close()

        return LibrarySnapshot(
            documents=documents,
            parsed_papers=parsed_papers,
            paper_chunks=paper_chunks,
            embeddings=embeddings,
            stale_document_ids=stale_document_ids,
        )

    def contains_checksum(
        self,
        checksum: str,
    ) -> bool:
        """Return whether a paper checksum is already stored."""
        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT 1
                FROM papers
                WHERE checksum = ?
                LIMIT 1
                """,
                (checksum,),
            ).fetchone()
        finally:
            connection.close()

        return row is not None

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """Delete a document and its chunks."""
        connection = self._connect()

        try:
            with connection:
                cursor = connection.execute(
                    """
                    DELETE FROM papers
                    WHERE document_id = ?
                    """,
                    (document_id,),
                )
        finally:
            connection.close()

        return cursor.rowcount > 0

    def _initialize_schema(self) -> None:
        """Create repository tables when they do not exist."""
        connection = self._connect()

        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    document_id TEXT PRIMARY KEY,
                    checksum TEXT NOT NULL UNIQUE,
                    document_json TEXT NOT NULL,
                    parsed_paper_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    stored_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    FOREIGN KEY (document_id)
                        REFERENCES papers(document_id)
                        ON DELETE CASCADE,
                    UNIQUE (document_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS
                    idx_chunks_document_id
                ON chunks(document_id);
                """
            )
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        """Open a configured SQLite connection."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        return connection
