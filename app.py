"""Streamlit entry point for ReconRAG."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from httpx import HTTPError
from ollama import ResponseError

from reconrag.config import get_settings
from reconrag.generation import (
    CitationValidator,
    OllamaAnswerGenerator,
)
from reconrag.ingestion.chunker import SectionAwareChunker
from reconrag.ingestion.parser import PdfParser, PdfParsingError
from reconrag.models import Chunk, Document, ParsedPaper
from reconrag.retrieval import (
    InMemoryVectorIndex,
    SentenceTransformerEmbedder,
)
from reconrag.services.ingest_service import store_pdf
from reconrag.storage import LibraryRepository


@st.cache_resource
def get_pdf_parser() -> PdfParser:
    """Reuse Docling across Streamlit reruns."""
    return PdfParser()


@st.cache_resource
def get_chunker(
    model_name: str,
    target_tokens: int,
    overlap_tokens: int,
) -> SectionAwareChunker:
    """Reuse the tokenizer across Streamlit reruns."""
    return SectionAwareChunker(
        model_name=model_name,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )


@st.cache_resource
def get_embedder(
    model_name: str,
) -> SentenceTransformerEmbedder:
    """Reuse the embedding model across Streamlit reruns."""
    return SentenceTransformerEmbedder(model_name)


@st.cache_resource
def get_answer_generator(
    model_name: str,
    host: str,
    max_tokens: int,
) -> OllamaAnswerGenerator:
    """Reuse the local answer generator across reruns."""
    return OllamaAnswerGenerator(
        model_name=model_name,
        host=host,
        max_tokens=max_tokens,
    )


@st.cache_resource
def get_library_repository(
    database_path: str,
) -> LibraryRepository:
    """Reuse the SQLite paper repository across reruns."""
    return LibraryRepository(Path(database_path))


def _page_label(
    page_numbers: list[int],
) -> str:
    """Format page numbers for display."""
    return ", ".join(str(page) for page in page_numbers) or "Unknown"


def _chunk_page_label(
    chunk: Chunk,
) -> str:
    """Format a chunk's page range."""
    if chunk.page_start is None:
        return "Unknown"

    if chunk.page_start == chunk.page_end:
        return str(chunk.page_start)

    return f"{chunk.page_start}–{chunk.page_end}"


def _document_from_paper(
    paper: ParsedPaper,
) -> Document:
    """Create retrieval metadata for a parsed paper."""
    return Document(
        id=paper.checksum,
        title=paper.title,
        filename=paper.filename,
        checksum=paper.checksum,
        ingested_at=datetime.now(UTC),
    )


def _initialize_library_state(
    database_path: str,
    embedding_model: str,
) -> None:
    """Restore documents, chunks, and embeddings from SQLite."""
    if st.session_state.get("library_initialized"):
        return

    repository = get_library_repository(database_path)

    snapshot = repository.load(
        expected_embedding_model=embedding_model,
    )

    vector_index = InMemoryVectorIndex(get_embedder(embedding_model))

    if snapshot.requires_reindex:
        vector_index.build(
            [
                (
                    document,
                    snapshot.paper_chunks.get(
                        document_id,
                        [],
                    ),
                )
                for (
                    document_id,
                    document,
                ) in snapshot.documents.items()
            ]
        )

        refreshed_embeddings = vector_index.snapshot()

        for document_id in snapshot.stale_document_ids:
            document = snapshot.documents[document_id]
            paper = snapshot.parsed_papers[document.checksum]

            document_embeddings = [
                indexed_embedding
                for indexed_embedding in refreshed_embeddings
                if indexed_embedding.document.id == document_id
            ]

            repository.save_paper(
                document=document,
                paper=paper,
                embeddings=document_embeddings,
                embedding_model=embedding_model,
            )
    else:
        vector_index.load(snapshot.embeddings)

    st.session_state["parsed_papers"] = snapshot.parsed_papers
    st.session_state["paper_chunks"] = snapshot.paper_chunks
    st.session_state["documents"] = snapshot.documents
    st.session_state["vector_index"] = vector_index
    st.session_state["library_initialized"] = True


def _delete_document_from_library(
    document_id: str,
    database_path: str,
    embedding_model: str,
) -> bool:
    """Delete a paper and reload the active library state."""
    repository = get_library_repository(database_path)

    if not repository.delete_document(document_id):
        return False

    snapshot = repository.load(expected_embedding_model=embedding_model)

    vector_index = InMemoryVectorIndex(get_embedder(embedding_model))
    vector_index.load(snapshot.embeddings)

    st.session_state["parsed_papers"] = snapshot.parsed_papers
    st.session_state["paper_chunks"] = snapshot.paper_chunks
    st.session_state["documents"] = snapshot.documents
    st.session_state["vector_index"] = vector_index

    return True


def _render_parsed_paper(
    paper: ParsedPaper,
    chunks: list[Chunk],
) -> bool:
    """Render metadata, chunks, and extracted text."""
    with st.expander(
        paper.title,
        expanded=True,
    ):
        first, second, third, fourth = st.columns(4)

        first.metric(
            "Pages",
            paper.page_count,
        )
        second.metric(
            "Text blocks",
            len(paper.blocks),
        )
        third.metric(
            "Chunks",
            len(chunks),
        )
        fourth.metric(
            "Checksum",
            paper.checksum[:8],
        )

        (
            chunks_tab,
            blocks_tab,
            markdown_tab,
        ) = st.tabs(
            [
                "Chunks",
                "Extracted blocks",
                "Markdown",
            ]
        )

        with chunks_tab:
            chunk_rows = [
                {
                    "Index": chunk.chunk_index,
                    "Pages": _chunk_page_label(chunk),
                    "Section": (chunk.section_heading or "—"),
                    "Tokens": chunk.token_count,
                    "Text": chunk.text,
                }
                for chunk in chunks
            ]

            st.dataframe(
                chunk_rows,
                hide_index=True,
                use_container_width=True,
            )

        with blocks_tab:
            preview_rows = [
                {
                    "Pages": _page_label(block.page_numbers),
                    "Type": block.label,
                    "Section": (block.section_heading or "—"),
                    "Text": block.text,
                }
                for block in paper.blocks[:100]
            ]

            st.dataframe(
                preview_rows,
                hide_index=True,
                use_container_width=True,
            )

            if len(paper.blocks) > 100:
                st.caption("Showing the first 100 extracted text blocks.")

        with markdown_tab:
            st.markdown(paper.markdown[:20_000])

            if len(paper.markdown) > 20_000:
                st.caption("Preview truncated to 20,000 characters.")

        st.divider()

        st.caption(
            "Removing this paper deletes its parsed "
            "content, chunks, and embeddings from the "
            "ReconRAG library. The original PDF is kept."
        )

        delete_confirmed = st.checkbox(
            "I understand and want to remove this paper.",
            key=f"confirm-delete-{paper.checksum}",
        )

        return st.button(
            "Remove paper from library",
            key=f"delete-paper-{paper.checksum}",
            disabled=not delete_confirmed,
        )


def render_app() -> None:
    """Render the ReconRAG application."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🔬",
        layout="wide",
    )

    try:
        with st.spinner("Loading the local research library..."):
            _initialize_library_state(
                database_path=str(settings.database_path),
                embedding_model=(settings.embedding_model),
            )
    except (
        OSError,
        RuntimeError,
        ValueError,
        sqlite3.Error,
    ) as exc:
        st.error("ReconRAG could not load the persistent paper library.")
        st.caption(str(exc))
        st.stop()

    st.title("🔬 ReconRAG")

    st.caption(
        "Evidence-grounded research assistant "
        "for sparse-view CT and "
        "3D reconstruction literature."
    )

    library_notice = st.session_state.pop(
        "library_notice",
        None,
    )

    if library_notice:
        st.success(library_notice)

    with st.sidebar:
        st.header("Project status")

        st.success("Milestone 8 · Citation integrity")

        parsed_papers = st.session_state.get(
            "parsed_papers",
            {},
        )

        st.metric(
            "Parsed papers",
            len(parsed_papers),
        )

        paper_chunks = st.session_state.get(
            "paper_chunks",
            {},
        )

        st.metric(
            "Generated chunks",
            sum(len(chunks) for chunks in paper_chunks.values()),
        )

        vector_index = st.session_state.get(
            "vector_index",
        )

        st.metric(
            "Indexed chunks",
            vector_index.size if vector_index else 0,
        )

        st.caption("All documents and embeddings remain on this machine.")

    library_tab, ask_tab = st.tabs(
        [
            "Paper library",
            "Ask ReconRAG",
        ]
    )

    with library_tab:
        st.subheader("Build the research collection")

        uploaded_files = st.file_uploader(
            "Upload research papers",
            type=["pdf"],
            accept_multiple_files=True,
        )

        parse_clicked = st.button(
            "Parse, chunk, and index papers",
            disabled=not uploaded_files,
            type="primary",
        )

        if parse_clicked:
            parser = get_pdf_parser()

            chunker = get_chunker(
                settings.embedding_model,
                settings.chunk_target_tokens,
                settings.chunk_overlap_tokens,
            )

            repository = get_library_repository(str(settings.database_path))

            new_document_ids: set[str] = set()

            progress = st.progress(
                0,
                text="Preparing ingestion...",
            )

            for index, uploaded_file in enumerate(
                uploaded_files,
                start=1,
            ):
                progress.progress(
                    (index - 1) / len(uploaded_files),
                    text=(f"Parsing and chunking {uploaded_file.name}..."),
                )

                try:
                    stored = store_pdf(
                        content=uploaded_file.getvalue(),
                        original_filename=(uploaded_file.name),
                        papers_dir=settings.papers_dir,
                    )

                    if repository.contains_checksum(stored.checksum):
                        st.info(f"{uploaded_file.name} is already in the library.")
                        continue

                    paper = parser.parse(stored.path)
                    chunks = chunker.chunk(paper)
                    document = _document_from_paper(paper)

                    st.session_state["parsed_papers"][paper.checksum] = paper

                    st.session_state["paper_chunks"][document.id] = chunks

                    st.session_state["documents"][document.id] = document

                    new_document_ids.add(document.id)

                except (
                    OSError,
                    ValueError,
                    PdfParsingError,
                    sqlite3.Error,
                ) as exc:
                    st.error(f"Could not process {uploaded_file.name}: {exc}")

            if new_document_ids:
                progress.progress(
                    0.95,
                    text="Creating local embeddings...",
                )

                documents = st.session_state["documents"]
                chunks_by_document = st.session_state["paper_chunks"]

                try:
                    vector_index = InMemoryVectorIndex(
                        get_embedder(settings.embedding_model)
                    )

                    vector_index.build(
                        [
                            (
                                document,
                                chunks_by_document.get(
                                    document_id,
                                    [],
                                ),
                            )
                            for (
                                document_id,
                                document,
                            ) in documents.items()
                        ]
                    )

                    indexed_embeddings = vector_index.snapshot()

                    for document_id in new_document_ids:
                        document = documents[document_id]
                        paper = st.session_state["parsed_papers"][document.checksum]

                        document_embeddings = [
                            indexed_embedding
                            for indexed_embedding in indexed_embeddings
                            if (indexed_embedding.document.id == document_id)
                        ]

                        repository.save_paper(
                            document=document,
                            paper=paper,
                            embeddings=(document_embeddings),
                            embedding_model=(settings.embedding_model),
                        )

                    st.session_state["vector_index"] = vector_index

                    st.success("New papers were saved to the persistent library.")

                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    sqlite3.Error,
                ) as exc:
                    st.session_state.pop(
                        "vector_index",
                        None,
                    )
                    st.error(f"Could not create and save the semantic index: {exc}")

                progress.progress(
                    1.0,
                    text=("Parsing, chunking, indexing, and storage complete."),
                )
            else:
                progress.progress(
                    1.0,
                    text="No new papers needed processing.",
                )

        parsed_papers = st.session_state.get(
            "parsed_papers",
            {},
        )

        if parsed_papers:
            delete_document_id: str | None = None
            delete_document_title: str | None = None

            for paper in list(parsed_papers.values()):
                chunks = st.session_state.get(
                    "paper_chunks",
                    {},
                ).get(
                    paper.checksum,
                    [],
                )

                delete_requested = _render_parsed_paper(
                    paper,
                    chunks,
                )

                if delete_requested:
                    delete_document_id = paper.checksum
                    delete_document_title = paper.title

            if delete_document_id is not None:
                try:
                    deleted = _delete_document_from_library(
                        document_id=(delete_document_id),
                        database_path=str(settings.database_path),
                        embedding_model=(settings.embedding_model),
                    )
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    sqlite3.Error,
                ) as exc:
                    st.error("ReconRAG could not remove the paper from the library.")
                    st.caption(str(exc))
                else:
                    if deleted:
                        st.session_state["library_notice"] = (
                            f"{delete_document_title} was removed from the library."
                        )
                        st.rerun()
                    else:
                        st.warning("The paper was no longer present in the library.")
        else:
            st.info("Upload a research paper to inspect its extracted structure.")

    with ask_tab:
        st.subheader("Ask the research collection")

        vector_index = st.session_state.get(
            "vector_index",
        )
        documents = st.session_state.get(
            "documents",
            {},
        )
        chunks_by_document = st.session_state.get(
            "paper_chunks",
            {},
        )

        if vector_index is None or vector_index.size == 0:
            st.info("Parse and index at least one paper before asking a question.")
        else:
            query = st.text_area(
                "Research question",
                placeholder=(
                    "How are 3D Gaussians initialized and optimized for sparse-view CT?"
                ),
                height=100,
            )

            document_options: list[str | None] = [
                None,
                *documents.keys(),
            ]

            selected_document = st.selectbox(
                "Paper filter",
                options=document_options,
                format_func=lambda document_id: (
                    "All papers"
                    if document_id is None
                    else documents[document_id].title
                ),
            )

            if selected_document is None:
                available_chunks = [
                    chunk for chunks in chunks_by_document.values() for chunk in chunks
                ]
            else:
                available_chunks = chunks_by_document.get(
                    selected_document,
                    [],
                )

            section_options: list[str | None] = [
                None,
                *sorted(
                    {
                        chunk.section_heading
                        for chunk in available_chunks
                        if chunk.section_heading
                    }
                ),
            ]

            selected_section = st.selectbox(
                "Section filter",
                options=section_options,
                format_func=lambda section: (
                    "All sections" if section is None else section
                ),
            )

            top_k = st.slider(
                "Number of evidence passages",
                min_value=1,
                max_value=10,
                value=min(
                    settings.retrieval_top_k,
                    10,
                ),
            )

            generate_clicked = st.button(
                "Generate grounded answer",
                type="primary",
                disabled=not query.strip(),
            )

            if generate_clicked:
                results = vector_index.search(
                    query=query,
                    top_k=top_k,
                    document_ids=(
                        None if selected_document is None else {selected_document}
                    ),
                    section_heading=(selected_section),
                )

                if not results:
                    st.warning("No evidence passages matched the current filters.")
                else:
                    generator = get_answer_generator(
                        model_name=(settings.generation_model),
                        host=(settings.ollama_host),
                        max_tokens=(settings.generation_max_tokens),
                    )

                    try:
                        with st.spinner(
                            "Reading the evidence and generating an answer..."
                        ):
                            answer = generator.generate(
                                question=query,
                                evidence=results,
                            )

                    except (
                        HTTPError,
                        ResponseError,
                        OSError,
                        RuntimeError,
                        ValueError,
                    ) as exc:
                        st.error(
                            "ReconRAG could not generate "
                            "an answer. Make sure Ollama "
                            "is open and "
                            f"{settings.generation_model} "
                            "is installed."
                        )
                        st.caption(str(exc))

                    else:
                        st.markdown("### Answer")
                        st.markdown(answer.text)

                        citation_validation = CitationValidator().validate(
                            text=answer.text,
                            evidence_count=len(answer.evidence),
                        )

                        if citation_validation.is_valid:
                            st.success(
                                "Citation syntax and evidence references passed."
                            )
                        else:
                            citation_issues: list[str] = []

                            if not (citation_validation.has_citations):
                                citation_issues.append(
                                    "the answer contains no citations"
                                )

                            if citation_validation.invalid_citations:
                                invalid_numbers = ", ".join(
                                    str(number)
                                    for number in (
                                        citation_validation.invalid_citations
                                    )
                                )
                                citation_issues.append(
                                    f"invalid evidence numbers: {invalid_numbers}"
                                )

                            if citation_validation.malformed_citations:
                                malformed = ", ".join(
                                    citation_validation.malformed_citations
                                )
                                citation_issues.append(
                                    f"malformed citations: {malformed}"
                                )

                            st.warning(
                                "Citation validation "
                                "requires manual review: "
                                + "; ".join(citation_issues)
                                + "."
                            )

                        st.caption(
                            "Citation validation checks "
                            "citation syntax and numbering, "
                            "not whether a passage logically "
                            "supports every claim."
                        )

                        (
                            model_column,
                            latency_column,
                            evidence_column,
                            citation_column,
                        ) = st.columns(4)

                        model_column.metric(
                            "Model",
                            answer.model_name,
                        )
                        latency_column.metric(
                            "Generation time", f"{answer.latency_seconds:.1f} s"
                        )
                        evidence_column.metric(
                            "Evidence passages",
                            len(answer.evidence),
                        )
                        citation_column.metric(
                            "Evidence cited",
                            (
                                f"{citation_validation.cited_evidence_count}"
                                f"/{citation_validation.evidence_count}"
                            ),
                        )

                        st.markdown("### Retrieved evidence")

                        for (
                            citation_number,
                            result,
                        ) in enumerate(
                            answer.evidence,
                            start=1,
                        ):
                            chunk = result.chunk
                            page_label = _chunk_page_label(chunk)

                            page_prefix = (
                                "Page"
                                if (
                                    chunk.page_start is not None
                                    and chunk.page_start == chunk.page_end
                                )
                                else "Pages"
                            )

                            section_label = chunk.section_heading or "Unknown section"

                            expander_label = (
                                f"[{citation_number}] "
                                f"{result.document.title}"
                                f" · {page_prefix} "
                                f"{page_label}"
                            )

                            with st.expander(expander_label):
                                st.caption(
                                    f"{section_label} · Similarity {result.score:.3f}"
                                )
                                st.write(chunk.text)


if __name__ == "__main__":
    render_app()
