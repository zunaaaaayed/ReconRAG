"""Streamlit entry point for ReconRAG."""

import streamlit as st

from reconrag.config import get_settings
from reconrag.ingestion.chunker import (
    SectionAwareChunker,
)
from reconrag.ingestion.parser import (
    PdfParser,
    PdfParsingError,
)
from reconrag.models import Chunk, ParsedPaper
from reconrag.services.ingest_service import store_pdf


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


def _render_parsed_paper(
    paper: ParsedPaper,
    chunks: list[Chunk],
) -> None:
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


def render_app() -> None:
    """Render the Milestone 2 application."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 ReconRAG")

    st.caption(
        "Evidence-grounded research assistant "
        "for sparse-view CT and "
        "3D reconstruction literature."
    )

    with st.sidebar:
        st.header("Project status")

        st.success("Milestone 2 · Section-aware chunking")

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
            "Parse and chunk papers",
            disabled=not uploaded_files,
            type="primary",
        )

        if parse_clicked:
            st.session_state.setdefault(
                "parsed_papers",
                {},
            )
            st.session_state.setdefault(
                "paper_chunks",
                {},
            )

            parser = get_pdf_parser()

            chunker = get_chunker(
                settings.embedding_model,
                settings.chunk_target_tokens,
                settings.chunk_overlap_tokens,
            )

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
                        content=(uploaded_file.getvalue()),
                        original_filename=(uploaded_file.name),
                        papers_dir=(settings.papers_dir),
                    )

                    paper = parser.parse(stored.path)
                    chunks = chunker.chunk(paper)

                    st.session_state["parsed_papers"][paper.checksum] = paper

                    st.session_state["paper_chunks"][paper.checksum] = chunks

                except (
                    OSError,
                    ValueError,
                    PdfParsingError,
                ) as exc:
                    st.error(str(exc))

            progress.progress(
                1.0,
                text="Parsing and chunking complete.",
            )

        parsed_papers = st.session_state.get(
            "parsed_papers",
            {},
        )

        if parsed_papers:
            for paper in parsed_papers.values():
                chunks = st.session_state.get(
                    "paper_chunks",
                    {},
                ).get(
                    paper.checksum,
                    [],
                )

                _render_parsed_paper(
                    paper,
                    chunks,
                )
        else:
            st.info("Upload one or more PDFs to inspect their chunks.")

    with ask_tab:
        st.subheader("Ask across the evidence")

        st.text_input(
            "Research question",
            placeholder=("Which methods address limited-angle CT reconstruction?"),
            disabled=True,
        )

        st.info("Evidence-backed question answering will be enabled after indexing.")

    st.divider()

    st.caption(
        "Research and educational use only. ReconRAG does not provide medical advice."
    )


if __name__ == "__main__":
    render_app()
