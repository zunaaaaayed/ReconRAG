"""Streamlit entry point for ReconRAG."""

import streamlit as st

from reconrag.config import get_settings
from reconrag.ingestion.parser import PdfParser, PdfParsingError
from reconrag.models import ParsedPaper
from reconrag.services.ingest_service import store_pdf


@st.cache_resource
def get_pdf_parser() -> PdfParser:
    """Reuse Docling's converter and loaded models across reruns."""
    return PdfParser()


def _page_label(page_numbers: list[int]) -> str:
    """Format a list of page numbers for display."""
    return ", ".join(str(page) for page in page_numbers) or "Unknown"


def _render_parsed_paper(paper: ParsedPaper) -> None:
    """Render paper metadata and a structured text preview."""
    with st.expander(paper.title, expanded=True):
        first, second, third = st.columns(3)

        first.metric("Pages", paper.page_count)
        second.metric("Text blocks", len(paper.blocks))
        third.metric("Checksum", paper.checksum[:8])

        preview_rows = [
            {
                "Pages": _page_label(block.page_numbers),
                "Type": block.label,
                "Section": block.section_heading or "—",
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

        with st.expander("Markdown preview"):
            st.markdown(paper.markdown[:20_000])

            if len(paper.markdown) > 20_000:
                st.caption("Preview truncated to 20,000 characters.")


def render_app() -> None:
    """Render the Milestone 1 application."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 ReconRAG")
    st.caption(
        "Evidence-grounded research assistant for sparse-view CT "
        "and 3D reconstruction literature."
    )

    with st.sidebar:
        st.header("Project status")
        st.success("Milestone 1 · PDF ingestion")

        parsed_papers = st.session_state.get(
            "parsed_papers",
            {},
        )

        st.metric(
            "Parsed papers",
            len(parsed_papers),
        )

        st.caption("All documents and embeddings remain on this machine.")

    library_tab, ask_tab = st.tabs(["Paper library", "Ask ReconRAG"])

    with library_tab:
        st.subheader("Build the research collection")

        uploaded_files = st.file_uploader(
            "Upload research papers",
            type=["pdf"],
            accept_multiple_files=True,
        )

        parse_clicked = st.button(
            "Parse papers",
            disabled=not uploaded_files,
            type="primary",
        )

        if parse_clicked:
            st.session_state.setdefault(
                "parsed_papers",
                {},
            )

            parser = get_pdf_parser()

            progress = st.progress(
                0,
                text="Preparing PDF ingestion...",
            )

            for index, uploaded_file in enumerate(
                uploaded_files,
                start=1,
            ):
                progress.progress(
                    (index - 1) / len(uploaded_files),
                    text=f"Parsing {uploaded_file.name}...",
                )

                try:
                    stored = store_pdf(
                        content=uploaded_file.getvalue(),
                        original_filename=uploaded_file.name,
                        papers_dir=settings.papers_dir,
                    )

                    paper = parser.parse(stored.path)

                    st.session_state["parsed_papers"][paper.checksum] = paper

                except (
                    OSError,
                    ValueError,
                    PdfParsingError,
                ) as exc:
                    st.error(str(exc))

            progress.progress(
                1.0,
                text="Parsing complete.",
            )

        parsed_papers = st.session_state.get(
            "parsed_papers",
            {},
        )

        if parsed_papers:
            for paper in parsed_papers.values():
                _render_parsed_paper(paper)
        else:
            st.info("Upload one or more PDFs to inspect their structured contents.")

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
