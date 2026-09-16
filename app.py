"""Streamlit entry point for ReconRAG."""

import streamlit as st

from reconrag.config import get_settings


def render_app() -> None:
    """Render the Milestone 0 application shell."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🔬",
        layout="wide",
    )

    st.title("ReconRAG")
    st.caption(
        "Evidence-grounded research assistant for sparse-view CT and "
        "3D reconstruction literature."
    )

    with st.sidebar:
        st.header("Project status")
        st.success("Milestone 0 · Foundation")
        st.metric("Indexed papers", 0)
        st.caption("All documents and embeddings remain on this machine.")

    library_tab, ask_tab = st.tabs(["Paper library", "Ask ReconRAG"])

    with library_tab:
        st.subheader("Build the research collection")
        st.file_uploader(
            "Upload research papers",
            type=["pdf"],
            accept_multiple_files=True,
            disabled=True,
            help="PDF ingestion will be enabled in Milestone 1.",
        )
        st.info("PDF ingestion is the next milestone.")

    with ask_tab:
        st.subheader("Ask across the evidence")
        st.text_input(
            "Research question",
            placeholder="Which methods address limited-angle CT reconstruction?",
            disabled=True,
        )
        st.info("Evidence-backed question answering will be enabled after indexing.")

    st.divider()
    st.caption(
        "Research and educational use only. ReconRAG does not provide medical advice."
    )


if __name__ == "__main__":
    render_app()
