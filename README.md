# ReconRAG

ReconRAG is an evidence-grounded research assistant for sparse-view CT, limited-angle reconstruction, Gaussian representations, and related 3D medical-imaging literature.

The project is being developed as a transparent, locally runnable RAG system. Its answers will preserve paper and page provenance and expose the passages used as evidence.

> [!IMPORTANT]
> ReconRAG is a research and educational tool. It does not provide medical advice or clinical decision support.

## Current status

**Milestone 0 — project foundation**

- [x] Python project and dependency management
- [x] Streamlit application shell
- [x] Environment-based configuration
- [x] Core data contracts
- [x] Smoke tests and linting
- [ ] Structured PDF ingestion
- [ ] Section-aware chunking
- [ ] Local vector indexing
- [ ] Evidence-backed question answering
- [ ] Retrieval evaluation

## Technology

- Python 3.13 and `uv`
- Streamlit
- Docling
- Sentence Transformers
- Qdrant local mode
- Pydantic Settings
- pytest and Ruff

## Local development

Install the locked dependencies:

```bash
uv sync
```

Start the application:

```bash
uv run streamlit run app.py
```

Run the quality checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Local papers

Place research PDFs in the untracked `papers/` directory. Do not commit papers, vector indexes, databases, environment files, or patient data.

## Planned pipeline

```text
PDF -> structured parsing -> citation-aware chunks -> embeddings
    -> vector retrieval -> evidence-grounded answer with page citations
```

## License

A license will be selected before the first public release.
