"""Command-line tools for building and running benchmarks."""

import argparse
import json
from pathlib import Path
from typing import Any

from reconrag.config import get_settings
from reconrag.evaluation.benchmark import (
    load_benchmark,
)
from reconrag.evaluation.evaluator import (
    RetrievalEvaluator,
    RetrievalReport,
)
from reconrag.retrieval import (
    BM25Index,
    HybridSearchIndex,
    InMemoryVectorIndex,
    RankedSearchIndex,
    SentenceTransformerEmbedder,
)
from reconrag.storage import LibraryRepository


def _build_parser() -> argparse.ArgumentParser:
    """Create the ReconRAG evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description=("Inspect and evaluate the persistent ReconRAG retrieval library.")
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    chunks_parser = subparsers.add_parser(
        "chunks",
        help=("List stored chunks for benchmark labelling."),
    )
    chunks_parser.add_argument(
        "--database",
        type=Path,
        help="SQLite library path.",
    )
    chunks_parser.add_argument(
        "--embedding-model",
        help="Expected embedding model.",
    )
    chunks_parser.add_argument(
        "--document-id",
        help="Only list chunks from this document.",
    )
    chunks_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON catalogue output path.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a retrieval benchmark.",
    )
    run_parser.add_argument(
        "benchmark",
        type=Path,
        help="Path to the benchmark JSON file.",
    )
    run_parser.add_argument(
        "--database",
        type=Path,
        help="SQLite library path.",
    )
    run_parser.add_argument(
        "--embedding-model",
        help="Expected embedding model.",
    )
    run_parser.add_argument(
        "--retriever",
        choices=[
            "semantic",
            "lexical",
            "hybrid",
        ],
        default="semantic",
        help=("Retrieval algorithm to evaluate (default: semantic)."),
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report output path.",
    )

    return parser


def _write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write a formatted JSON artifact."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _page_label(
    page_start: int | None,
    page_end: int | None,
) -> str:
    """Format a stored chunk page range."""
    if page_start is None:
        return "Unknown"

    if page_start == page_end:
        return str(page_start)

    return f"{page_start}-{page_end}"


def _chunk_catalogue(
    repository: LibraryRepository,
    embedding_model: str,
    document_id: str | None,
) -> list[dict[str, Any]]:
    """Build serializable metadata for stored chunks."""
    snapshot = repository.load(expected_embedding_model=embedding_model)

    catalogue: list[dict[str, Any]] = []

    for current_document_id, document in snapshot.documents.items():
        if document_id is not None and current_document_id != document_id:
            continue

        for chunk in snapshot.paper_chunks.get(
            current_document_id,
            [],
        ):
            catalogue.append(
                {
                    "document_id": current_document_id,
                    "document_title": document.title,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "pages": _page_label(
                        chunk.page_start,
                        chunk.page_end,
                    ),
                    "section": (chunk.section_heading or "Unknown"),
                    "text": chunk.text,
                }
            )

    return catalogue


def _print_chunk_catalogue(
    catalogue: list[dict[str, Any]],
) -> None:
    """Print a compact chunk catalogue to the terminal."""
    print(
        "\t".join(
            [
                "DOCUMENT",
                "INDEX",
                "CHUNK ID",
                "PAGES",
                "SECTION",
                "PREVIEW",
            ]
        )
    )

    for item in catalogue:
        preview = " ".join(str(item["text"]).split())[:100]

        print(
            "\t".join(
                [
                    str(item["document_id"]),
                    str(item["chunk_index"]),
                    str(item["chunk_id"]),
                    str(item["pages"]),
                    str(item["section"]),
                    preview,
                ]
            )
        )


def _report_payload(
    benchmark_name: str,
    retriever_metadata: dict[str, Any],
    report: RetrievalReport,
) -> dict[str, Any]:
    """Convert an evaluation report to JSON data."""
    return {
        "benchmark": benchmark_name,
        "retriever": retriever_metadata,
        "case_count": report.case_count,
        "mean_recall_at_k": (report.mean_recall_at_k),
        "mean_reciprocal_rank": (report.mean_reciprocal_rank),
        "hit_rate": report.hit_rate,
        "results": [
            {
                "case_id": result.case_id,
                "question": result.question,
                "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
                "relevant_chunk_ids": sorted(result.relevant_chunk_ids),
                "recall_at_k": (result.recall_at_k),
                "reciprocal_rank": (result.reciprocal_rank),
                "hit": result.hit,
            }
            for result in report.results
        ],
    }


def _print_report(
    benchmark_name: str,
    report: RetrievalReport,
) -> None:
    """Print benchmark results and aggregate metrics."""
    print(f"Benchmark: {benchmark_name}")
    print(f"Cases: {report.case_count}")
    print(f"Mean Recall@k: {report.mean_recall_at_k:.3f}")
    print(f"Mean Reciprocal Rank: {report.mean_reciprocal_rank:.3f}")
    print(f"Hit rate: {report.hit_rate:.3f}")
    print()

    for result in report.results:
        status = "HIT" if result.hit else "MISS"

        print(
            f"[{status}] {result.case_id} "
            f"recall={result.recall_at_k:.3f} "
            f"rr={result.reciprocal_rank:.3f}"
        )
        print(f"  {result.question}")
        print("  Retrieved: " + ", ".join(result.retrieved_chunk_ids))
        print("  Relevant: " + ", ".join(sorted(result.relevant_chunk_ids)))


def _run_chunks_command(
    args: argparse.Namespace,
) -> None:
    """Export or display stored chunks."""
    settings = get_settings()
    database_path = (
        args.database if args.database is not None else settings.database_path
    )
    embedding_model = (
        args.embedding_model
        if args.embedding_model is not None
        else settings.embedding_model
    )

    repository = LibraryRepository(database_path)
    catalogue = _chunk_catalogue(
        repository=repository,
        embedding_model=embedding_model,
        document_id=args.document_id,
    )

    if args.output is not None:
        _write_json(
            args.output,
            catalogue,
        )
        print(f"Wrote {len(catalogue)} chunks to {args.output}.")
    else:
        _print_chunk_catalogue(catalogue)


def _run_benchmark_command(
    args: argparse.Namespace,
) -> None:
    """Evaluate a selected persistent retrieval index."""
    settings = get_settings()
    database_path = (
        args.database if args.database is not None else settings.database_path
    )
    embedding_model = (
        args.embedding_model
        if args.embedding_model is not None
        else settings.embedding_model
    )

    repository = LibraryRepository(database_path)
    snapshot = repository.load(expected_embedding_model=embedding_model)

    if not snapshot.documents:
        raise SystemExit("The persistent paper library is empty.")

    entries = [
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

    if not any(chunks for _, chunks in entries):
        raise SystemExit("The library contains no stored chunks.")

    requires_semantic_index = args.retriever in {
        "semantic",
        "hybrid",
    }

    if requires_semantic_index and snapshot.requires_reindex:
        raise SystemExit(
            "Stored embeddings use another model. "
            "Start the Streamlit app once to reindex "
            "the library before evaluation."
        )

    if requires_semantic_index and not snapshot.embeddings:
        raise SystemExit("The library contains no stored embeddings.")

    search_index: RankedSearchIndex
    retriever_metadata: dict[str, Any]

    if args.retriever == "semantic":
        semantic_index = InMemoryVectorIndex(
            SentenceTransformerEmbedder(embedding_model)
        )
        semantic_index.load(snapshot.embeddings)

        search_index = semantic_index
        retriever_metadata = {
            "type": "dense_cosine",
            "embedding_model": (embedding_model),
        }

    elif args.retriever == "lexical":
        lexical_index = BM25Index()
        lexical_index.build(entries)

        search_index = lexical_index
        retriever_metadata = {
            "type": "bm25",
            "k1": lexical_index.k1,
            "b": lexical_index.b,
        }

    else:
        semantic_index = InMemoryVectorIndex(
            SentenceTransformerEmbedder(embedding_model)
        )
        semantic_index.load(snapshot.embeddings)

        lexical_index = BM25Index()
        lexical_index.build(entries)

        hybrid_index = HybridSearchIndex(
            semantic_index=semantic_index,
            lexical_index=lexical_index,
        )

        search_index = hybrid_index
        retriever_metadata = {
            "type": "hybrid_rrf",
            "embedding_model": (embedding_model),
            "bm25_k1": lexical_index.k1,
            "bm25_b": lexical_index.b,
            "rrf_k": hybrid_index.rrf_k,
            "candidate_multiplier": (hybrid_index.candidate_multiplier),
            "semantic_weight": (hybrid_index.semantic_weight),
            "lexical_weight": (hybrid_index.lexical_weight),
        }

    benchmark = load_benchmark(args.benchmark)

    evaluator = RetrievalEvaluator(search_index)
    report = evaluator.evaluate(benchmark.to_retrieval_cases())

    print(f"Retriever: {args.retriever}")
    _print_report(
        benchmark.name,
        report,
    )

    if args.output is not None:
        _write_json(
            args.output,
            _report_payload(
                benchmark.name,
                retriever_metadata,
                report,
            ),
        )
        print()
        print(f"Wrote evaluation report to {args.output}.")


def main() -> None:
    """Run the selected evaluation command."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "chunks":
        _run_chunks_command(args)
    elif args.command == "run":
        _run_benchmark_command(args)


if __name__ == "__main__":
    main()
