# ReconRAG Retrieval Evaluation

This directory contains labelled retrieval benchmarks and frozen
evaluation reports for ReconRAG.

## Closed-Uncertainty benchmark

The initial benchmark contains eight manually labelled questions from
one sparse-view CT uncertainty paper. The indexed paper contains 85
section-aware chunks.

Questions cover:

- analytic uncertainty propagation;
- uncertainty evaluation metrics;
- foreground ranking failure;
- shared reconstruction error;
- calibration limitations;
- the whole-volume versus in-object gap;
- learned error prediction;
- held-out human CT evaluation.

All questions use `top_k=5`. Relevance labels were audited after the
initial run to identify answer-bearing passages in the abstract,
introduction, main text, and appendices.

## Results

| Retriever | Recall@5 | MRR | Hit rate |
|---|---:|---:|---:|
| Dense semantic | 0.896 | 0.938 | 1.000 |
| BM25 lexical | 0.740 | 0.750 | 0.875 |
| Hybrid RRF | 0.781 | 1.000 | 1.000 |

The dense semantic retriever remains ReconRAG's default because it
retrieves the most complete evidence set. Equal-weight reciprocal rank
fusion improves the position of the first relevant result but reduces
overall evidence recall.

The BM25 and hybrid implementations remain available for future
experiments. They should be reconsidered after adding benchmarks from
more papers rather than tuning fusion weights against this small
development benchmark.

## Limitations

- The benchmark contains only eight questions from one paper.
- Document filters make this a within-paper passage-ranking benchmark;
  it does not evaluate cross-paper routing.
- The source PDF and local SQLite database are not committed.
- Chunk identifiers depend on the document and chunking configuration.
- The benchmark evaluates retrieval, not answer faithfulness or
  generation quality.

## Reproduce the comparison

```bash
uv run python -m reconrag.evaluation.cli run \
  benchmarks/closed_uncertainty.json \
  --retriever semantic \
  --output benchmarks/results/closed_uncertainty-semantic.json

uv run python -m reconrag.evaluation.cli run \
  benchmarks/closed_uncertainty.json \
  --retriever lexical \
  --output benchmarks/results/closed_uncertainty-lexical.json

uv run python -m reconrag.evaluation.cli run \
  benchmarks/closed_uncertainty.json \
  --retriever hybrid \
  --output benchmarks/results/closed_uncertainty-hybrid.json