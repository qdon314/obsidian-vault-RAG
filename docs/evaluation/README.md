# Evaluation System

This directory contains comprehensive documentation for the RAG evaluation framework.

## Overview

The evaluation system provides tools for:

1. **Query Curation** - Creating and managing ground-truth evaluation datasets
2. **Retrieval Evaluation** - Measuring how well the retriever finds relevant chunks
3. **End-to-End Evaluation** - Testing the full pipeline from query to generated answer
4. **Answer Quality Assessment** - LLM-as-judge scoring for generated answers

## Documentation

| Document | Description |
|----------|-------------|
| [Query Generation](query_generation.md) | Creating evaluation queries with the Streamlit UI |
| [Running Evaluations](running_evaluations.md) | Using the evaluation harness |
| [Metrics Reference](metrics.md) | Retrieval and answer quality metrics |
| [Traces and Logging](traces_and_logging.md) | Observability and debugging |

## Quick Start

### 1. Build an Index

```bash
make index  # or make index-dummy for testing
```

### 2. Create Evaluation Queries

Launch the query curation UI:

```bash
make eval
```

Browse chunks, optionally generate query suggestions with LLM, and save queries with ground-truth chunk IDs.

### 3. Run Evaluation

```python
from pathlib import Path
from src.rag.eval.harness import load_eval_queries, run_full_eval
from src.rag.app.container import build_container

# Load queries and container
queries = load_eval_queries(Path("experiments/queries.jsonl"))
container = build_container()

# Run evaluation
run = run_full_eval(
    eval_queries=queries,
    container=container,
    top_k=10,
    run_generation=True,
    use_llm_judge=True,
)

# Access results
print(f"Overall Recall@10: {run.aggregates.overall['recall@10']:.2%}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation System                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Query      │    │  Eval        │    │   Metrics    │  │
│  │   Curation   │───▶│  Harness     │───▶│   & Reports  │  │
│  │   (UI)       │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  EvalQuery   │    │  Container   │    │  EvalRun     │  │
│  │  Dataset     │    │  (Pipeline)  │    │  Artifacts   │  │
│  │  (.jsonl)    │    │              │    │  (.json)     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### EvalQuery

The fundamental unit of evaluation. Contains:

- **Query text** - The question to ask
- **Relevant chunk IDs** - Ground truth for retrieval evaluation
- **Expected answer** - Reference answer for generation evaluation
- **Metadata** - Query type, difficulty, tags, optional retrieval filter

### EvalDataset

A collection of EvalQuery objects with filtering and statistics:

```python
dataset = EvalDataset(queries)
hard_queries = dataset.filter_by_difficulty(Difficulty.hard)
factual_queries = dataset.filter_by_type(QueryType.factual)
print(dataset.stats())
```

### EvalRun

Complete evaluation run output including:

- Per-query results with retrieval and answer metrics
- Aggregated metrics overall and by query type/difficulty
- Execution metadata (models used, timing, configuration)

## File Locations

```
src/rag/eval/
├── schema.py      # EvalQuery, EvalDataset, QueryType, Difficulty
├── models.py      # EvalResult, EvalRun, metrics dataclasses
├── metrics.py     # Retrieval metric calculations
└── harness.py     # Evaluation orchestration

eval/app/
├── main.py        # Streamlit entry point
├── state.py       # Session state management
├── wizard.py      # Create/Review tabs
└── components/    # UI components

src/rag/adapters/
├── query_suggestion/    # LLM query generation
├── eval_persistence/    # JSONL storage
└── chunk_loading/       # Chunk loading for UI
```

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Build commands and project overview
- [Architecture](../ARCHITECTURE.md) - System design
- [API Reference](../API_REFERENCE.md) - Domain models and ports
