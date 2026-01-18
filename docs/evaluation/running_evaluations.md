# Running Evaluations

This document covers using the evaluation harness to run retrieval and end-to-end evaluations.

## Overview

The evaluation harness (`src/rag/eval/harness.py`) orchestrates evaluation runs through the RAG pipeline. It supports:

- **Retrieval-only evaluation** - Test embedding and vector search quality
- **Full pipeline evaluation** - Test retrieve → rerank → context → generate
- **LLM-as-judge evaluation** - Automated answer quality scoring

## Prerequisites

1. **Built index** - Run `make index` or `make index-dummy`
2. **Evaluation queries** - Create via UI or manually in JSONL format
3. **API keys** - Set in `settings.toml` or environment variables

## Quick Start

### Loading Queries

```python
from pathlib import Path
from src.rag.eval.harness import load_eval_queries

queries = load_eval_queries(Path("experiments/queries.jsonl"))
print(f"Loaded {len(queries)} queries")
```

### Building the Container

```python
from src.rag.app.container import build_container

# Uses settings from settings.toml
container = build_container()
```

### Running Retrieval Evaluation

```python
from src.rag.eval.harness import run_retrieval_eval

# Single query
result = run_retrieval_eval(
    query=queries[0],
    retriever=container.retriever,
    top_k=10,
)

print(f"Retrieved: {result.retrieved_chunk_ids}")
print(f"Recall@10: {result.recall_at_k(10):.2%}")
```

### Running Full Evaluation

```python
from src.rag.eval.harness import run_full_eval

run = run_full_eval(
    eval_queries=queries,
    container=container,
    top_k=10,              # Retrieve this many candidates
    keep_k=5,              # Keep this many after reranking
    token_budget=4000,     # Context window budget
    run_generation=True,   # Run the generator
    use_llm_judge=True,    # Score answers with LLM
    score_ids="reranked",  # Evaluate reranked or retrieved IDs
)

# Access results
print(f"Overall Recall@10: {run.aggregates.overall['recall@10']:.2%}")
print(f"Mean Correctness: {run.aggregates.answer_quality['mean_correctness']:.2%}")
```

## Configuration Options

### run_full_eval Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_queries` | list[EvalQuery] | required | Queries to evaluate |
| `container` | Container | required | Pipeline container |
| `top_k` | int | 10 | Number of candidates to retrieve |
| `keep_k` | int | None | Number to keep after reranking (defaults to top_k) |
| `token_budget` | int | 4000 | Max tokens for context |
| `run_generation` | bool | False | Whether to run the generator |
| `use_llm_judge` | bool | False | Whether to score answers with LLM |
| `score_ids` | str | "retrieved" | Which IDs to score: "retrieved" or "reranked" |
| `save_path` | Path | None | Where to save results |

### Retrieval-Only Mode

For fast iteration on embedding/retrieval changes:

```python
run = run_full_eval(
    eval_queries=queries,
    container=container,
    top_k=20,
    run_generation=False,  # Skip generation
    use_llm_judge=False,   # Skip LLM judge
)
```

### With Filters

Queries with filters in their metadata automatically apply them during retrieval:

```python
# Query with filter
query = EvalQuery(
    qid="q_001",
    query="What is the API endpoint?",
    relevant_chunk_ids={"c1", "c2"},
    metadata={
        "filter": {
            "type": "Eq",
            "field": "doc_id",
            "value": "api_reference.md"
        }
    }
)

# Filter is automatically applied
result = run_retrieval_eval(
    query=query,
    retriever=container.retriever,
    top_k=10,
)
```

## Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      run_full_eval()                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  for each EvalQuery:                                            │
│    ┌──────────────────────────────────────────────────────┐     │
│    │  1. Retrieve candidates (top_k, with filter)          │     │
│    │     └─ embedder.embed() → vectorstore.search()        │     │
│    ├──────────────────────────────────────────────────────┤     │
│    │  2. Rerank candidates (if enabled)                    │     │
│    │     └─ reranker.rerank() → keep top keep_k            │     │
│    ├──────────────────────────────────────────────────────┤     │
│    │  3. Build context (if run_generation)                 │     │
│    │     └─ context_builder.build() → ContextPack          │     │
│    ├──────────────────────────────────────────────────────┤     │
│    │  4. Generate answer (if run_generation)               │     │
│    │     └─ generator.generate() → Answer                  │     │
│    ├──────────────────────────────────────────────────────┤     │
│    │  5. Evaluate answer (if use_llm_judge)                │     │
│    │     └─ evaluate_answer_quality() → AnswerQualityMetrics│    │
│    ├──────────────────────────────────────────────────────┤     │
│    │  6. Compute retrieval metrics                         │     │
│    │     └─ recall, precision, MRR, NDCG, etc.             │     │
│    └──────────────────────────────────────────────────────┘     │
│                                                                  │
│  aggregate_results() → EvalAggregates                           │
│  save_run() → JSONL + JSON artifacts                            │
│                                                                  │
│  return EvalRun                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Output Format

### EvalRun Structure

```python
@dataclass
class EvalRun:
    run_id: str                    # Unique run identifier
    meta: EvalRunMeta              # Configuration and environment
    results: list[EvalResult]      # Per-query results
    aggregates: EvalAggregates     # Aggregated metrics
    artifacts: dict[str, Path]     # Saved file paths
```

### Per-Query Results

```python
@dataclass
class EvalResult:
    qid: str
    query: str
    query_type: QueryType
    difficulty: Difficulty

    # Retrieval metrics
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: set[str]
    retrieval_metrics: dict        # recall@k, precision@k, etc.

    # Answer metrics (if run_generation)
    generated_answer: str
    expected_answer: str
    answer_quality: AnswerQualityMetrics

    # Timing
    latency_ms: float
```

### Aggregated Metrics

```python
@dataclass
class EvalAggregates:
    overall: dict                  # Full dataset metrics
    by_type: dict[QueryType, dict] # Breakdown by query type
    by_difficulty: dict[Difficulty, dict]  # Breakdown by difficulty
    answer_quality: dict           # Mean correctness, completeness, etc.
    latency_ms: dict               # p50, p95, p99
```

## Saving and Loading Results

### Saving Results

```python
from src.rag.eval.harness import save_run

artifacts = save_run(
    run=run,
    output_dir=Path("experiments/runs"),
)

print(f"Results: {artifacts['results']}")  # JSONL file
print(f"Metrics: {artifacts['metrics']}")  # JSON file
```

### Loading Previous Results

```python
import json
from pathlib import Path

# Load aggregated metrics
with open("experiments/runs/run_001_metrics.json") as f:
    metrics = json.load(f)

# Load per-query results
with open("experiments/runs/run_001_results.jsonl") as f:
    results = [json.loads(line) for line in f]
```

## Comparing Runs

Example script to compare two evaluation runs:

```python
import json
from pathlib import Path

def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def compare_runs(run_a: Path, run_b: Path):
    a = load_metrics(run_a)
    b = load_metrics(run_b)

    print("Metric            | Run A   | Run B   | Delta")
    print("-" * 50)

    for metric in ["recall@10", "mrr", "map"]:
        val_a = a["overall"].get(metric, 0)
        val_b = b["overall"].get(metric, 0)
        delta = val_b - val_a
        sign = "+" if delta > 0 else ""
        print(f"{metric:17} | {val_a:.3f}   | {val_b:.3f}   | {sign}{delta:.3f}")

# Usage
compare_runs(
    Path("experiments/runs/baseline_metrics.json"),
    Path("experiments/runs/improved_metrics.json"),
)
```

## Common Workflows

### Baseline Evaluation

```bash
# 1. Build index
make index

# 2. Create queries
make eval  # Use Streamlit UI

# 3. Run baseline
python -c "
from pathlib import Path
from src.rag.eval.harness import load_eval_queries, run_full_eval, save_run
from src.rag.app.container import build_container

queries = load_eval_queries(Path('experiments/queries.jsonl'))
container = build_container()
run = run_full_eval(queries, container, top_k=10)
save_run(run, Path('experiments/runs/baseline'))
print(f'Recall@10: {run.aggregates.overall[\"recall@10\"]:.2%}')
"
```

### A/B Testing Configurations

```python
from src.rag.app.container import build_container
from src.rag.eval.harness import run_full_eval, save_run

# Test different chunking strategies
for chunk_size in [256, 512, 1024]:
    container = build_container(
        overrides={"chunking.chunk_size": chunk_size}
    )
    run = run_full_eval(queries, container, top_k=10)
    save_run(run, Path(f"experiments/runs/chunk_{chunk_size}"))
```

### Debugging Low Scores

```python
# Find worst-performing queries
sorted_results = sorted(
    run.results,
    key=lambda r: r.retrieval_metrics.get("recall@10", 0)
)

for result in sorted_results[:5]:  # Bottom 5
    print(f"\nQuery: {result.query}")
    print(f"Expected: {result.relevant_chunk_ids}")
    print(f"Retrieved: {result.retrieved_chunk_ids[:5]}")
    print(f"Recall@10: {result.retrieval_metrics['recall@10']:.2%}")
```

## See Also

- [Query Generation](query_generation.md) - Creating evaluation queries
- [Metrics Reference](metrics.md) - Understanding metrics
- [Traces and Logging](traces_and_logging.md) - Debugging evaluations
