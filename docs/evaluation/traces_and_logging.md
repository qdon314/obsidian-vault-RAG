# Traces and Logging

This document covers the observability system for debugging and analyzing RAG pipeline execution.

## Overview

Every query through the pipeline produces a `QueryTrace` - a structured record containing:

- Query text and timing
- Retrieved and reranked candidates
- Context building decisions
- Generated answer
- Per-stage timing breakdown

Traces are persisted to JSONL files for post-hoc analysis and debugging.

## QueryTrace Model

```python
@dataclass(frozen=True, slots=True)
class QueryTrace:
    # Identity
    trace_id: str                  # Unique hex ID (UUID4)
    query: str                     # Original query text
    created_at: datetime           # Timestamp (UTC)

    # Retrieval stage
    top_k: int                     # Number of candidates requested
    retrieved: Sequence[Candidate] # Retrieved candidates with scores

    # Rerank stage
    reranked: Sequence[Candidate]  # Reranked candidates
    keep_k: int | None             # Truncation limit after rerank
    reranker: str | None           # Reranker name (if enabled)

    # Context build stage
    token_budget: int              # Max tokens for context
    packed_chunk_ids: Sequence[str]  # Chunk IDs that fit in context

    # Generation stage
    model: str | None              # LLM model name
    latency_ms: int | None         # Total pipeline latency
    estimated_cost_usd: float | None  # Estimated API cost

    # Result
    answer: Answer | None          # Generated answer with citations
    metadata: Mapping[str, Any]    # Custom metadata and timing breakdown
```

## QueryLogger Port

The `QueryLogger` protocol defines the interface for persisting traces:

```python
class QueryLogger(Protocol):
    def log(self, trace: QueryTrace) -> None:
        ...
```

### JsonlQueryLogger

The primary implementation persists traces to JSONL files:

```python
from src.rag.adapters.logging import JsonlQueryLogger

logger = JsonlQueryLogger(
    path="traces/queries.jsonl",
    redact_text=False,  # Set True for privacy
)
```

**Features:**
- Atomic appends (POSIX-safe for single-process)
- Optional text redaction for sensitive content
- Automatic directory creation
- JSON serialization of dataclasses

**Redacted Fields** (when `redact_text=True`):
- `text`, `page_content`, `chunk_text`
- `context_text`, `answer`

## Pipeline Integration

The `run_query` function automatically creates and logs traces:

```python
from src.rag.app.query_runner import run_query

result = run_query(
    query="What is X?",
    retriever=retriever,
    reranker=reranker,
    context_builder=context_builder,
    generator=generator,
    logger=logger,  # QueryLogger instance
    top_k=10,
    keep_k=5,
    token_budget=4000,
)

# Trace was automatically logged to the configured path
```

### Timing Breakdown

Each trace includes detailed timing in `metadata["timing_ms"]`:

```python
{
    "timing_ms": {
        "retrieval": 45,    # Vector search time
        "rerank": 120,      # Reranking time
        "context": 5,       # Context building time
        "generation": 850,  # LLM generation time
        "total": 1020       # End-to-end latency
    }
}
```

## Trace File Format

Traces are stored as JSONL (one JSON object per line):

```jsonl
{"trace_id": "a1b2c3...", "query": "What is X?", "created_at": "2024-01-15T14:30:00Z", ...}
{"trace_id": "d4e5f6...", "query": "How does Y work?", "created_at": "2024-01-15T14:30:05Z", ...}
```

### Example Trace

```json
{
  "trace_id": "a1b2c3d4e5f6789012345678",
  "query": "What is the API endpoint for authentication?",
  "created_at": "2024-01-15T14:30:00.123456+00:00",
  "top_k": 10,
  "retrieved": [
    {
      "chunk": {
        "chunk_id": "c_abc123",
        "doc_id": "api_reference.md",
        "text": "The authentication endpoint is...",
        "start_char": 1500,
        "end_char": 1800,
        "metadata": {"section": "Authentication"}
      },
      "score": 0.89
    }
  ],
  "reranked": [...],
  "keep_k": 5,
  "reranker": "cross-encoder",
  "token_budget": 4000,
  "packed_chunk_ids": ["c_abc123", "c_def456"],
  "model": "gpt-4o-mini",
  "latency_ms": 1020,
  "estimated_cost_usd": 0.0015,
  "answer": {
    "text": "The authentication endpoint is POST /api/auth...",
    "citations": [{"chunk_id": "c_abc123", "excerpt": "..."}]
  },
  "metadata": {
    "timing_ms": {
      "retrieval": 45,
      "rerank": 120,
      "context": 5,
      "generation": 850,
      "total": 1020
    }
  }
}
```

## Reading and Analyzing Traces

### Loading Traces

```python
import json
from pathlib import Path

def load_traces(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]

traces = load_traces(Path("traces/queries.jsonl"))
print(f"Loaded {len(traces)} traces")
```

### Analyzing Latency

```python
import statistics

latencies = [t["latency_ms"] for t in traces if t.get("latency_ms")]

print(f"Mean latency: {statistics.mean(latencies):.0f}ms")
print(f"P50 latency: {statistics.median(latencies):.0f}ms")
print(f"P95 latency: {sorted(latencies)[int(len(latencies) * 0.95)]:.0f}ms")
```

### Finding Slow Queries

```python
slow_queries = [
    t for t in traces
    if t.get("latency_ms", 0) > 2000
]

for t in slow_queries[:5]:
    timing = t.get("metadata", {}).get("timing_ms", {})
    print(f"Query: {t['query'][:50]}...")
    print(f"  Retrieval: {timing.get('retrieval', 0)}ms")
    print(f"  Rerank: {timing.get('rerank', 0)}ms")
    print(f"  Generation: {timing.get('generation', 0)}ms")
```

### Debugging Retrieval Issues

```python
def analyze_retrieval(trace: dict):
    """Analyze what the retriever returned vs what was kept."""
    retrieved = trace.get("retrieved", [])
    reranked = trace.get("reranked", [])
    packed = trace.get("packed_chunk_ids", [])

    print(f"Query: {trace['query']}")
    print(f"Retrieved {len(retrieved)} → Reranked {len(reranked)} → Packed {len(packed)}")

    print("\nTop retrieved candidates:")
    for c in retrieved[:5]:
        chunk = c["chunk"]
        print(f"  {chunk['chunk_id'][:12]}... (score={c['score']:.3f})")
        print(f"    doc: {chunk['doc_id']}")
        print(f"    text: {chunk['text'][:100]}...")
```

## Evaluation Integration

Traces are used during evaluation for:

1. **Retrieval debugging** - See exactly which chunks were retrieved
2. **Reranking analysis** - Compare before/after reranking
3. **Context debugging** - See which chunks fit in the token budget
4. **Answer attribution** - Link answers to source chunks

### Correlating Traces with Eval Results

```python
from src.rag.eval.harness import run_full_eval

# Run evaluation (traces are logged automatically)
run = run_full_eval(queries, container, top_k=10)

# Load corresponding traces
traces = load_traces(Path("traces/queries.jsonl"))

# Match traces to eval results by query text
def find_trace(query: str, traces: list[dict]) -> dict | None:
    for t in reversed(traces):  # Most recent first
        if t["query"] == query:
            return t
    return None

for result in run.results:
    trace = find_trace(result.query, traces)
    if trace:
        print(f"Query: {result.query[:50]}...")
        print(f"  Recall@10: {result.retrieval_metrics['recall@10']:.2%}")
        print(f"  Latency: {trace['latency_ms']}ms")
```

## Best Practices

### Trace Storage

1. **Rotate logs** - Use date-based filenames: `traces_2024-01-15.jsonl`
2. **Compress old logs** - JSONL compresses well with gzip
3. **Redact in production** - Enable `redact_text=True` for user data

### Debugging Workflow

1. **Identify failing queries** - Low recall, incorrect answers
2. **Load corresponding traces** - Match by query text or trace_id
3. **Analyze retrieval** - Check scores, document sources
4. **Check reranking** - Did reranking help or hurt?
5. **Inspect context** - What chunks made it into the prompt?
6. **Review answer** - Are citations correct?

### Performance Monitoring

Monitor these timing metrics:
- `retrieval` > 200ms: Embedding or vector search slow
- `rerank` > 500ms: Reranker bottleneck
- `generation` > 2000ms: LLM latency

## Configuration

Logger configuration in container setup:

```python
from src.rag.adapters.logging import JsonlQueryLogger

logger = JsonlQueryLogger(
    path=settings.paths.traces_file,
    redact_text=settings.logging.redact_text,
)

container = Container(
    ...,
    logger=logger,
)
```

## See Also

- [Running Evaluations](running_evaluations.md) - Using the evaluation harness
- [Metrics Reference](metrics.md) - Understanding metrics
- [API Reference](../API_REFERENCE.md) - QueryTrace model details
