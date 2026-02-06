# Phase 2: Retrieval Quality, Evaluation & Observability Analysis

## Date: 2026-02-05
## Status: COMPLETED

---

## 2.1 Retrieval Quality Assessment

### 2.1.1 Vector Store Implementations

#### InMemoryVectorStore (src/rag/adapters/vectorstores/in_memory_store.py)

```python
@dataclass
class InMemoryVectorStore:
    """Pure in-memory vector store with cosine similarity search."""
    _chunks: list[Chunk] = field(default_factory=list)
    _vectors: list[Vector] = field(default_factory=list)
```

**Capabilities:**
- Cosine similarity search via numpy
- Metadata filtering via InMemoryFilterEvaluator
- No persistence (save/load are no-ops)

**Limitations:**
- O(N) search complexity (brute force)
- Memory-only (lost on restart)
- Single-threaded

**Use Case:** Testing, small datasets (<10K vectors)

---

#### JsonlVectorStore (src/rag/adapters/vectorstores/jsonl_store.py)

```python
@dataclass
class JsonlVectorStore:
    """JSONL-persisted vector store with in-memory search."""
    path: Path
    _chunks: list[Chunk] = field(default_factory=list)
    _vectors: list[Vector] = field(default_factory=list)
```

**Capabilities:**
- Persists to JSONL file (one line per chunk+vector)
- Loads entire index into memory on startup
- Same O(N) search as InMemoryVectorStore

**Limitations:**
- Load time increases with index size
- Memory usage equals full index size
- No incremental updates (full rewrite on save)

**Use Case:** Small to medium datasets (<1M vectors), development

---

#### QdrantVectorStore (src/rag/adapters/vectorstores/qdrant_store.py)

```python
@dataclass(slots=True)
class QdrantVectorStore:
    """Qdrant-backed vector store for scalable similarity search."""
    collection_name: str
    vector_size: int
    url: str | None = None          # Remote server
    path: str | None = None         # Local disk
    api_key: str | None = None      # Qdrant Cloud
    distance: Distance = Distance.COSINE
```

**Capabilities:**
- HNSW approximate nearest neighbor search
- Supports local disk, remote server, or Qdrant Cloud
- Persistent collections
- Query filtering via QdrantFilterCompiler

**Limitations:**
- No HNSW parameter tuning exposed (ef, m, etc.)
- No replica configuration
- No connection pooling

**Use Case:** Production deployments requiring scale

---

### 2.1.2 Retriever Implementation

**VectorRetriever** (src/rag/adapters/retrieval/vector_retriever.py):

```python
@dataclass(frozen=True, slots=True)
class VectorRetriever:
    embedder: Embedder
    store: VectorStore

    def retrieve(self, query: str, *, top_k: int, where: Where = None) -> list[Candidate]:
        q_vec = self.embedder.embed_texts([query])[0]
        return self.store.search(query_vector=q_vec, top_k=top_k, where=where)
```

**Analysis:**
- Simple composition pattern
- Synchronous blocking calls
- No query pre-processing
- No hybrid search (keyword + vector)

---

### 2.1.3 Reranker Implementation

**HeuristicReranker** (src/rag/adapters/reranking/rerank_heuristic.py):

```python
class HeuristicReranker:
    """
    Cheap reranker:
    - starts from vector similarity score
    - adds lexical overlap boost between query and chunk text
    - optionally diversifies by doc_id
    """
    def __init__(self, *, overlap_weight: float = 0.15, diversify: bool = True, max_per_doc: int = 3)
```

**Scoring Formula:**
```
new_score = base_score + overlap_weight * (token_overlap_ratio)
```

**Diversification:**
- Limits chunks per document (default max_per_doc=3)
- Prevents single document from dominating results

**Limitations:**
- No learned model (cross-encoder, BGE, etc.)
- Simple token overlap doesn't capture semantic similarity
- No query intent classification

---

### 2.1.4 Retrieval Quality Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No hybrid search (keyword + vector) | Misses exact matches, acronyms | P1 |
| No query expansion (synonyms) | Low recall for paraphrased queries | P1 |
| No learned reranker | Suboptimal ranking vs cross-encoders | P2 |
| No query classification/routing | One-size-fits-all retrieval | P2 |
| No relevance feedback | Cannot learn from user interactions | P3 |

---

## 2.2 Evaluation Framework Assessment

### 2.2.1 Evaluation Metrics (src/rag/eval/metrics.py)

**Retrieval Metrics:**

| Metric | Implementation | Quality |
|--------|----------------|---------|
| Recall@k | `recall_at_k()` | Correct |
| Precision@k | `precision_at_k()` | Correct |
| Hit Rate@k | `hit_rate_at_k()` | Correct |
| MRR | `mrr()` | Correct |
| MAP | `average_precision()` | Correct |
| NDCG@k | `ndcg_at_k()` | Correct (binary relevance) |
| Semantic Similarity | `semantic_similarity()` | Uses cosine of embeddings |

**Metric Aggregation (summarize function):**
- Computes mean across all queries
- Returns RetrievalSummary with all metrics at k=1,3,5,10

---

### 2.2.2 LLM-as-Judge Implementation (src/rag/eval/judges.py)

**Gold Judge (Answer Quality vs Expected):**

```python
GOLD_JUDGE_VERSION = "gold_v2"
GOLD_JUDGE_PROMPT = """
Evaluate GENERATED ANSWER vs EXPECTED ANSWER on:
- CORRECTNESS (0-5): Factual accuracy
- COMPLETENESS (0-5): Coverage of key points
- RELEVANCE (0-5): Direct answer to query
"""
```

**Groundedness Judge (Hallucination Detection):**

```python
GROUNDEDNESS_JUDGE_VERSION = "groundedness_v3"
GROUNDEDNESS_JUDGE_PROMPT = """
Analyze whether GENERATED ANSWER is supported by RETRIEVED CONTEXT:
- answerable_from_context: bool
- evidence_bounded: bool (every claim supported)
- supported_claims: int
- unsupported_claims: int
- claims: List[Claim] with role (core/peripheral) and supported bool
"""
```

**Known Issues (from docs/KNOWN_ISSUES.md):**
- Correctness and completeness showing 0 for accurate responses
- High correctness AND high hallucination scores (contradictory)
- Gold judge calibration issues

---

### 2.2.3 Evaluation Harness (src/rag/eval/harness.py)

**run_full_eval() capabilities:**
- Retrieval-only or full pipeline evaluation
- LLM judge integration (optional)
- Per-query latency tracking
- Metadata tracking (models, versions, config)

**EvalRun output structure:**
```python
@dataclass
class EvalRun:
    meta: EvalRunMeta          # Run metadata
    results: tuple[EvalResult] # Per-query results
    aggregates: EvalAggregates # Summary statistics
```

**Aggregation includes:**
- Overall metrics (all queries)
- By query type (factual, synthesis, etc.)
- By difficulty (easy, medium, hard)
- Answer quality metrics (when LLM judge enabled)

---

### 2.2.4 Results Analyzer UI (eval/app/results_analyzer.py)

**Features:**
- Single run analysis with metrics tables
- Side-by-side run comparison with delta highlighting
- Multi-run trending with time series charts
- Query explorer with filtering
- Trace viewer for debugging

**Architecture:**
- Domain models: RunSummary, LoadedRun, RunComparison, TrendAnalysis
- Services: ComparisonService, FilterService, TrendService
- Adapters: FilesystemRunLoader

---

### 2.2.5 Evaluation Framework Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| Judge calibration issues | Inaccurate quality metrics | P0 |
| No statistical significance testing | Cannot trust metric differences | P1 |
| No experiment tracking (W&B, MLflow) | Hard to compare experiments | P1 |
| No failure case analysis | Cannot understand why retrieval fails | P2 |
| No embedding space visualization | Cannot debug semantic gaps | P2 |

---

## 2.3 Observability Infrastructure

### 2.3.1 Current Logging Implementation

**JsonlQueryLogger** (src/rag/adapters/logging/jsonl_logger.py):

```python
class JsonlQueryLogger:
    def __init__(self, path: str | Path, *, redact_text: bool = False)
    
    def log(self, trace: QueryTrace) -> None:
        # Appends JSON line to file
        # Atomic append on POSIX (single-process)
```

**QueryTrace structure:**
```python
@dataclass
class QueryTrace:
    trace_id: str
    query: str
    created_at: datetime
    retrieved: tuple[Candidate, ...]
    reranked: tuple[Candidate, ...]
    packed_chunk_ids: tuple[str, ...]
    model: str | None
    latency_ms: int
    metadata: dict  # Includes timing breakdown
    answer: Answer | None
```

**Timing breakdown (from query_runner.py):**
```python
"timing_ms": {
    "retrieval": t_retrieval_ms,
    "rerank": t_rerank_ms,
    "context": t_context_ms,
    "generation": t_gen_ms,
    "total": total_ms,
}
```

---

### 2.3.2 Observability Gaps

| Missing Component | Purpose | Industry Standard |
|-------------------|---------|-------------------|
| Metrics endpoint | Latency/error/throughput metrics | Prometheus |
| Distributed tracing | Request flow across services | OpenTelemetry/Jaeger |
| Structured logging | Machine-readable logs | JSON to stdout |
| Health checks | Container orchestration | HTTP /healthz |
| Alerting thresholds | Proactive issue detection | PagerDuty/Opsgenie |
| Cost tracking | API spend monitoring | Custom metrics |
| Dashboards | Visual monitoring | Grafana/Datadog |

---

### 2.3.3 Logging Limitations

**Current Issues:**
1. JSONL files are local-only (not aggregated)
2. No log rotation (files grow indefinitely)
3. No log levels (DEBUG, INFO, WARN, ERROR)
4. No correlation IDs for distributed tracing
5. File locking not implemented (multi-process unsafe)

---

## 2.4 Phase 2 Conclusions

### Retrieval Quality Grade: B+

**Strengths:**
- Multiple vector store backends (memory, JSONL, Qdrant)
- HNSW support via Qdrant
- Metadata filtering
- Heuristic reranking with diversity

**Weaknesses:**
- No hybrid search
- No learned reranker
- No query expansion
- O(N) search in memory/JSONL stores

### Evaluation Framework Grade: A-

**Strengths:**
- Comprehensive metrics (Recall, NDCG, MRR, MAP)
- LLM-as-judge for answer quality
- Groundedness/hallucination detection
- Rich results analysis UI
- Query type and difficulty breakdowns

**Weaknesses:**
- Judge calibration issues (known bugs)
- No statistical significance testing
- No experiment tracking integration

### Observability Grade: C

**Strengths:**
- Query traces with full pipeline data
- Per-stage timing breakdown
- JSONL format is human-readable

**Weaknesses:**
- No metrics aggregation
- No production logging infrastructure
- No health checks
- No alerting

---

*Next: Phase 3 - Scalability Patterns, Failure Modes & Production Readiness*
