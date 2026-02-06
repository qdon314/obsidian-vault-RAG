# Phase 1: Architecture Exploration Results

## Date: 2026-02-05
## Status: COMPLETED

---

## 1.1 System Overview

The Obsidian Vault RAG system implements a **Retrieval-Augmented Generation** pipeline with a focus on observability and evaluation. The architecture follows the **Hexagonal (Ports & Adapters)** pattern, enabling clean separation of concerns and easy component swapping.

### Key Architectural Goals (from docs/ARCHITECTURE.md)

1. **Observability First**: Every query generates a complete trace with all intermediate results
2. **Evaluation as First-Class**: Built-in evaluation framework with comprehensive metrics
3. **Clean Boundaries**: Protocol-based interfaces allow easy swapping of implementations
4. **Reproducibility**: Stable IDs for documents and chunks enable deterministic behavior

---

## 1.2 Hexagonal Architecture Implementation

### Ports (Abstract Interfaces) - src/rag/ports/

| Port | Purpose | File |
|------|---------|------|
| Chunker | Splits Documents into Chunks | chunker.py |
| Embedder | Text to vectors | embedder.py |
| VectorStore | Stores and searches vectors | vector_store.py |
| Retriever | Query to Candidates | retriever.py |
| Reranker | Re-scores candidates | reranker.py |
| ContextBuilder | Packs candidates into prompt | context_builder.py |
| Generator | Produces final answer | generator.py |
| QueryLogger | Persist query traces | logger.py |

### Domain Models - src/rag/domain/models.py

| Model | Responsibility |
|-------|---------------|
| Document | Raw source unit before chunking; stable ID from content hash |
| Chunk | Embedding unit with provenance (offsets, section info) |
| Candidate | Retrieved chunk + retrieval/rerank scores |
| Citation | Source pointer for answer attribution |
| ContextPack | Final evidence bundle for generator |
| Answer | LLM output with citations and abstention flag |
| QueryTrace | Complete observability record for debugging/evaluation |

### Adapters (Concrete Implementations) - src/rag/adapters/

| Adapter | Port | Description |
|---------|------|-------------|
| FixedChunker | Chunker | Character-based chunking (800/120) |
| ObsidianStructuralChunker | Chunker | Markdown-aware structural chunking |
| ObsidianPropositionChunker | Chunker | Proposition-based chunking (seq2seq) |
| OpenAIEmbedder | Embedder | OpenAI text-embedding-3-large |
| DummyEmbedder | Embedder | Random vectors for testing |
| JsonlVectorStore | VectorStore | JSONL-persisted, in-memory search |
| InMemoryVectorStore | VectorStore | Pure in-memory (no persistence) |
| QdrantVectorStore | VectorStore | Qdrant backend for scale |
| VectorRetriever | Retriever | Composes Embedder + VectorStore |
| HeuristicReranker | Reranker | Lexical overlap boost + diversity |
| NoOpReranker | Reranker | Pass-through (baseline) |
| SimpleContextBuilder | ContextBuilder | Token budget + deduplication |
| OpenAIChatGenerator | Generator | GPT-4.1-mini chat completions |
| JsonlQueryLogger | QueryLogger | JSONL append logging |

---

## 1.3 Configuration System

### Settings Structure (settings.toml)

```toml
[paths]
vault_dir = "/path/to/vault"
artifacts_dir = "artifacts"
index_dir = "artifacts/indexes/obsidian"

[chunking]
backend = "obsidian_structural" # or "fixed", "obsidian_proposition"
chunk_size = 800
overlap = 120

[embeddings]
backend = "openai"
model = "text-embedding-3-large"
cache_embeddings = true

[vectorstore]
backend = "jsonl" # or "memory", "qdrant"

[retrieval]
top_k = 8

[rerank]
enabled = true
backend = "heuristic"
keep_k = 4

[llm]
model = "gpt-4.1-mini"
temperature = 0.2
max_tokens = 1024
```

### Container Composition (src/rag/app/container.py)

The Container dataclass wires all adapters together:

```python
@dataclass(frozen=True, slots=True)
class Container:
    chunker: Chunker
    context_builder: ContextBuilder
    embedder: Embedder
    generator: Generator
    ingestor: Ingestor
    store: VectorStore
    retriever: Retriever
    logger: QueryLogger
    reranker: Reranker
```

---

## 1.4 Data Flow Architecture

### Indexing Pipeline

```
Raw File (*.md, *.txt)
    ↓
Document (doc_id = hash(source + path + content))
    ↓
Chunking (strategy-dependent)
    ↓
Chunk (chunk_id = {doc_id}:{strategy}:{index}:{start}-{end})
    ↓
Embedding (OpenAI / Dummy)
    ↓
VectorStore (JSONL / InMemory / Qdrant)
```

### Query Pipeline

```
User Query
    ↓
Embedding
    ↓
Vector Search (top_k candidates)
    ↓
Reranking (heuristic / noop)
    ↓
Context Building (token budget)
    ↓
LLM Generation
    ↓
Answer + Citations + QueryTrace
```

---

## 1.5 Design Principles (Verified)

### 1. Stable Identifiers
- `doc_id`: Hash of (source + path + content)
- `chunk_id`: `{doc_id}:{strategy}:{index}:{start}-{end}`
- Enables reproducible experiments and cache-friendly operations

### 2. Metadata Preservation
- Document metadata → Chunk metadata → Citation metadata
- Section headings and paths preserved for context

### 3. Token Budget Enforcement
- Simple heuristic: ~4 characters per token
- Stops adding chunks when budget exceeded
- Deduplication prevents redundant content

### 4. Observability
- Every query generates a `QueryTrace` containing:
  - All retrieved candidates with scores
  - All reranked candidates with scores
  - Packed chunk IDs (what went to LLM)
  - Timing breakdown by stage
  - Final answer with citations

### 5. Composability
- Small, focused adapters composed via the `Container`
- Each adapter does one thing well
- CLI can override settings for experiments

### 6. Evaluation as First-Class
- `EvalQuery` schema for ground truth
- Retrieval metrics (Recall, Precision, MRR, NDCG)
- Answer quality metrics (LLM-as-judge)
- Breakdowns by query type and difficulty

---

## 1.6 Project Structure

```
obsidian-vault-RAG/
├── src/rag/
│   ├── adapters/          # Concrete implementations
│   │   ├── chunking/
│   │   ├── embedding/
│   │   ├── vectorstores/
│   │   ├── retrieval/
│   │   ├── reranking/
│   │   ├── context_building/
│   │   ├── generation/
│   │   ├── ingestion/
│   │   ├── logging/
│   │   └── filters/
│   ├── domain/            # Domain models
│   ├── ports/             # Abstract interfaces
│   ├── app/               # Orchestration
│   │   ├── container.py
│   │   ├── query_runner.py
│   │   └── pipeline.py
│   ├── eval/              # Evaluation framework
│   │   ├── harness.py
│   │   ├── metrics.py
│   │   ├── judges.py
│   │   └── reducers.py
│   └── settings.py
├── eval/                  # Evaluation UI and datasets
│   ├── app/               # Streamlit results analyzer
│   ├── datasets/
│   └── scripts/
├── tests/                 # Test suite
├── infra/                 # Terraform AWS infrastructure
├── docs/                  # Documentation
└── scripts/               # Build/query scripts
```

---

## 1.7 Dependencies (pyproject.toml)

```toml
dependencies = [
    "llama-index",
    "chromadb",
    "python-dotenv",
    "rich",
    "dataclasses-json>=0.6.0",
    "torch",
    "llama-index-vector-stores-chroma",
    "llama-index-embeddings-huggingface"
]

[project.optional-dependencies]
openai = ["openai", "llama-index-llms-openai"]
qdrant = ["qdrant-client>=1.7.0"]
ui = ["streamlit>=1.30.0", "plotly>=5.18.0"]
aws = ["boto3>=1.34.0"]
dev = ["pytest", "mypy", "ruff"]
```

---

## 1.8 Key Files Summary

| File | Purpose |
|------|---------|
| src/rag/settings.py | TOML configuration loader with env overrides |
| src/rag/app/container.py | Dependency injection container |
| src/rag/app/query_runner.py | Full RAG pipeline orchestration |
| src/rag/domain/models.py | Core domain models (Document, Chunk, etc.) |
| src/rag/eval/harness.py | Evaluation orchestration |
| src/rag/eval/metrics.py | Retrieval metrics implementation |
| src/rag/eval/judges.py | LLM-as-judge prompts and evaluation |
| eval/app/results_analyzer.py | Streamlit evaluation results UI |

---

## 1.9 Phase 1 Conclusions

### Strengths Identified

1. **Clean Architecture**: Hexagonal pattern properly implemented with clear port/adapter separation
2. **Type Safety**: Heavy use of dataclasses, type hints, and frozen immutability
3. **Configuration Management**: Centralized TOML config with CLI overrides
4. **Documentation**: Comprehensive docs (ARCHITECTURE.md, CLAUDE.md, etc.)
5. **Testing Structure**: Unit tests for adapters, domain logic

### Initial Concerns (to be explored in Phase 2-3)

1. No async support identified in synchronous code paths
2. OpenAI client instantiated per-call (no connection pooling)
3. No retry logic visible in adapter implementations
4. SQLite embedding cache is local-only (no distributed cache)

---

*Next: Phase 2 - Retrieval Quality & Evaluation Framework Analysis*
