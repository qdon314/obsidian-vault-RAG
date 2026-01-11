# Production-Minded RAG System

A retrieval-augmented generation (RAG) system for answering questions over a document corpus, with a deliberate focus on retrieval behavior, evaluation, and failure modes rather than prompt tuning or UI polish.

## Why This Exists

In practice, most RAG failures come from retrieval issues, not generation. If the right information isn't surfaced, no amount of prompt engineering fixes the result.

This project explores how choices around chunking, embeddings, retrieval, and reranking actually affect downstream answers—and makes those effects visible.

---

## Architecture

```
Documents
  → Ingestion & Chunking
    → Embeddings
      → Vector Retrieval
        → (Optional) Reranking
          → Context Building
            → LLM Generation
```

### Design Principles

- Components with clear boundaries (ports/adapters pattern)
- Intermediate results that can be inspected (JSONL logging, query traces)
- Easy to swap or experiment with retrieval and reranking strategies
- Evaluation as a first-class concern

---

## Project Structure

```
src/rag/
├── domain/           # Core data models (Document, Chunk, Candidate, Answer, QueryTrace)
├── ports/            # Abstract interfaces (Chunker, Embedder, Retriever, Generator, etc.)
├── adapters/         # Concrete implementations
│   ├── chunking/     # Fixed-size chunker
│   ├── embedding/    # OpenAI, dummy embedder, SQLite cache
│   ├── generation/   # OpenAI chat
│   ├── ingestion/    # Filesystem loader, Obsidian markdown loader
│   ├── retrieval/    # Vector retriever
│   ├── vectorstores/ # JSONL store, in-memory store
│   ├── reranking/    # Heuristic reranker, no-op
│   ├── context_building/
│   └── logging/      # JSONL query logger
├── app/              # Pipeline orchestration & dependency injection
├── eval/             # Evaluation harness & metrics
└── settings.py       # Configuration loading

scripts/
├── build_index.py    # Build index from corpus
├── ask.py            # Query the system
└── project_state.py  # Project inspection

experiments/
├── run_eval.py       # Evaluation runner
├── generate_queries.py
├── curate_queries.py
└── create_starter_set.py
```

---

## Features

### Ingestion
- **Obsidian-aware loading**: Expands transclusions/embeds, preserves structure
- **Multiple formats**: Markdown (.md), plain text (.txt)
- **Document tracking**: Stable doc_id from content hash

### Chunking
- **Fixed-size chunking**: 800 chars default, 120 char overlap
- **Metadata preservation**: Section headings, paths, language tags
- **Chunk provenance**: Stable chunk_id, document offsets

### Embeddings
- **OpenAI backend**: text-embedding-3-large (configurable)
- **Dummy embedder**: Random vectors for testing/cost-free experiments
- **SQLite caching**: Persistent embedding cache to reduce API calls

### Vector Storage
- **JSONL store**: Human-readable chunks.jsonl + embeddings.jsonl
- **In-memory store**: For experiments & testing
- **Cosine similarity search**

### Reranking
- **Heuristic reranker**: Lexical overlap boost + diversity by doc_id
- **No-op reranker**: Baseline (vector similarity only)

### Context Building
- **Token budget enforcement**: Fits chunks within max tokens
- **Deduplication**: Removes duplicate chunks

### Generation
- **OpenAI integration**: GPT-4.1-mini (configurable)
- **Temperature control**: 0.2 default for consistency

### Query Tracing
- **Structured QueryTrace**: Documents every retrieval stage
- **JSONL persistence**: One query per line for analysis
- **Includes**: Retrieved/reranked chunk IDs, scores, latency

---

## Evaluation System

### Retrieval Metrics
- Recall@k, Precision@k, Hit rate@k
- Mean Reciprocal Rank (MRR)
- Average Precision (AP)
- NDCG@k

### Answer Quality (LLM-as-Judge)
- Correctness, Completeness, Relevance (0-5 scale)
- Hallucination detection
- Semantic similarity

### Aggregation
- Overall metrics
- Breakdowns by query type and difficulty
- Latency percentiles (p50, p95)

```bash
python -m experiments.run_eval \
  --queries experiments/eval_queries.jsonl \
  --run-generation \
  --use-llm-judge \
  --top-k 10 \
  --keep-k 4
```

---

## Getting Started

### Requirements

- Python >= 3.11
- Conda (recommended)

### Setup

```bash
conda env create -f environment.yml
conda activate rag-obsidian
pip install -e ".[openai]"  # or [ollama] for local models
```

If using OpenAI, create a `.env` file:

```bash
OPENAI_API_KEY='your-key-here'
```

### Build an Index

```bash
python scripts/build_index.py \
  --corpus ~/obsidian-vault \
  --index-name my_index
```

Or use dummy embeddings for testing:

```bash
python scripts/build_index.py \
  --corpus ~/obsidian-vault \
  --index-name my_index \
  --use-dummy-embeddings
```

### Query the System

```bash
python scripts/ask.py \
  --index my_index \
  --q "What is this project about?"
```

### Make Commands

```bash
make help          # Show available commands
make index         # Build index with OpenAI embeddings
make index-dummy   # Build index with dummy embeddings
make ask QUERY="your question here"
make tail-logs     # Inspect recent query logs
make clean-index   # Remove index (dangerous)
```

---

## Configuration

Configuration lives in `settings.toml`:

```toml
[paths]
vault_dir = "~/obsidian-vault"
artifacts_dir = "artifacts"

[chunking]
chunk_size = 800
overlap = 120

[embeddings]
backend = "openai"  # or "dummy"
model = "text-embedding-3-large"

[retrieval]
top_k = 8

[rerank]
enabled = true
backend = "heuristic"
keep_k = 4

[llm]
backend = "openai"
model = "gpt-4.1-mini"
temperature = 0.2
```

CLI flags can override settings for one-off experiments.

---

## Open Questions

- How far reranking scales before cost dominates
- When multi-hop or decomposed retrieval is actually worth it
- How to balance determinism with model-driven ranking

---

## Future Work

### Near-term
- Additional chunking strategies (semantic, header-aware)
- LLM-based reranking implementations
- Larger, more diverse evaluation datasets

### Longer-term
- Multi-hop retrieval
- Query decomposition
- Agent-style orchestration
