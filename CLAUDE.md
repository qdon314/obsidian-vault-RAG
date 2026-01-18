# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install with specific extras
pip install -e ".[dev]"          # Development (pytest, ruff)
pip install -e ".[openai]"       # OpenAI embeddings/generation
pip install -e ".[qdrant]"       # Qdrant vector store
pip install -e ".[ui]"           # Streamlit evaluation UI

# Run tests
pytest                           # All tests
pytest tests/path/to/test.py    # Single file
pytest -k "test_name"           # Single test by name

# Linting & formatting
ruff check .                    # Check for issues
ruff format .                   # Auto-format
ruff check --fix .              # Auto-fix lint issues

# Build index and query
make index                      # Build with OpenAI embeddings
make index-dummy                # Build with dummy embeddings (no API cost)
make ask QUERY="your question"  # Query the index
make eval                       # Launch Streamlit evaluation app
```

## Architecture Overview

This is a **Hexagonal Architecture (Ports & Adapters)** RAG system:

### Core Layers

**Ports** (`src/rag/ports/`): Protocol-based interfaces (PEP 544 structural subtyping). Key protocols:
- `Chunker` → splits Documents into Chunks
- `Embedder` → text to vectors
- `VectorStore` → stores and searches vectors
- `Retriever` → query to Candidates (composes Embedder + VectorStore)
- `Reranker` → re-scores candidates
- `ContextBuilder` → packs candidates into prompt within token budget
- `Generator` → produces final answer

**Domain** (`src/rag/domain/`): Immutable data models (frozen dataclasses):
- `Document`, `Chunk`, `Candidate` - content objects
- `ContextPack`, `Answer`, `Citation` - output objects
- `QueryTrace` - complete observability record per query
- `Filter` hierarchy (`Eq`, `In`, `Contains`, `Range`, `And`, `Or`, `Not`) - abstract filter DSL

**Adapters** (`src/rag/adapters/`): Concrete implementations organized by responsibility:
- `chunking/`: fixed, obsidian_structural
- `embedding/`: openai, dummy
- `vectorstores/`: memory, jsonl, qdrant
- `filters/`: inmemory_evaluator, qdrant_compiler

**App** (`src/rag/app/`): Orchestration layer:
- `container.py` - Dependency injection via frozen `Container` dataclass
- `query_runner.py` - Full pipeline: retrieve → rerank → context → generate → trace
- `pipeline.py` - Simple composable functions

### Query Pipeline Flow

```
run_query():
  retriever.retrieve(query, top_k, where)
    → reranker.rerank(query, candidates)
      → context_builder.build(query, candidates, token_budget)
        → generator.generate(query, context)
          → QueryTrace logged with per-stage timing
```

### Filter System

Filters use a domain DSL that compiles to backend-specific formats:
- `InMemoryFilterEvaluator`: evaluates filters against metadata dicts
- `QdrantFilterCompiler`: translates to Qdrant query DSL

### Configuration

All configuration in `settings.toml`. Key sections:
- `[paths]` - vault_dir, artifacts_dir
- `[chunking]` - backend, chunk_size, overlap
- `[embeddings]` - backend (openai/dummy), model
- `[vectorstore]` - backend (memory/jsonl/qdrant)
- `[llm]` - model, temperature
- `[rerank]` - enabled, backend

CLI flags override settings for experiments.

## Key Conventions

- **All domain objects are frozen dataclasses** - use `dataclasses.replace()` for modifications
- **Protocols, not inheritance** - adapters implement via structural subtyping
- **Stable IDs** - doc_id and chunk_id are content-hashed for reproducibility
- **Metadata threading** - optional `metadata` parameter on port methods for tracing
- **Offset tracking** - chunks store `start_char`/`end_char` for precise sourcing

## Evaluation System

`src/rag/eval/` contains:
- `harness.py` - runs eval queries through retrieval and full pipeline
- `metrics.py` - recall@k, precision@k, NDCG, MRR, MAP
- `schema.py` - `EvalQuery`, `QueryType`, `Difficulty`

`eval/app/` contains the Streamlit query curation UI.
