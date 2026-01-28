# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment (IMPORTANT)

This repository **does not rely on environment activation**.

All Python commands **must be run via the pinned interpreter**, using one of:

- `make <target>` (preferred for common workflows)
- `./scripts/py ...` (for ad-hoc Python commands)
- `./scripts/pip ...` (for dependency management)

**Never run** `python`, `pip`, `pytest`, `ruff`, or `streamlit` directly.

This ensures commands always run in the correct environment, including in
non-interactive shells used by Claude Code.

## Build & Test Commands

```bash
# Install project with extras (always use scripts/pip)
./scripts/pip install -e ".[dev]"      # pytest, ruff, mypy
./scripts/pip install -e ".[openai]"   # OpenAI embeddings / generation
./scripts/pip install -e ".[qdrant]"   # Qdrant vector store
./scripts/pip install -e ".[ui]"       # Streamlit evaluation UI

# Run tests
make test                              # Full test suite
./scripts/py -m pytest                 # Equivalent (ad-hoc)
./scripts/py -m pytest tests/foo.py    # Single test file
./scripts/py -m pytest -k test_name    # Filter by test name

# Linting & formatting
make lint
make fmt

# Ad-hoc
./scripts/py -m ruff check .
./scripts/py -m ruff format .
./scripts/py -m ruff check --fix .

make typecheck
./scripts/py -m mypy rag

# Build index and query
make index                             # Build with OpenAI embeddings
make index-dummy                       # Build with dummy embeddings
make ask QUERY="your question"         # Query the index
make results                           # Launch Streamlit evaluation UI

# Environment Sanity Check
make env-check
./scripts/py -c "import sys; print(sys.executable)"
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
