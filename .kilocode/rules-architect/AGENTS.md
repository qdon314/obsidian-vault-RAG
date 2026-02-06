# AGENTS.md — Architect Mode

This file provides architectural guidance to agents when working with code in this repository.

## Critical Command Discipline

**Never run `python`, `pip`, `pytest`, `ruff`, or `streamlit` directly.**

Always use the pinned interpreter via:
- `make <target>` for common workflows
- `./scripts/py ...` for ad-hoc Python commands
- `./scripts/pip ...` for dependency management

## Running Tests

```bash
make test                              # Full test suite
./scripts/py -m pytest tests/foo.py    # Single test file
./scripts/py -m pytest -k test_name    # Filter by test name
```

## Architecture Pattern

**Hexagonal Architecture (Ports & Adapters)** - See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

### Ports (`src/rag/ports/`)

Protocol-based interfaces using PEP 544 structural subtyping:
- `Chunker` → splits Documents into Chunks
- `Embedder` → text to vectors
- `VectorStore` → stores and searches vectors
- `Retriever` → query to Candidates (composes Embedder + VectorStore)
- `Reranker` → re-scores candidates
- `ContextBuilder` → packs candidates into prompt within token budget
- `Generator` → produces final answer

### Domain (`src/rag/domain/`)

Immutable frozen dataclasses:
- `Document`, `Chunk`, `Candidate` - content objects
- `ContextPack`, `Answer`, `Citation` - output objects
- `QueryTrace` - complete observability record per query
- `Filter` hierarchy (`Eq`, `In`, `Contains`, `Range`, `And`, `Or`, `Not`) - abstract filter DSL

### Adapters (`src/rag/adapters/`)

Concrete implementations organized by responsibility:
- `chunking/`: fixed, obsidian_structural, proposition
- `embedding/`: openai, dummy
- `vectorstores/`: memory, jsonl, qdrant
- `filters/`: inmemory_evaluator, qdrant_compiler

### App (`src/rag/app/`)

Orchestration layer:
- [`Container`](src/rag/app/container.py:38) - Dependency injection via frozen dataclass
- [`run_query()`](src/rag/app/query_runner.py:15) - Full pipeline: retrieve → rerank → context → generate → trace

## Key Architectural Conventions

- **Domain objects are frozen dataclasses** - use `dataclasses.replace()` for modifications
- **Protocols, not inheritance** - adapters implement via structural subtyping
- **Stable IDs** - `doc_id` and `chunk_id` are content-hashed for reproducibility
- **Metadata threading** - optional `metadata` parameter on port methods for tracing
- **Offset tracking** - chunks store `start_char`/`end_char` for precise sourcing

## Configuration

All configuration in [`settings.toml`](settings.toml). CLI flags override settings for experiments.

Key sections:
- `[paths]` - vault_dir, artifacts_dir
- `[chunking]` - backend, chunk_size, overlap
- `[embeddings]` - backend (openai/dummy), model
- `[vectorstore]` - backend (memory/jsonl/qdrant)
- `[llm]` - model, temperature
- `[rerank]` - enabled, backend

## Code Style

- Ruff: target Python 3.11, line-length 100, double quotes
- MyPy: `mypy_path = ["src"]` with explicit package bases
- Tests: fixtures in [`tests/conftest.py`](tests/conftest.py), naming checks softened

## Git

- Do NOT auto-commit
- Do NOT add "Co-authored-by" attribution
- Provide suggested commits at end of work
