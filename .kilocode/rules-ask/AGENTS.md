# AGENTS.md — Ask Mode

This file provides guidance to agents when answering questions about this repository.

## Project Overview

This is a **Hexagonal Architecture (Ports & Adapters)** RAG system for Obsidian vaults with:
- Protocol-based interfaces (PEP 544 structural subtyping)
- Immutable frozen dataclasses for domain models
- Stable content-hashed IDs for reproducibility
- Built-in evaluation framework with comprehensive metrics

## Key Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Comprehensive architecture overview
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) - API documentation
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) - User-facing documentation
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) - Configuration reference
- [`settings.toml`](settings.toml) - Canonical configuration file

## Architecture Layers

**Ports** (`src/rag/ports/`): Abstract interfaces
- `Chunker`, `Embedder`, `Retriever`, `Generator`, `VectorStore`, etc.

**Domain** (`src/rag/domain/`): Immutable data models
- `Document`, `Chunk`, `Candidate`, `QueryTrace`, etc.

**Adapters** (`src/rag/adapters/`): Concrete implementations
- Chunking, embedding, vector stores, reranking, etc.

**App** (`src/rag/app/`): Orchestration
- [`Container`](src/rag/app/container.py:38) for dependency injection
- [`run_query()`](src/rag/app/query_runner.py:15) for full pipeline

## Evaluation System

- `src/rag/eval/` - Evaluation harness and metrics
- `eval/app/` - Streamlit results analyzer UI
- `eval/datasets/` - Query datasets (JSONL format)
