# Architecture Documentation

This document provides a comprehensive overview of the RAG system architecture, including component relationships, data flows, and design patterns.

## Table of Contents

- [Overview](#overview)
- [Architecture Pattern](#architecture-pattern)
- [System Architecture Diagrams](#system-architecture-diagrams)
- [Data Flow](#data-flow)
- [Component Responsibilities](#component-responsibilities)
- [Design Principles](#design-principles)

---

## Overview

The Obsidian Vault RAG system implements a **Retrieval-Augmented Generation** pipeline with a focus on observability and evaluation. The architecture follows the **Hexagonal (Ports & Adapters)** pattern, enabling clean separation of concerns and easy component swapping.

### Key Architectural Goals

1. **Observability First**: Every query generates a complete trace with all intermediate results
2. **Evaluation as First-Class**: Built-in evaluation framework with comprehensive metrics
3. **Clean Boundaries**: Protocol-based interfaces allow easy swapping of implementations
4. **Reproducibility**: Stable IDs for documents and chunks enable deterministic behavior

---

## Architecture Pattern

### Hexagonal Architecture (Ports & Adapters)

```mermaid
graph TB
    App["Application Layer<br/>(CLI, Scripts, Streamlit UI)"]
    DI["Dependency Injection<br/>Container (app/container.py)<br/>Composes Ports with Adapter Implementations"]

    App --> DI
    DI --> Ports & Domain & Adapters

    subgraph Ports ["PORTS (Abstract Interfaces)"]
        direction TB
        Chunker
        Embedder
        Retriever
        Generator
        VectorStore
    end

    subgraph Domain ["DOMAIN MODELS"]
        direction TB
        Document
        Chunk
        Candidate
        Answer
        QueryTrace
    end

    subgraph Adapters ["ADAPTERS (Concrete Implementations)"]
        direction TB
        OpenAI
        JSONL_Store["JSONL Store"]
        Heuristic_Reranker["Heuristic Reranker"]
    end
```

### Benefits

- **Testability**: Protocol-based interfaces allow easy mocking
- **Flexibility**: Swap OpenAI for local models without code changes
- **Clarity**: Clear separation between "what" (ports) and "how" (adapters)

---

## System Architecture Diagrams

### High-Level System Overview

```mermaid
graph TB
    subgraph "Input Sources"
        Vault[Obsidian Vault]
        Files[Text Files]
    end

    subgraph "Ingestion Pipeline"
        Ingestor[FilesystemIngestor]
        Loaders[Document Loaders]
        Chunker[FixedChunker]
    end

    subgraph "Embedding & Storage"
        Embedder[Embedder]
        VectorStore[VectorStore]
    end

    subgraph "Query Pipeline"
        Retriever[Retriever]
        Reranker[Reranker]
        ContextBuilder[ContextBuilder]
        Generator[Generator]
    end

    subgraph "Observability"
        Logger[QueryLogger]
        Traces[Query Traces]
    end

    subgraph "Output"
        Answer[Answer + Citations]
    end

    Vault --> Ingestor
    Files --> Ingestor
    Ingestor --> Loaders
    Loaders --> Chunker
    Chunker --> Embedder
    Embedder --> VectorStore

    VectorStore --> Retriever
    Retriever --> Reranker
    Reranker --> ContextBuilder
    ContextBuilder --> Generator
    Generator --> Answer
    Generator --> Logger
    Logger --> Traces
```

### Indexing Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as build_index.py
    participant Ingestor as FilesystemIngestor
    participant Loader as ObsidianMarkdownLoader
    participant Chunker as FixedChunker
    participant Embedder as OpenAIEmbedder
    participant Store as JsonlVectorStore

    User->>CLI: ./scripts/py scripts/build_index.py --corpus ~/vault
    CLI->>Ingestor: ingest(vault_path)

    loop For each file
        Ingestor->>Loader: load(file_path)
        Loader-->>Ingestor: Document
    end

    Ingestor-->>CLI: List[Document], IngestReport

    loop For each document
        CLI->>Chunker: chunk(document)
        Chunker-->>CLI: List[Chunk]
        CLI->>Embedder: embed_texts([chunk.text])
        Embedder-->>CLI: List[Vector]
        CLI->>Store: upsert(chunks, vectors)
    end

    CLI->>Store: save()
    Store-->>CLI: chunks.jsonl written
    CLI-->>User: Index built: N documents, M chunks
```

### Query Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as ask.py
    participant Runner as run_query()
    participant Retriever as VectorRetriever
    participant Embedder as OpenAIEmbedder
    participant Store as VectorStore
    participant Reranker as HeuristicReranker
    participant Builder as SimpleContextBuilder
    participant Generator as OpenAIChatGenerator
    participant Logger as JsonlQueryLogger

    User->>CLI: ./scripts/py scripts/ask.py --index my_index --q "What is X?"
    CLI->>Runner: run_query(query, ...)

    Runner->>Retriever: retrieve(query, top_k=8)
    Retriever->>Embedder: embed_texts([query])
    Embedder-->>Retriever: query_vector
    Retriever->>Store: search(query_vector, top_k=8)
    Store-->>Retriever: List[Candidate]
    Retriever-->>Runner: candidates (8)

    Runner->>Reranker: rerank(query, candidates)
    Reranker-->>Runner: reranked_candidates
    Note over Runner: Truncate to keep_k=4

    Runner->>Builder: build(query, candidates, token_budget)
    Builder-->>Runner: ContextPack

    Runner->>Generator: generate(query, context)
    Generator-->>Runner: Answer

    Runner->>Logger: log(QueryTrace)
    Logger-->>Runner: Appended to traces.jsonl

    Runner-->>CLI: QueryRunResult
    CLI-->>User: Answer + Citations
```

### Container Composition

```mermaid
graph LR
    subgraph "Container"
        direction TB
        C[Container]
    end

    subgraph "Ports (Interfaces)"
        Chunker[Chunker]
        Embedder[Embedder]
        VectorStore[VectorStore]
        Retriever[Retriever]
        Reranker[Reranker]
        ContextBuilder[ContextBuilder]
        Generator[Generator]
        Logger[QueryLogger]
    end

    subgraph "Adapters (Implementations)"
        FC[FixedChunker]
        OAE[OpenAIEmbedder]
        DE[DummyEmbedder]
        JSONL[JsonlVectorStore]
        MEM[InMemoryVectorStore]
        VR[VectorRetriever]
        HR[HeuristicReranker]
        NR[NoOpReranker]
        SCB[SimpleContextBuilder]
        OAG[OpenAIChatGenerator]
        JQL[JsonlQueryLogger]
    end

    C --> Chunker
    C --> Embedder
    C --> VectorStore
    C --> Retriever
    C --> Reranker
    C --> ContextBuilder
    C --> Generator
    C --> Logger

    Chunker -.-> FC
    Embedder -.-> OAE
    Embedder -.-> DE
    VectorStore -.-> JSONL
    VectorStore -.-> MEM
    Retriever -.-> VR
    Reranker -.-> HR
    Reranker -.-> NR
    ContextBuilder -.-> SCB
    Generator -.-> OAG
    Logger -.-> JQL
```

---

## Data Flow

### Document Processing Pipeline

```mermaid
graph TB
    Raw["Raw File (*.md, *.txt)"]
    Doc["<strong>Document</strong><br/>- doc_id (hash-based, stable)<br/>- text (full content)<br/>- source ('filesystem')<br/>- uri (file path)<br/>- metadata"]
    Chunk["<strong>Chunk</strong><br/>- chunk_id (doc_id:strategy:idx)<br/>- doc_id (reference)<br/>- text (chunk content)<br/>- chunk_index, start_char, end_char<br/>- section_heading, section_path<br/>- metadata"]
    Vec["<strong>Vector</strong><br/>- list[float] (e.g., 3072 dims)"]
    Store["<strong>VectorStore (JSONL / In-Memory)</strong><br/>- Chunk + Vector pairs<br/>- Cosine similarity search"]

    Raw --> Doc
    Doc -- "Chunking (800 chars, 120 overlap)" --> Chunk
    Chunk -- "Embedding (OpenAI / Dummy)" --> Vec
    Vec -- "Storage" --> Store
```

### Query Processing Pipeline

```mermaid
graph TB
    Query["User Query: 'What is X?'"]
    QVec["Query Vector"]
    Cand["<strong>Candidates</strong><br/>- chunk: Chunk<br/>- score: float (similarity)<br/>- rerank_score: float | None"]
    Reranked["Reranked Candidates (keep_k=4)"]
    Context["<strong>ContextPack</strong><br/>- query<br/>- chunks: List[Chunk]<br/>- rendered_context: str<br/>- citations: List[Citation]<br/>- token_budget"]
    Ans["<strong>Answer</strong><br/>- query<br/>- text (generated response)<br/>- citations<br/>- confidence"]

    Query -- "Embedding" --> QVec
    QVec -- "Vector Search (top_k=8)" --> Cand
    Cand -- "Reranking (lexical boost + diversity)" --> Reranked
    Reranked -- "Context Building (token_budget=1500)" --> Context
    Context -- "LLM Generation" --> Ans
```

---

## Component Responsibilities

### Domain Models

| Model | Responsibility |
|-------|---------------|
| `Document` | Raw source unit before chunking; stable ID from content hash |
| `Chunk` | Embedding unit with provenance (offsets, section info) |
| `Candidate` | Retrieved chunk + retrieval/rerank scores |
| `Citation` | Source pointer for answer attribution |
| `ContextPack` | Final evidence bundle for generator |
| `Answer` | LLM output with citations and optional confidence metadata |
| `QueryTrace` | Complete observability record for debugging/evaluation |

### Ports (Interfaces)

| Port | Methods | Purpose |
|------|---------|---------|
| `Ingestor` | `ingest()` | Convert raw inputs to Documents |
| `Chunker` | `chunk()` | Split Document into Chunks |
| `Embedder` | `embed_texts()` | Convert text to dense vectors |
| `VectorStore` | `upsert()`, `search()`, `save()`, `load()` | Store and search embeddings |
| `Retriever` | `retrieve()` | Fetch candidate chunks for query |
| `Reranker` | `rerank()` | Re-order candidates by relevance |
| `ContextBuilder` | `build()` | Pack candidates into prompt context |
| `Generator` | `generate()` | Produce answer from context |
| `QueryLogger` | `log()` | Persist query traces |

### Adapters (Implementations)

| Adapter | Port | Description |
|---------|------|-------------|
| `FilesystemIngestor` | `Ingestor` | Walks directory tree, delegates to loaders |
| `ObsidianMarkdownLoader` | - | Loads .md with transclusion expansion |
| `TextLoader` | - | Loads plain .txt files |
| `FixedChunker` | `Chunker` | Character-based chunking (800/120) |
| `ObsidianStructuralChunker` | `Chunker` | Markdown-aware structural chunking |
| `ObsidianPropositionChunker` | `Chunker` | Proposition-based chunking (seq2seq) |
| `OpenAIEmbedder` | `Embedder` | OpenAI text-embedding-3-large |
| `DummyEmbedder` | `Embedder` | Random vectors for testing |
| `JsonlVectorStore` | `VectorStore` | JSONL-persisted, in-memory search |
| `InMemoryVectorStore` | `VectorStore` | Pure in-memory (no persistence) |
| `BM25Retriever` | `Retriever` | Keyword based search |
| `VectorRetriever` | `Retriever` | Pure vector similarity search |
| `HybridRetriever` | `Retriever` | Vector + keyword search with RRF fusion |
| `HeuristicReranker` | `Reranker` | Lexical overlap boost + diversity |
| `NoOpReranker` | `Reranker` | Pass-through (baseline) |
| `SimpleContextBuilder` | `ContextBuilder` | Token budget + deduplication |
| `PropositionAwareContextBuilder` | `ContextBuilder` | Proposition expansion + deduplication |
| `OpenAIChatGenerator` | `Generator` | GPT-4.1-mini chat completions |
| `JsonlQueryLogger` | `QueryLogger` | JSONL append logging |

---

## Design Principles

### 1. Stable Identifiers

Documents and chunks have deterministic IDs based on content:
- `doc_id`: Hash of (source + path + content)
- `chunk_id`: `{doc_id}:{strategy}:{index}:{start}-{end}`

This enables reproducible experiments and cache-friendly operations.

### 2. Metadata Preservation

Information flows through the pipeline without loss:
- Document metadata → Chunk metadata → Citation metadata
- Section headings and paths preserved for context

### 3. Token Budget Enforcement

Context building explicitly respects LLM context windows:
- Simple heuristic: ~4 characters per token
- Stops adding chunks when budget exceeded
- Deduplication prevents redundant content

### 4. Observability

Every query generates a `QueryTrace` containing:
- All retrieved candidates with scores
- All reranked candidates with scores
- Packed chunk IDs (what went to LLM)
- Timing breakdown by stage
- Final answer with citations

### 5. Composability

Small, focused adapters composed via the `Container`:
- Each adapter does one thing well
- Container wires dependencies
- CLI can override settings for experiments

### 6. Evaluation as First-Class

Built-in evaluation framework with:
- `EvalQuery` schema for ground truth
- Retrieval metrics (Recall, Precision, MRR, NDCG)
- Answer quality metrics (LLM-as-judge)
- Breakdowns by query type and difficulty

---

## File Organization

```
src/rag/
├── domain/              # Core data models
│   ├── models.py        # Document, Chunk, Candidate, Answer, etc.
│   └── filters.py       # Filter AST for metadata queries
│
├── ports/               # Abstract interfaces (Protocol classes)
│   ├── chunker.py       # Chunker protocol
│   ├── embedder.py      # Embedder protocol
│   ├── retriever.py     # Retriever protocol
│   ├── reranker.py      # Reranker protocol
│   ├── context_builder.py
│   ├── generator.py
│   ├── vector_store.py
│   ├── logger.py
│   └── ...
│
├── adapters/            # Concrete implementations
│   ├── chunking/        # Fixed, ObsidianStructural, ObsidianProposition
│   │   └── _markdown.py # Shared markdown parsing infrastructure
│   ├── embedding/       # OpenAI, Dummy, SQLite cache
│   ├── generation/      # OpenAI chat
│   ├── ingestion/       # Filesystem, loaders
│   ├── retrieval/       # VectorRetriever
│   ├── reranking/       # Heuristic, NoOp
│   ├── vectorstores/    # JSONL, in-memory
│   ├── context_building/ # Simple, PropositionAware
│   │   └── _shared.py   # Shared utilities (token estimation, dedupe)
│   ├── logging/
│   └── filters/
│
├── app/                 # Application orchestration
│   ├── container.py     # Dependency injection
│   ├── pipeline.py      # Core functions
│   └── query_runner.py  # Full pipeline with tracing
│
├── eval/                # Evaluation framework
│   ├── schema.py        # EvalQuery, QueryType, Difficulty
│   └── models.py        # EvalResult, RetrievalSummary
│
└── settings.py          # Configuration loading
```

---

## Configuration Flow

```mermaid
graph TB
    Settings["settings.toml (defaults)"]
    Load["Settings.load_settings()"]
    CLI["CLI Arguments (overrides)"]
    Overrides["ContainerOverrides"]
    Build["build_container(cfg, overrides)"]
    Container["Container (fully wired)"]
    Run["run_query() / index_document()"]

    Settings --> Load --> CLI --> Overrides --> Build --> Container --> Run
```

The configuration system allows:
1. Sensible defaults in `settings.toml`
2. CLI overrides for one-off experiments
3. Environment variables for secrets (`OPENAI_API_KEY`)
