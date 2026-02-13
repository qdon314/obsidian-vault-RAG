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
| `ChunkStore` | `get_chunks()`, `store_chunks()`, `list_all_chunk_ids()` | ID-based chunk content storage (distributed mode) |
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
| `HydratingRetriever` | `Retriever` | Wrapper: hydrates thin chunks from ChunkStore |
| `S3ChunkStore` | `ChunkStore` | S3 JSONL shards + Postgres index |
| `HeuristicReranker` | `Reranker` | Lexical overlap boost + diversity |
| `NoOpReranker` | `Reranker` | Pass-through (baseline) |
| `SimpleContextBuilder` | `ContextBuilder` | Token budget + deduplication |
| `PropositionAwareContextBuilder` | `ContextBuilder` | Proposition expansion + deduplication |
| `OpenAIChatGenerator` | `Generator` | GPT-4.1-mini chat completions |
| `JsonlQueryLogger` | `QueryLogger` | JSONL append logging |

---

## Distributed Chunk Storage

When operating in distributed mode (Qdrant remote + S3 + Postgres), the system separates chunk content storage from vector search. This enables thin Qdrant payloads (smaller index, faster search) while maintaining full chunk hydration at query time.

### Distributed Query Flow

```mermaid
sequenceDiagram
    participant Client
    participant HR as HydratingRetriever
    participant VR as VectorRetriever
    participant Qdrant as Qdrant (thin payloads)
    participant CS as S3ChunkStore
    participant PG as Postgres
    participant S3

    Client->>HR: retrieve(query, top_k=8)
    HR->>VR: retrieve(query, top_k=8)
    VR->>Qdrant: search(query_vector, top_k=8)
    Qdrant-->>VR: Candidates (IDs + scores, no text)
    VR-->>HR: Candidates with empty text

    Note over HR: Detect empty text → needs hydration

    HR->>CS: get_chunks([chunk_ids])
    CS->>PG: SELECT s3_key, line_offset WHERE chunk_id IN (...)
    PG-->>CS: (s3_key, offset) rows
    CS->>S3: GET shards (parallel, grouped by key)
    S3-->>CS: JSONL shard contents
    CS-->>HR: {chunk_id: Chunk} with full text

    Note over HR: Replace stubs with hydrated chunks

    HR-->>Client: Candidates with full text
```

### Distributed Indexing Flow (Dual-Write)

```mermaid
sequenceDiagram
    participant Pipeline as index_documents()
    participant Chunker
    participant Embedder
    participant Qdrant as Qdrant (thin payloads)
    participant CS as S3ChunkStore
    participant S3
    participant PG as Postgres

    Pipeline->>Chunker: chunk(doc)
    Chunker-->>Pipeline: chunks

    Pipeline->>Embedder: embed_texts(chunk_texts)
    Embedder-->>Pipeline: vectors

    par Dual-write
        Pipeline->>Qdrant: upsert(chunks, vectors) [thin payload: IDs + metadata only]
        Pipeline->>CS: store_chunks(chunks)
        CS->>S3: PUT shard JSONL (one per doc_id)
        CS->>PG: UPSERT chunk_index rows
    end
```

### Component Roles

| Component | Role |
|-----------|------|
| `ChunkStore` (port) | ID-based chunk storage/retrieval protocol |
| `S3ChunkStore` (adapter) | S3 JSONL shards + Postgres index |
| `QdrantVectorStore.thin_payloads` | Store only IDs + filterable metadata, no text |
| `HydratingRetriever` (wrapper) | Wraps any Retriever; batch-hydrates empty-text chunks |
| `ChunkStorage` (settings) | Configuration section for distributed chunk storage |

### S3 Shard Layout

```
s3://{bucket}/{prefix}/shards/{doc_id_hash[:4]}/{doc_id_hash}.jsonl
```

Each shard is a JSONL file containing one `Chunk.to_dict()` per line, grouped by document. The 4-character hash prefix provides balanced distribution across S3 partition prefixes.

### Postgres Index Schema

```sql
CREATE TABLE chunk_index (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    s3_key      TEXT NOT NULL,
    line_offset INT  NOT NULL
);
CREATE INDEX idx_chunk_index_doc_id ON chunk_index(doc_id);
CREATE INDEX idx_chunk_index_s3_key ON chunk_index(s3_key);
```

### Configuration

```toml
[chunk_storage]
backend = "s3"                                     # "none" | "s3"
s3_bucket = "my-rag-chunks"
s3_prefix = "obsidian"
postgres_dsn = "postgresql://user:pass@host:5432/rag"
max_s3_workers = 4                                  # parallel S3 fetch threads
```

When `backend = "none"` (default), the system operates in local mode with fat Qdrant payloads or JSONL storage. No code paths change.

---

## Distributed Ingestion

The distributed ingestion system enables scalable, parallel document processing using S3 as the corpus-of-record, SQS for task distribution, and Postgres for job/task state.

### Architecture

```mermaid
graph TB
    subgraph "Control Plane"
        CLI[CLI: start_ingestion.py]
        Enum[Enumerator Service]
    end

    subgraph "AWS Infrastructure"
        S3[S3 Bucket<br/>Corpus of Record]
        SQS[SQS Queue<br/>Task Distribution]
        RDS[(RDS Postgres<br/>Job/Task State)]
    end

    subgraph "Worker Fleet"
        W1[Worker 1]
        W2[Worker 2]
        W3[Worker N...]
    end

    subgraph "Vector Store"
        VS[VectorStore<br/>Qdrant/JSONL]
    end

    CLI -->|1. Enumerate corpus| Enum
    Enum -->|2. Store raw docs| S3
    Enum -->|3. Upsert docs/tasks| RDS
    Enum -->|4. Enqueue messages| SQS
    W1 -->|5. Receive + ack/nack| SQS
    W1 -->|6. Lease/complete/fail task| RDS
    W1 -->|7. Fetch raw doc| S3
    W1 -->|8. Chunk/embed/upsert| VS
    W2 -->|5. Receive + ack/nack| SQS
    W3 -->|5. Receive + ack/nack| SQS
```

### Flow

1. **Enumerator** (control plane entry point):
   - Creates an `IngestJob` in Postgres (`CREATED`)
   - Stores each raw `Document` in S3 via `RawDocumentStore`
   - Upserts a `DocumentRecord` and creates one `IngestTask` per `doc_id`
   - Enqueues one SQS message per document: `{"job_id","corpus_id","doc_id"}`
   - Marks the job `RUNNING` (or `COMPLETED` when no docs)

2. **Workers** (parallel processing):
   - Poll SQS via `TaskQueue.receive()`
   - Resolve `DocumentRecord` from Postgres, then lease task by `job_id + doc_id`
   - Process the raw document from S3: chunk -> embed -> vector upsert -> chunk-store write
   - Mark task `SUCCEEDED` or `RETRYABLE` in Postgres, then `ack`/`nack` SQS

### Domain Models

| Model | Purpose |
|-------|---------|
| `IngestJob` | Top-level job state for a corpus indexing run |
| `IngestTask` | Single-document task with lease/attempt/error fields |
| `DocumentRecord` | Corpus-of-record pointer (`s3_raw_key`) and content hash |
| `JobStatus` | `CREATED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED` |
| `TaskStatus` | `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `RETRYABLE` |

### Ports

| Port | Methods | Purpose |
|------|---------|---------|
| `IngestJobStore` | `create_job()`, `create_tasks()`, `acquire_task()`, `complete_task()`, `fail_task()`, `upsert_document()` | Postgres-backed job/task/document persistence |
| `RawDocumentStore` | `store_document()`, `get_document()` | Raw corpus storage and retrieval |
| `TaskQueue` | `send()`, `send_batch()`, `receive()`, `ack()`, `nack()` | SQS task distribution lifecycle |

### Message and Lease Contract

- SQS message body must include: `job_id`, `corpus_id`, `doc_id`.
- Worker must lease by `job_id` and `doc_id` to avoid cross-document task mismatch.
- Lease reclaim behavior:
  - claimable: `PENDING`, `RETRYABLE`
  - reclaimable: `RUNNING` with expired `lease_expires_at`
- Queue outcome:
  - `ack`: successful processing, or message references already-completed work
  - `nack`: missing document record, task failure, or lease mismatch

### Configuration

```toml
[distributed_ingestion]
enabled = true
postgres_dsn = "postgresql://user:pass@host:5432/rag"
sqs_queue_url = "https://sqs.us-east-1.amazonaws.com/123/rag-tasks"
corpus_s3_bucket = "rag-prod-artifacts"
corpus_s3_prefix = "corpus"
worker_lease_duration_s = 300
max_task_retries = 3
```

### CLI Usage

```bash
# Start a new ingestion job (enumerator from local corpus path)
./scripts/py scripts/start_ingestion.py \
  --corpus /path/to/corpus \
  --corpus-id my-corpus \
  --index-name my-index

# Run a worker (typically ECS/Fargate in production)
./scripts/py scripts/run_worker.py --worker-id worker-1
```

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
│   ├── chunk_storage/   # S3ChunkStore
│   ├── retrieval/       # VectorRetriever, HydratingRetriever
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
