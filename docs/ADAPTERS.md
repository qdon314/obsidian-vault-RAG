# Adapters Reference

Complete documentation for all adapter implementations in the RAG system. Adapters are concrete implementations of the port interfaces.

## Table of Contents

- [Chunking Adapters](#chunking-adapters)
- [Embedding Adapters](#embedding-adapters)
- [Vector Store Adapters](#vector-store-adapters)
- [Retrieval Adapters](#retrieval-adapters)
- [Reranking Adapters](#reranking-adapters)
- [Context Building Adapters](#context-building-adapters)
- [Generation Adapters](#generation-adapters)
- [Ingestion Adapters](#ingestion-adapters)
- [Logging Adapters](#logging-adapters)

---

## Chunking Adapters

### FixedChunker

**Location:** `src/rag/adapters/chunking/fixed.py`

Character-based chunker with configurable size and overlap.

```python
@dataclass(frozen=True, slots=True)
class FixedChunker:
    chunk_size: int = 1200       # Characters per chunk
    overlap: int = 150           # Overlap between consecutive chunks
    strategy_name: str = "fixed_chars_v1"
```

**Behavior:**
- Splits document text into fixed-size character chunks
- Uses configurable overlap to preserve context across boundaries
- Generates stable chunk IDs: `{doc_id}:{strategy}:{index}:{start}-{end}`
- Preserves document metadata in each chunk
- Skips empty chunks

**Example:**
```python
from rag.adapters.chunking.fixed import FixedChunker
from rag.domain.models import Document

chunker = FixedChunker(chunk_size=800, overlap=120)
doc = Document(doc_id="doc1", text="...", source="filesystem", uri="/path/file.md")
chunks = chunker.chunk(doc)
```

**Chunk ID Format:**
```
doc123:fixed_chars_v1:0:0-800      # First chunk
doc123:fixed_chars_v1:1:680-1480   # Second chunk (with overlap)
```

### ObsidianStructuralChunker

**Location:** `src/rag/adapters/chunking/obsidian_structural.py`

Structure-aware chunker that respects markdown semantic boundaries.  Parses headings, lists, tables, callouts, and code blocks, then packs them into size-constrained chunks with optional overlap.

```python
@dataclass(frozen=True, slots=True)
class ObsidianStructuralChunker:
    target_chars: int = 4000         # Soft target chunk size
    hard_max_chars: int = 5200       # Hard maximum (only paragraphs split)
    overlap_blocks: int = 1          # Trailing blocks carried to next chunk
    include_heading_preamble: bool = True  # Prepend "Title: X\nPath: Y"
    strategy_name: str = "obsidian_structural_v1"
```

**Pipeline (shared parsing from `_markdown.py`):**
1. **Code block isolation** -- separates fenced code from markdown
2. **Section parsing** -- builds heading hierarchy
3. **Block detection** -- identifies para, list, callout, table, code blocks
4. **Chunk assembly** -- packs blocks into chunks respecting size constraints

**Example:**
```python
from rag.adapters.chunking.obsidian_structural import ObsidianStructuralChunker

chunker = ObsidianStructuralChunker(target_chars=2000, overlap_blocks=1)
chunks = chunker.chunk(doc)
```

### ObsidianPropositionChunker

**Location:** `src/rag/adapters/chunking/proposition/chunker.py`

Proposition-based chunker that decomposes documents into atomic, self-contained sentences using a seq2seq model.  Based on the [Dense X Retrieval](https://arxiv.org/abs/2312.06648) approach.

```python
@dataclass(frozen=True, slots=True)
class ObsidianPropositionChunker:
    propositionizer: Propositionizer    # seq2seq model wrapper
    passage_target_chars: int = 900     # Soft target for propositionizer input
    passage_hard_max_chars: int = 1400  # Hard max for propositionizer input
    include_heading_preamble: bool = False
    overlap_blocks: int = 0
    strategy_name: str = "obsidian_proposition_v1"
```

**Pipeline:**
1. Stages 1-3 from the shared `_markdown.py` parsing pipeline (same as structural chunker)
2. Code blocks emitted as chunks unchanged
3. Non-code blocks packed into passages (~900 chars)
4. Each passage fed to `Propositionizer` (batch inference)
5. Each returned proposition becomes one Chunk

**Chunk Metadata:**

Proposition chunks carry extra metadata for downstream expansion:

| Key | Description |
|-----|-------------|
| `chunk_kind` | Always `"proposition"` |
| `passage_index` | Index of parent passage within its section |
| `prop_index` | Index of this proposition within parent passage |
| `parent_passage_text` | Full text of the source passage |
| `parent_start_char` | Start offset of parent passage in document |
| `parent_end_char` | End offset of parent passage in document |

**Example:**
```python
from rag.adapters.chunking.proposition.backends import T5Propositionizer
from rag.adapters.chunking.proposition.chunker import ObsidianPropositionChunker

propositionizer = T5Propositionizer()  # loads HF model
chunker = ObsidianPropositionChunker(propositionizer=propositionizer)
chunks = chunker.chunk(doc)
# Each chunk.text is a single self-contained proposition
```

**Model:** `chentong00/propositionizer-wiki-flan-t5-large` (HuggingFace, runs on CPU or CUDA).

---

## Embedding Adapters

### OpenAIEmbedder

**Location:** `src/rag/adapters/embedding/openai_embedder.py`

Production embedder using OpenAI's embedding API.

```python
@dataclass(frozen=True, slots=True)
class OpenAIEmbedder:
    api_key: str
    model: str = "text-embedding-3-small"
```

**Features:**
- Uses official OpenAI Python client
- Returns vectors in same order as input texts
- Supports batch embedding

**Supported Models:**
| Model | Dimensions | Use Case |
|-------|------------|----------|
| `text-embedding-3-large` | 3072 | Highest quality |
| `text-embedding-3-small` | 1536 | Good balance of quality/cost |

**Example:**
```python
from rag.adapters.embedding.openai_embedder import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key="sk-...",
    model="text-embedding-3-large"
)
vectors = embedder.embed_texts(["Hello world", "Goodbye world"])
```

### DummyEmbedder

**Location:** `src/rag/adapters/embedding/dummy_embedder.py`

Random vector embedder for testing without API costs.

```python
@dataclass(frozen=True, slots=True)
class DummyEmbedder:
    dim: int = 128
```

**Use Cases:**
- Testing pipeline without API calls
- Development without incurring costs
- CI/CD environments

**Example:**
```python
from rag.adapters.embedding.dummy_embedder import DummyEmbedder

embedder = DummyEmbedder(dim=128)
vectors = embedder.embed_texts(["test text"])  # Returns random vectors
```

### CachedEmbedder

**Location:** `src/rag/adapters/embedding/sqlite_cache.py`

Caching wrapper that stores embeddings in SQLite.

**Features:**
- Wraps any embedder
- Caches text → vector mappings
- Reduces API calls for repeated texts

**Example:**
```python
from pathlib import Path

from rag.adapters.embedding.sqlite_cache import CachedEmbedder
from rag.adapters.embedding.openai_embedder import OpenAIEmbedder

base_embedder = OpenAIEmbedder(api_key="sk-...")
cached_embedder = CachedEmbedder(
    embedder=base_embedder,
    db_path=Path("./cache/embeddings.db")
)
```

---

## Vector Store Adapters

### JsonlVectorStore

**Location:** `src/rag/adapters/vectorstores/jsonl_store.py`

Disk-persisted store using JSONL format.

```python
@dataclass(slots=True)
class JsonlVectorStore:
    path: Path
```

**File Structure:**
```
{path}/
└── chunks.jsonl    # One JSON object per line
```

**JSONL Format:**
```json
{"chunk": {"chunk_id": "...", "doc_id": "...", "text": "..."}, "vector": [0.1, 0.2, ...]}
```

**Features:**
- Human-readable format for inspection
- Atomic save with temp file swap
- In-memory search with cosine similarity
- Filter support via `InMemoryFilterEvaluator`

**Methods:**
| Method | Description |
|--------|-------------|
| `load()` | Load chunks from `chunks.jsonl` into memory |
| `save()` | Persist in-memory data to `chunks.jsonl` |
| `upsert()` | Add chunks and vectors to memory |
| `search()` | Cosine similarity search |
| `count()` | Return number of stored chunks |

**Example:**
```python
from pathlib import Path
from rag.adapters.vectorstores.jsonl_store import JsonlVectorStore

store = JsonlVectorStore(path=Path("./artifacts/index"))
store.load()  # Load existing data
store.upsert(chunks=[chunk1, chunk2], vectors=[vec1, vec2])
store.save()  # Persist to disk
```

### InMemoryVectorStore

**Location:** `src/rag/adapters/vectorstores/in_memory_store.py`

Pure in-memory store without persistence.

**Use Cases:**
- Unit tests
- Experiments where persistence isn't needed
- Quick prototyping

**Example:**
```python
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore

store = InMemoryVectorStore()
store.upsert(chunks=[chunk], vectors=[vector])
results = store.search(query_vector=query_vec, top_k=5)
```

### QdrantVectorStore

**Location:** `src/rag/adapters/vectorstores/qdrant_store.py`

Qdrant-backed vector store for scalable similarity search. Supports both local and remote Qdrant instances.

```python
@dataclass(slots=True)
class QdrantVectorStore:
    collection_name: str           # Name of the Qdrant collection
    vector_size: int               # Dimension of vectors (must match embedder output)
    url: str | None = None         # Qdrant server URL (for remote)
    path: str | None = None        # Path for local disk persistence
    api_key: str | None = None     # API key for Qdrant Cloud
    distance: Distance = Distance.COSINE  # Distance metric
```

**Deployment Modes:**

| Mode | Configuration | Description |
|------|---------------|-------------|
| In-memory | No `url` or `path` | Fast, non-persistent (testing) |
| Local disk | Set `path` | Persistent local storage |
| Remote server | Set `url` | Connect to Qdrant server |
| Qdrant Cloud | Set `url` + `api_key` | Managed cloud service |

**Features:**
- Auto-creates collections with proper vector configuration
- Uses `QdrantFilterCompiler` for metadata filtering
- Supports cosine, Euclidean, and dot product distances
- Chunk IDs are hashed to UUIDs for Qdrant compatibility
- Uses `Chunk.to_dict()` / `Chunk.from_dict()` for serialization

**Methods:**
| Method | Description |
|--------|-------------|
| `upsert()` | Add chunks and vectors to collection |
| `search()` | Vector similarity search with optional filters |
| `count()` | Return number of stored points |
| `save()` | No-op (Qdrant handles persistence) |
| `load()` | Ensure collection exists |
| `clear()` | Delete all points in collection |
| `delete_collection()` | Delete the entire collection |

**Example:**
```python
from rag.adapters.vectorstores.qdrant_store import QdrantVectorStore

# In-memory mode (testing)
store = QdrantVectorStore(
    collection_name="chunks",
    vector_size=3072
)

# Local disk persistence
store = QdrantVectorStore(
    collection_name="chunks",
    vector_size=3072,
    path="./artifacts/qdrant"
)

# Remote Qdrant server
store = QdrantVectorStore(
    collection_name="chunks",
    vector_size=3072,
    url="http://localhost:6333"
)

# Qdrant Cloud
store = QdrantVectorStore(
    collection_name="chunks",
    vector_size=3072,
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key"
)

# Search with filters
from rag.domain.filters import Eq, And
results = store.search(
    query_vector=query_vec,
    top_k=10,
    where=And(clauses=[Eq(field="source", value="filesystem")])
)
```

**Installation:**
```bash
./scripts/pip install -e ".[qdrant]"
```

---

## Retrieval Adapters

### VectorRetriever

**Location:** `src/rag/adapters/retrieval/vector_retriever.py`

Composes an Embedder and VectorStore for pure vector similarity retrieval.

```python
@dataclass(frozen=True, slots=True)
class VectorRetriever:
    embedder: Embedder
    store: VectorStore
```

**Workflow:**
1. Embed the query text
2. Search the vector store with the query vector
3. Return top-k candidates

**Example:**
```python
from rag.adapters.retrieval.vector_retriever import VectorRetriever

retriever = VectorRetriever(embedder=embedder, store=store)
candidates = retriever.retrieve("What is X?", top_k=10)
```

### BM25Retriever

**Location:** `src/rag/adapters/retrieval/bm25_retriever.py`

Pure-Python BM25 keyword retriever with no external dependencies. Provides lexical matching for exact term retrieval.

```python
@dataclass
class BM25Retriever:
    k1: float = 1.5          # Term frequency saturation parameter
    b: float = 0.75          # Length normalization parameter
```

**Algorithm:**

BM25 scores documents based on term frequency and inverse document frequency:

```
score = Σ(idf(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d|/avgdl)))
```

Where:
- `f(t,d)` = frequency of term t in document d
- `|d|` = document length (in tokens)
- `avgdl` = average document length
- `idf(t)` = inverse document frequency of term t

**Methods:**

| Method | Description |
|--------|-------------|
| `index(chunks: list[Chunk])` | Build BM25 index from chunks |
| `retrieve(query, top_k, where)` | Score and rank chunks by BM25 |

**Example:**
```python
from rag.adapters.retrieval.bm25_retriever import BM25Retriever

bm25 = BM25Retriever(k1=1.5, b=0.75)
bm25.index(chunks)  # Build index from loaded chunks
candidates = bm25.retrieve("RAG architecture", top_k=10)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k1` | 1.5 | Term frequency saturation (higher = more saturation) |
| `b` | 0.75 | Length normalization (0-1, higher = more normalization) |

**Tokenization:**

Simple lowercase split tokenization. No stemming or stopword removal (intentionally simple to avoid dependencies).

**When to Use:**

| Use Case | Recommendation |
|----------|----------------|
| Exact term matching | ✓ Ideal |
| Acronyms | ✓ Ideal |
| Rare technical terms | ✓ Ideal |
| Semantic/conceptual queries | ✗ Use VectorRetriever |
| General natural language | △ Okay, but vector is usually better |

### HybridRetriever

**Location:** `src/rag/adapters/retrieval/hybrid_retriever.py`

Combines two retrievers (typically vector + BM25) using Reciprocal Rank Fusion (RRF). No external dependencies.

```python
@dataclass(frozen=True, slots=True)
class HybridRetriever:
    primary: Retriever       # Usually VectorRetriever
    secondary: Retriever     # Usually BM25Retriever
    primary_weight: float = 0.7
    secondary_weight: float = 0.3
    rrf_k: int = 60
```

**Workflow:**
1. Retrieve candidates from primary retriever (top_k * 2)
2. Retrieve candidates from secondary retriever (top_k * 2)
3. Fuse results using RRF: `score = Σ(weight / (k + rank))`
4. Return top-k fused candidates

**RRF Formula:**

```
RRF Score = Σ(weight / (k + rank))
```

Where:
- `weight` is `primary_weight` or `secondary_weight`
- `k` is the RRF constant (default 60)
- `rank` is the position in each result list (1-indexed)

Items appearing in both result lists receive higher fused scores.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `primary` | required | Primary retriever (usually vector) |
| `secondary` | required | Secondary retriever (usually BM25) |
| `primary_weight` | 0.7 | Weight for primary results |
| `secondary_weight` | 0.3 | Weight for secondary results |
| `rrf_k` | 60 | RRF constant (higher = more rank smoothing) |

**Example:**
```python
from rag.adapters.retrieval.hybrid_retriever import HybridRetriever
from rag.adapters.retrieval.vector_retriever import VectorRetriever
from rag.adapters.retrieval.bm25_retriever import BM25Retriever

# Build BM25 index
bm25 = BM25Retriever()
bm25.index(chunks)

# Create hybrid retriever
hybrid = HybridRetriever(
    primary=VectorRetriever(embedder=embedder, store=store),
    secondary=bm25,
    primary_weight=0.7,
    secondary_weight=0.3,
)
candidates = hybrid.retrieve("RAG architecture", top_k=10)
```

**When to Use Hybrid:**

| Query Type | Vector Only | Hybrid |
|------------|-------------|--------|
| Semantic/conceptual | ✓ Good | ✓ Good |
| Rare terms | ✗ Poor | ✓ Good |
| Acronyms | ✗ Poor | ✓ Good |
| Proper nouns | △ Okay | ✓ Good |
| Keyword-heavy | △ Okay | ✓ Good |

**Configuration:**

```toml
[retrieval]
backend = "hybrid"           # "vector" | "hybrid"
top_k = 8

[retrieval.hybrid]
primary_weight = 0.7
secondary_weight = 0.3
rrf_k = 60
bm25_k1 = 1.5
bm25_b = 0.75
```

---

## Reranking Adapters

### HeuristicReranker

**Location:** `src/rag/adapters/reranking/rerank_heuristic.py`

Cheap reranker using lexical overlap and diversification.

```python
class HeuristicReranker:
    def __init__(
        self,
        *,
        overlap_weight: float = 0.15,
        diversify: bool = True,
        max_per_doc: int = 3
    ):
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `overlap_weight` | 0.15 | Weight for lexical overlap boost |
| `diversify` | True | Enable document diversification |
| `max_per_doc` | 3 | Max chunks from same document |

**Algorithm:**
1. Start with vector similarity score
2. Calculate lexical overlap: `common_tokens / query_tokens`
3. New score = `base_score + overlap_weight * overlap`
4. If diversifying, limit chunks per document

**Example:**
```python
from rag.adapters.reranking.rerank_heuristic import HeuristicReranker

reranker = HeuristicReranker(overlap_weight=0.2, diversify=True, max_per_doc=2)
reranked = reranker.rerank("What is X?", candidates)
```

### NoOpReranker

**Location:** `src/rag/adapters/reranking/rerank_noop.py`

Pass-through reranker that returns candidates unchanged.

**Use Cases:**
- Baseline experiments (vector similarity only)
- When reranking is disabled

**Example:**
```python
from rag.adapters.reranking.rerank_noop import NoOpReranker

reranker = NoOpReranker()
# Returns candidates unchanged
reranked = reranker.rerank("query", candidates)
```

---

## Context Building Adapters

### SimpleContextBuilder

**Location:** `src/rag/adapters/context_building/simple_context_builder.py`

Builds context within token budget with deduplication.

```python
@dataclass(frozen=True, slots=True)
class SimpleContextBuilder:
    min_score: float | None = None    # Optional score threshold
    max_chunks: int = 12              # Max chunks to include
    dedupe: bool = True               # Remove near-duplicates
    include_scores: bool = False      # Show scores in context
```

**Algorithm:**
1. Sort candidates by `rerank_score` (or `score` if no reranking)
2. For each candidate:
   - Skip if below `min_score`
   - Skip if dedupe enabled and similar chunk already seen
   - Estimate tokens: `~4 chars/token`
   - Break if exceeds `token_budget`
   - Add to context and create citation
3. Render formatted context string

**Token Estimation:**
```python
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 chars per token
```

**Rendered Context Format:**
```
You are given CONTEXT chunks from a document corpus. Answer the QUESTION using only the CONTEXT.
If the answer is not supported by the CONTEXT, say you don't know.
CONTEXT:
[1]
Source: Document Title /path/to/file.md
Chunk text content here...

[2]
Source: Another Doc /path/to/other.md
More chunk content...
```

**Example:**
```python
from rag.adapters.context_building.simple_context_builder import SimpleContextBuilder

builder = SimpleContextBuilder(
    min_score=0.5,
    max_chunks=5,
    dedupe=True
)
context_pack = builder.build("What is X?", candidates, token_budget=1500)
```

### PropositionAwareContextBuilder

**Location:** `src/rag/adapters/context_building/propositional_context_builder.py`

Extends the context-building step to handle proposition chunks.  Expands short proposition text back to the parent passage so the LLM receives enough surrounding context.

```python
@dataclass(frozen=True, slots=True)
class PropositionAwareContextBuilder:
    min_score: float | None = None
    max_chunks: int = 12
    dedupe: bool = True
    include_scores: bool = False
    expand_propositions: bool = True      # Expand propositions to passages
    expansion_mode: str = "passage"       # "passage" | "none"
    include_prop_header: bool = True      # Show "Proposition: ..." before passage
```

**Expansion Behavior:**
- When `expand_propositions=True` and `expansion_mode="passage"`, proposition chunks are rendered as their parent passage text (stored in `chunk.metadata["parent_passage_text"]`).
- Optionally prefixed with `Proposition: <retrieved text>` so the LLM sees what was actually retrieved.
- Non-proposition chunks (code, para, list, etc.) are rendered as-is.

**Dual-Layer Deduplication:**
1. **Passage identity**: For expanded propositions, track `doc_id:start:end` of the parent passage.  Skip if the same passage was already included (prevents repeating a passage when multiple propositions from it are retrieved).
2. **Text signature**: Normalize and truncate the rendered text (800 chars).  Skip if already seen.

**Rendered Context Format (with `include_prop_header=True`):**
```
[1]
Source: My Note /vault/note.md
Proposition: Python was created by Guido van Rossum.

Passage:
Python is a programming language created by Guido van Rossum.
It was first released in 1991 and emphasizes code readability.
```

**Example:**
```python
from rag.adapters.context_building.propositional_context_builder import (
    PropositionAwareContextBuilder,
)

builder = PropositionAwareContextBuilder(
    expand_propositions=True,
    expansion_mode="passage",
    include_prop_header=True,
)
context_pack = builder.build("Who created Python?", candidates, token_budget=2000)
```

---

## Generation Adapters

### OpenAIChatGenerator

**Location:** `src/rag/adapters/generation/openai_chat.py`

OpenAI chat completions generator.

```python
@dataclass(frozen=True, slots=True)
class OpenAIChatGenerator:
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
```

**System Prompt:**
```
You are a precise assistant. Use only the provided CONTEXT.
If the answer cannot be found in the CONTEXT, say you don't know.
```

**User Prompt Format:**
```
{rendered_context}
QUESTION:
{query}

Answer clearly and cite chunk numbers like [1], [2] where relevant.
```

**Example:**
```python
from rag.adapters.generation.openai_chat import OpenAIChatGenerator

generator = OpenAIChatGenerator(
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.2
)
answer = generator.generate("What is X?", context_pack)
```

---

## Ingestion Adapters

### FilesystemIngestor

**Location:** `src/rag/adapters/ingestion/filesystem.py`

Walks directory tree and loads documents.

**Features:**
- Recursive directory traversal
- Skips hidden files (optional)
- Filters by file extension
- Delegates to format-specific loaders

**Configuration via Settings:**
```toml
[ingestion]
recursive = true
skip_hidden = true
allowed_extensions = [".md", ".txt"]
```

**Example:**
```python
from rag.adapters.ingestion.filesystem import FilesystemIngestor
from rag.adapters.ingestion.loaders.obsidian_markdown_loader import ObsidianMarkdownLoader
from rag.adapters.ingestion.loaders.text_loader import TextLoader

ingestor = FilesystemIngestor(
    text_loader=TextLoader(),
    markdown_loader=ObsidianMarkdownLoader(vault_dir=Path("~/vault"))
)
documents, report = ingestor.ingest(["/path/to/vault"])
```

### ObsidianMarkdownLoader

**Location:** `src/rag/adapters/ingestion/loaders/obsidian_markdown_loader.py`

Loads Markdown files with Obsidian-specific features.

**Features:**
- Expands transclusions/embeds (`![[filename]]` syntax)
- Preserves heading hierarchy
- Configurable embed depth limit

**Configuration:**
```toml
[ingestion]
expand_embeds = true
max_embed_depth = 4
```

### TextLoader

**Location:** `src/rag/adapters/ingestion/loaders/text_loader.py`

Simple plain text file loader.

---

## Logging Adapters

### JsonlQueryLogger

**Location:** `src/rag/adapters/logging/jsonl_logger.py`

Appends QueryTrace objects to JSONL file.

```python
@dataclass(frozen=True, slots=True)
class JsonlQueryLogger:
    path: Path
    redact_text: bool = False  # Optional text redaction
```

**Features:**
- Atomic append (POSIX-safe for single process)
- Optional text redaction for privacy
- Human-readable JSON format

**Output Location:**
```
artifacts/logs/traces.jsonl
```

**Example Log Entry:**
```json
{
  "trace_id": "abc123",
  "query": "What is X?",
  "created_at": "2024-01-15T10:30:00Z",
  "top_k": 8,
  "retrieved": [...],
  "reranked": [...],
  "packed_chunk_ids": ["chunk1", "chunk2"],
  "model": "gpt-4o-mini",
  "latency_ms": 1234,
  "answer": {...},
  "metadata": {
    "timing_ms": {
      "retrieval": 100,
      "rerank": 50,
      "context": 20,
      "generation": 1000,
      "total": 1234
    }
  }
}
```

---

## Filter Adapters

The filter system provides a backend-agnostic way to express metadata filters using an AST (Abstract Syntax Tree). Filter nodes are defined in `src/rag/domain/filters.py` and compiled to backend-specific representations.

### Filter AST

**Location:** `src/rag/domain/filters.py`

| Filter | Description | Example |
|--------|-------------|---------|
| `Eq(field, value)` | Exact equality | `Eq("source", "filesystem")` |
| `In(field, values)` | Membership check | `In("type", ["md", "txt"])` |
| `Contains(field, value)` | Array contains value | `Contains("tags", "python")` |
| `Prefix(field, prefix)` | String prefix match | `Prefix("uri", "/docs/")` |
| `Range(field, gte, lte, gt, lt)` | Range query | `Range("score", gte=0.5)` |
| `And(clauses)` | All clauses must match | `And([Eq(...), Eq(...)])` |
| `Or(clauses)` | Any clause must match | `Or([Eq(...), Eq(...)])` |
| `Not(clause)` | Negation | `Not(Eq("draft", True))` |

### InMemoryFilterEvaluator

**Location:** `src/rag/adapters/filters/inmemory_evaluator.py`

Evaluates filter AST against chunk metadata in Python. Used by `JsonlVectorStore` and `InMemoryVectorStore`.

```python
class InMemoryFilterEvaluator:
    def matches(self, where: Where, metadata: Mapping[str, object]) -> bool:
        ...
```

**Example:**
```python
from rag.domain.filters import Eq, And
from rag.adapters.filters.inmemory_evaluator import InMemoryFilterEvaluator

evaluator = InMemoryFilterEvaluator()
filter = And(clauses=[
    Eq(field="source", value="filesystem"),
    Eq(field="language", value="python")
])

# Check if metadata matches filter
matches = evaluator.matches(filter, {"source": "filesystem", "language": "python"})

# Used automatically by JsonlVectorStore during search
results = store.search(query_vector=vec, top_k=10, where=filter)
```

### QdrantFilterCompiler

**Location:** `src/rag/adapters/filters/qdrant_compiler.py`

Compiles filter AST to Qdrant's native filter format. Used by `QdrantVectorStore`.

```python
class QdrantFilterCompiler:
    def compile(self, where: Where) -> QdrantFilter | None:
        ...
```

**Filter Translation:**

| Filter AST | Qdrant Equivalent |
|------------|-------------------|
| `Eq(field, value)` | `FieldCondition(key=field, match=MatchValue(value=value))` |
| `In(field, values)` | `FieldCondition(key=field, match=MatchAny(any=values))` |
| `Contains(field, value)` | `FieldCondition(key=field, match=MatchValue(value=value))` |
| `Prefix(field, prefix)` | `FieldCondition(key=field, match=MatchText(text=prefix))` |
| `Range(field, ...)` | `FieldCondition(key=field, range=Range(...))` |
| `And(clauses)` | `Filter(must=[...])` |
| `Or(clauses)` | `Filter(should=[...])` |
| `Not(clause)` | `Filter(must_not=[...])` |

**Example:**
```python
from rag.domain.filters import Eq, And, Range
from rag.adapters.filters.qdrant_compiler import QdrantFilterCompiler

compiler = QdrantFilterCompiler()

# Compile a filter
filter_ast = And(clauses=[
    Eq(field="source", value="filesystem"),
    Range(field="score", gte=0.5, lte=1.0)
])
qdrant_filter = compiler.compile(filter_ast)

# Used automatically by QdrantVectorStore during search
results = store.search(query_vector=vec, top_k=10, where=filter_ast)
```

**Nested Boolean Logic:**

The compiler supports arbitrarily nested boolean expressions:
```python
from rag.domain.filters import Eq, And, Or, Not

complex_filter = And(clauses=[
    Or(clauses=[
        Eq(field="type", value="note"),
        Eq(field="type", value="article")
    ]),
    Not(clause=Eq(field="draft", value=True))
])
```

---

## Adapter Selection

The `Container` automatically selects adapters based on settings:

| Setting | Options | Adapter |
|---------|---------|---------|
| `embeddings.backend` | `"openai"` | `OpenAIEmbedder` |
| | `"dummy"` | `DummyEmbedder` |
| `vectorstore.backend` | `"jsonl"` | `JsonlVectorStore` |
| | `"memory"` | `InMemoryVectorStore` |
| | `"qdrant"` | `QdrantVectorStore` |
| `rerank.backend` | `"heuristic"` | `HeuristicReranker` |
| | `"noop"` | `NoOpReranker` |
| `rerank.enabled` | `false` | `NoOpReranker` |

**Vector Store Filter Adapters:**

| VectorStore | Filter Adapter |
|-------------|----------------|
| `JsonlVectorStore` | `InMemoryFilterEvaluator` |
| `InMemoryVectorStore` | `InMemoryFilterEvaluator` |
| `QdrantVectorStore` | `QdrantFilterCompiler` |

**Override via CLI:**
```bash
./scripts/py scripts/build_index.py --use-dummy-embeddings
./scripts/py scripts/ask.py --index my_index --q "What is X?" --use-dummy-embeddings
```
