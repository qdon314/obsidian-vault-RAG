# API Reference

Complete reference documentation for all ports (interfaces), domain models, and adapters in the Regulatory Corpus RAG system.

## Table of Contents

- [Domain Models](#domain-models)
- [Regulatory Domain Models](#regulatory-domain-models)
- [Ports (Interfaces)](#ports-interfaces)
- [Filter System](#filter-system)
- [Evaluation Schema](#evaluation-schema)

---

## Domain Models

All domain models are immutable dataclasses located in `src/rag/domain/models.py`.

### Document

A raw source unit before chunking.

```python
@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str           # Stable ID (hash-based from content + path)
    text: str             # Full document content
    source: str           # Origin: "filesystem", "web", "notion", "github"
    uri: str              # Path or URL
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
```

**Usage Example:**
```python
from rag.domain.models import Document

doc = Document(
    doc_id="abc123",
    text="This is the document content...",
    source="filesystem",
    uri="/path/to/file.md",
    metadata={"author": "user"}
)
```

### Chunk

A piece of a Document used for embedding/retrieval.

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str         # Stable ID: {doc_id}:{strategy}:{index}:{start}-{end}
    doc_id: str           # Reference to parent document
    text: str             # Chunk content

    # Provenance within the document
    chunk_index: int
    start_char: int | None = None
    end_char: int | None = None

    # Helpful for markdown/code corpora
    section_heading: str | None = None
    section_path: str | None = None    # e.g., "H1 > H2 > H3"
    language: str | None = None        # e.g., "python", "markdown"

    metadata: Mapping[str, Any] = field(default_factory=dict)
```

**Methods:**
- `from_dict(data: dict) -> Chunk` - Create from dictionary
- `to_dict() -> dict` - Convert to dictionary

### Candidate

A retrieved chunk plus scores from retrieval and optional reranking.

```python
@dataclass(frozen=True, slots=True)
class Candidate:
    chunk: Chunk
    score: float              # Retrieval similarity score (higher is better)
    rerank_score: float | None = None  # Optional reranker score
    debug: Mapping[str, Any] = field(default_factory=dict)  # For debugging
```

### Citation

A pointer to a source used in the final answer.

```python
@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    doc_id: str
    uri: str | None = None
    quote: str | None = None           # Small excerpt used/displayed
    section_heading: str | None = None
    section_path: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

### ContextPack

The final set of evidence given to the generator.

```python
@dataclass(frozen=True, slots=True)
class ContextPack:
    query: str
    chunks: Sequence[Chunk]
    rendered_context: str        # Formatted string for LLM
    citations: Sequence[Citation]
    token_budget: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

### Answer

Final model output.

```python
@dataclass(frozen=True, slots=True)
class Answer:
    query: str
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### QueryRunResult

Result of a full pipeline execution.

```python
@dataclass(frozen=True, slots=True)
class QueryRunResult:
    trace_id: str
    answer: Answer
    context_pack: ContextPack
    retrieved_chunk_ids: tuple[str, ...]
    reranked_chunk_ids: tuple[str, ...]
    packed_chunk_ids: tuple[str, ...]
    latency_ms: int
```

### QueryTrace

A structured record for observability and evaluation.

```python
@dataclass(frozen=True, slots=True)
class QueryTrace:
    trace_id: str
    query: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Retrieval
    top_k: int = 10
    retrieved: Sequence[Candidate] = field(default_factory=tuple)

    # Rerank
    reranked: Sequence[Candidate] = field(default_factory=tuple)
    keep_k: int | None = None
    reranker: str | None = None

    # Context build
    token_budget: int = 0
    packed_chunk_ids: Sequence[str] = field(default_factory=tuple)

    # Generation
    model: str | None = None
    latency_ms: int | None = None
    estimated_cost_usd: float | None = None

    # Final
    answer: Answer | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

### IngestReport

Summary statistics from an ingestion run.

```python
@dataclass(frozen=True, slots=True)
class IngestReport:
    scanned: int
    loaded: int
    skipped_hidden: int
    skipped_extension: int
    skipped_too_large: int
    skipped_empty: int
    failed: int
    by_extension: Mapping[str, int] = field(default_factory=dict)
```

---

## Regulatory Domain Models

### CaseDocument

**Location:** `src/rag/domain/case_documents.py`

An NRC case document fetched from the ADAMS API, with parsed dates and enriched metadata.

```python
@dataclass(frozen=True, slots=True)
class CaseDocument:
    accession_number: str       # e.g. "ML12345A678"
    title: str
    document_type: str          # raw ADAMS document type string
    document_date: datetime | None = None
    content: str | None = None
    author_name: str | None = None
    author_affiliation: str | None = None
    addressee: str | None = None
    addressee_affiliation: str | None = None
    docket_numbers: tuple[str, ...] = ()
    license_numbers: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    url: str | None = None
    estimated_pages: int | None = None
    case_metadata: CaseMetadata = field(default_factory=CaseMetadata)
```

### CaseMetadata

Classification results, extracted regulatory references, and provenance signals.

```python
@dataclass(frozen=True, slots=True)
class CaseMetadata:
    case_category: CaseCategory = CaseCategory.UNKNOWN
    case_subcategory: CaseSubcategory = CaseSubcategory.UNKNOWN
    case_category_method: str = "rules"
    case_category_confidence: float = 0.0
    case_category_reasons: tuple[str, ...] = ()
    case_signals: tuple[str, ...] = ()
    regulation_parts: tuple[str, ...] = ()
    regulation_sections: tuple[str, ...] = ()
    dockets: tuple[str, ...] = ()
    citation_keys: tuple[str, ...] = ()
    citation_spans: tuple[Mapping[str, object], ...] = ()
```

**Methods:**
- `validate()` -- Checks that subcategory matches category
- `to_dict()` -- Flattens to JSON-safe dict for storage in `Document.metadata`

### CaseCategory / CaseSubcategory

Top-level and fine-grained classification enums for NRC case documents.

```python
class CaseCategory(StrEnum):
    UNKNOWN = "unknown"
    LICENSING = "licensing"
    INSPECTION = "inspection"
    ENFORCEMENT = "enforcement"
    ADJUDICATION = "adjudication"
    RULEMAKING = "rulemaking"
    GENERIC_COMMUNICATION = "generic_communication"
    VENDOR_PART21 = "vendor_part21"
    DECOMMISSIONING = "decommissioning"
    INCIDENT = "incident"
    SECURITY = "security"
    OPERATIONS = "operations"
    QUALITY_ASSURANCE = "quality_assurance"
```

Subcategory values are namespaced by parent category (e.g., `enforcement_notice_of_violation`). Sentinel values `UNKNOWN` and `OTHER` are valid under any category.

### CaseClassification

Result of classifying a case document.

```python
@dataclass(frozen=True, slots=True)
class CaseClassification:
    category: CaseCategory
    subcategory: CaseSubcategory
    confidence: float
    method: str           # "rules" | "llm" | "manual"
    reasons: tuple[str, ...] = ()
```

### CitationSpan

**Location:** `src/rag/domain/citations.py`

A single citation extracted from document text, with span offsets and confidence scoring.

```python
@dataclass(frozen=True, slots=True)
class CitationSpan:
    kind: str             # "cfr" | "cfrpart" | "cfrapp" | "docket" | "adams" | "nureg" | "ris" | "gl" | "in"
    raw: str              # exact matched text
    key: str              # canonical key (stable, for dedup/linking)
    start: int            # span start in normalized text
    end: int              # span end in normalized text
    confidence: float     # 0.0-1.0 (deterministic scoring)
    source_field: str     # "title" | "content" | "metadata"
    context: str | None = None   # short context window for debugging
    attrs: dict[str, object] = field(default_factory=dict)  # parsed structure
```

**Canonical key format:** `{kind}:{identifier}` (e.g., `cfr:10:50.46`, `docket:50-247`, `adams:ML12345A678`)

### FetchReport

Summary statistics from a case-document fetch run.

```python
@dataclass(frozen=True, slots=True)
class FetchReport:
    total_requested: int
    fetched: int
    skipped: int
    failed: int
    by_document_type: Mapping[str, int] = field(default_factory=dict)
    errors: Sequence[str] = field(default_factory=tuple)
```

### Distributed Ingestion Models

**Location:** `src/rag/domain/ingestion.py`

| Model | Purpose |
|-------|---------|
| `IngestJob` | Top-level job state for a corpus indexing run |
| `IngestTask` | Single-document task with lease/attempt/error fields |
| `DocumentRecord` | Corpus-of-record pointer (`s3_raw_key`) and content hash |
| `JobStatus` | `CREATED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED` |
| `TaskStatus` | `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `RETRYABLE` |

---

## Ports (Interfaces)

All ports are Python `Protocol` classes (structural subtyping). They define the interface contracts that adapters must implement.

### Ingestor

Converts raw inputs (paths, URLs, etc.) into Documents.

```python
class Ingestor(Protocol):
    def ingest(
        self,
        inputs: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None
    ) -> tuple[list[Document], IngestReport]:
        ...
```

**Parameters:**
- `inputs`: Sequence of paths or URLs to ingest
- `metadata`: Optional metadata passed to documents

**Returns:**
- Tuple of (list of Documents, IngestReport with statistics)

**Implementations:** `FilesystemIngestor`

---

### Chunker

Splits a Document into Chunks.

```python
class Chunker(Protocol):
    def chunk(
        self,
        doc: Document,
        *,
        metadata: Mapping[str, object] | None = None
    ) -> list[Chunk]:
        ...
```

**Parameters:**
- `doc`: Document to split
- `metadata`: Optional metadata added to chunks

**Returns:**
- List of Chunk objects

**Implementations:** `FixedChunker`

---

### Embedder

Turns text into dense vectors.

```python
class Embedder(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None
    ) -> list[Vector]:
        ...
```

**Type Alias:**
```python
Vector = list[float]
```

**Parameters:**
- `texts`: Sequence of strings to embed
- `metadata`: Optional metadata for logging

**Returns:**
- List of vectors (same order as input texts)

**Implementations:** `OpenAIEmbedder`, `DummyEmbedder`, `CachedEmbedder`

---

### VectorStore

Stores (Chunk, Vector) pairs and supports similarity search.

```python
class VectorStore(Protocol):
    def upsert(
        self,
        *,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        metadata: Mapping[str, object] | None = None
    ) -> None:
        ...

    def search(
        self,
        *,
        query_vector: Vector,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None
    ) -> list[Candidate]:
        ...

    def count(self) -> int:
        ...

    def all_chunks(self) -> list[Chunk]:
        ...

    def save(self) -> None:
        """Persist the store to disk, if applicable."""
        ...

    def load(self) -> None:
        """Load the store from disk, if applicable."""
        ...
```

**Methods:**

| Method | Description |
|--------|-------------|
| `upsert()` | Insert or update chunk/vector pairs |
| `search()` | Find similar chunks by vector similarity |
| `count()` | Return total number of stored chunks |
| `save()` | Persist to disk (JSONL stores only) |
| `load()` | Load from disk (JSONL stores only) |

**Implementations:** `JsonlVectorStore`, `InMemoryVectorStore`

---

### Retriever

Retrieves candidate chunks for a query string.

```python
class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None
    ) -> list[Candidate]:
        ...
```

**Parameters:**
- `query`: User's natural language query
- `top_k`: Maximum number of candidates to return
- `where`: Optional filter expression (see Filter System)
- `metadata`: Optional metadata for logging

**Returns:**
- List of Candidate objects sorted by similarity score (descending)

**Implementations:** `VectorRetriever`

---

### Reranker

Re-orders candidates based on relevance to the query.

```python
class Reranker(Protocol):
    @property
    def name(self) -> str:
        ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        *,
        metadata: Mapping[str, object] | None = None
    ) -> list[Candidate]:
        ...
```

**Parameters:**
- `query`: User's query
- `candidates`: Candidates from retrieval
- `metadata`: Optional metadata

**Returns:**
- Reranked list of Candidates with `rerank_score` set

**Implementations:** `HeuristicReranker`, `NoOpReranker`

---

### ContextBuilder

Takes candidates and constructs the final prompt context within a token budget.

```python
class ContextBuilder(Protocol):
    def build(
        self,
        query: str,
        candidates: Sequence[Candidate],
        *,
        token_budget: int,
        metadata: Mapping[str, object] | None = None
    ) -> ContextPack:
        ...
```

**Parameters:**
- `query`: User's query
- `candidates`: Candidates to pack into context
- `token_budget`: Maximum tokens for context
- `metadata`: Optional metadata

**Returns:**
- ContextPack with rendered context and citations

**Implementations:** `SimpleContextBuilder`

---

### Generator

Produces an answer from the query + context.

```python
class Generator(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def generate(
        self,
        query: str,
        context: ContextPack,
        *,
        metadata: Mapping[str, object] | None = None
    ) -> Answer:
        ...
```

**Parameters:**
- `query`: User's query
- `context`: ContextPack with evidence
- `metadata`: Optional metadata

**Returns:**
- Answer with generated text and citations

**Implementations:** `OpenAIChatGenerator`

---

### QueryLogger

Persists query traces (typically JSONL).

```python
class QueryLogger(Protocol):
    def log(self, trace: QueryTrace) -> None:
        ...
```

**Parameters:**
- `trace`: QueryTrace object to persist

**Implementations:** `JsonlQueryLogger`

---

### NrcAdamsClient

**Location:** `src/rag/ports/nrc_adams_client.py`

Port for NRC ADAMS Public Search API interactions.

```python
class NrcAdamsClient(Protocol):
    def search_documents(
        self,
        query: str = "",
        *,
        filters: list[dict[str, object]] | None = None,
        any_filters: list[dict[str, object]] | None = None,
        sort_by: str = "DocumentDate",
        sort_desc: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[AdamsSearchResult]:
        """Search for documents with automatic pagination."""
        ...

    def get_document(
        self,
        accession_number: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> AdamsDocument | None:
        """Fetch a single document by accession number."""
        ...
```

**Data types:**
- `AdamsSearchResult` -- A single result from an ADAMS search query (accession number, title, document type, date)
- `AdamsDocument` -- Full document with content, docket numbers, license numbers, keywords

**Implementations:** `HttpAdamsClient` (`src/rag/adapters/ingestion/case/http_client.py`)

---

## Filter System

The filter system provides an AST-like structure for metadata filtering. Located in `src/rag/domain/filters.py`.

### Filter Types

```python
# Base class (abstract)
@dataclass(frozen=True, slots=True)
class Filter: ...

# Equality check
@dataclass(frozen=True, slots=True)
class Eq(Filter):
    field: str
    value: Any

# Membership check
@dataclass(frozen=True, slots=True)
class In(Filter):
    field: str
    values: Sequence[Any]

# Contains check (for lists/strings)
@dataclass(frozen=True, slots=True)
class Contains(Filter):
    field: str
    value: Any

# String prefix match
@dataclass(frozen=True, slots=True)
class Prefix(Filter):
    field: str
    prefix: str

# Range query
@dataclass(frozen=True, slots=True)
class Range(Filter):
    field: str
    gte: Any | None = None   # Greater than or equal
    lte: Any | None = None   # Less than or equal
    gt: Any | None = None    # Greater than
    lt: Any | None = None    # Less than

# Boolean combinations
@dataclass(frozen=True, slots=True)
class And(Filter):
    clauses: Sequence[Filter]

@dataclass(frozen=True, slots=True)
class Or(Filter):
    clauses: Sequence[Filter]

@dataclass(frozen=True, slots=True)
class Not(Filter):
    clause: Filter

# Type alias for nullable filters
Where = Filter | None
```

### Usage Examples

```python
from rag.domain.filters import Eq, In, And, Range, Prefix

# Simple equality
filter1 = Eq(field="source", value="filesystem")

# Membership
filter2 = In(field="language", values=["python", "typescript"])

# Range query
filter3 = Range(field="created_at", gte="2024-01-01")

# String prefix
filter4 = Prefix(field="uri", prefix="/docs/")

# Combined filters
filter5 = And(clauses=[
    Eq(field="source", value="filesystem"),
    In(field="language", values=["python", "typescript"]),
])
```

---

## Evaluation Schema

Located in `src/rag/eval/schema.py`.

### QueryType

Types of queries in the evaluation set.

```python
class QueryType(str, Enum):
    FACTUAL = "factual"           # Simple fact lookup
    COMPARISON = "comparison"     # Comparing two or more concepts
    AGGREGATION = "aggregation"   # Requires synthesizing multiple chunks
    PROCEDURAL = "procedural"     # How-to questions
    DEFINITION = "definition"     # What is X?
    CAUSAL = "causal"             # Why/how questions requiring reasoning
    TEMPORAL = "temporal"         # Time-based queries
    NEGATION = "negation"         # Questions about what is NOT in the vault
    MULTI_HOP = "multi_hop"       # Requires connecting multiple pieces of information
```

### Difficulty

Difficulty levels for queries.

```python
class Difficulty(str, Enum):
    EASY = "easy"     # Direct match, single chunk
    MEDIUM = "medium" # Requires 2-3 chunks or some reasoning
    HARD = "hard"     # Multi-hop reasoning, synthesis across many chunks
```

### EvalQuery

A single evaluation query with ground truth annotations.

```python
@dataclass(frozen=True, slots=True)
class EvalQuery:
    qid: str                              # Unique identifier
    query: str                            # Query string
    relevant_chunk_ids: set[str]          # Ground truth chunks

    # Optional expected answer
    expected_answer: str | None = None
    expected_answer_alternatives: list[str] = field(default_factory=list)

    # Query characteristics
    query_type: QueryType = QueryType.FACTUAL
    difficulty: Difficulty = Difficulty.EASY
    requires_synthesis: bool = False

    # Additional context
    notes: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: str | None = None
    created_by: str | None = None

    # Negative examples
    is_unanswerable: bool = False
    unanswerable_reason: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
```

**Methods:**
- `to_dict() -> dict` - Convert to dictionary for JSON serialization
- `from_dict(data: dict) -> EvalQuery` - Create from dictionary

### QuerySuggestion

A suggested query generated from a chunk by an LLM.

```python
@dataclass(frozen=True, slots=True)
class QuerySuggestion:
    query: str
    query_type: QueryType
    difficulty: Difficulty
    requires_synthesis: bool
    notes: str | None = None
```

### EvalDataset

A collection of evaluation queries with metadata.

```python
@dataclass(frozen=True, slots=True)
class EvalDataset:
    name: str
    version: str
    description: str
    queries: list[EvalQuery]
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Methods:**
- `to_dict() -> dict` - Convert to dictionary
- `from_dict(data: dict) -> EvalDataset` - Create from dictionary
- `filter_by_type(query_type: QueryType) -> list[EvalQuery]`
- `filter_by_difficulty(difficulty: Difficulty) -> list[EvalQuery]`
- `filter_by_tags(tags: set[str]) -> list[EvalQuery]`
- `stats() -> dict` - Get dataset statistics

---

## Pipeline Functions

Located in `src/rag/app/`.

### run_query

Execute a full RAG query pipeline with tracing.

```python
def run_query(
    query: str,
    *,
    retriever: Retriever,
    reranker: Reranker,
    context_builder: ContextBuilder,
    generator: Generator,
    logger: QueryLogger,
    top_k: int,
    keep_k: int | None,
    token_budget: int,
    where: Where = None,
    metadata: Mapping[str, object] | None = None,
) -> QueryRunResult:
```

**Pipeline Stages:**
1. **Retrieval**: Fetch top_k candidates from vector store
2. **Reranking**: Re-score candidates, optionally truncate to keep_k
3. **Context Building**: Pack chunks into prompt within token_budget
4. **Generation**: Generate answer using LLM
5. **Logging**: Record QueryTrace

**Returns:**
- `QueryRunResult` with trace_id, answer, context_pack, chunk IDs, and latency

### index_document

Index a single document.

```python
def index_document(
    doc: Document,
    *,
    chunker: Chunker,
    embedder: Embedder,
    store: VectorStore,
    metadata: Mapping[str, object] | None = None,
) -> int:
```

**Returns:**
- Number of chunks created

### rag_answer

Simplified pipeline without reranking or logging.

```python
def rag_answer(
    query: str,
    *,
    retriever: Retriever,
    context_builder: ContextBuilder,
    generator: Generator,
    top_k: int = 10,
    token_budget: int = 1800,
    where: Where = None,
    metadata: Mapping[str, object] | None = None,
) -> Answer:
```

---

## Container & Dependency Injection

### Container

Holds all composed adapters.

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

### ContainerOverrides

CLI overrides for settings.

```python
@dataclass(frozen=True, slots=True)
class ContainerOverrides:
    embedder_backend: Literal["openai", "dummy"] | None = None
    dummy_embed_dim: int | None = None
    store_backend: Literal["memory", "jsonl"] | None = None
    jsonl_index_dir: Path | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    vault_dir: Path | None = None
    top_k: int | None = None
    rerank_backend: Literal["heuristic", "noop"] | None = None
    rerank_enabled: bool | None = None
```

### build_container

Construct a Container from Settings and Overrides.

```python
def build_container(
    *,
    cfg: Settings | None = None,
    overrides: ContainerOverrides | None = None,
) -> Container:
```

**Usage:**
```python
from rag.app.container import build_container, ContainerOverrides

# Default configuration
container = build_container()

# With overrides
container = build_container(
    overrides=ContainerOverrides(
        embedder_backend="dummy",
        chunk_size=1000,
    )
)
```
