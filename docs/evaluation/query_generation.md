# Query Generation and Curation

This document covers creating and managing evaluation queries using the Streamlit curation UI.

## Overview

Evaluation queries are the foundation of the evaluation system. Each query consists of:

- A natural language question
- Ground-truth relevant chunk IDs (for retrieval evaluation)
- An expected answer (for generation evaluation)
- Metadata (query type, difficulty, tags, filters)

## Launching the Curation UI

```bash
make eval
# or
streamlit run eval/app/main.py
```

The UI opens with a sidebar showing:
- Chunks file path (from `settings.toml`)
- Output queries file path
- Statistics (loaded chunks, documents, query counts)

## UI Workflow

The UI has two main tabs: **Create** and **Review**.

### Create Tab

#### Step 1: Browse and Select Chunks

The chunk browser shows a two-column layout:

- **Left column**: Document tree with chunk counts
- **Right column**: Searchable chunk list for the selected document

Select chunks that should be retrieved for your query. You can select multiple chunks as ground truth.

**Chunk Preview Details:**
- Chunk ID and document ID
- Chunk index and character offsets
- Section information (if available)
- Full chunk text
- Custom metadata

#### Step 2: Generate Query Suggestions (Optional)

Click "Generate Suggestions" to use LLM-powered query generation:

```python
# Uses OpenAIQuerySuggester internally
suggester = OpenAIQuerySuggester(
    api_key=settings.openai_api_key,
    model="gpt-4o-mini",
    temperature=0.8,
)
suggestions = suggester.suggest_queries(chunk, num_suggestions=3)
```

Each suggestion includes:
- Query text
- Suggested query type
- Difficulty estimate
- Whether synthesis is required
- Generation notes

Click "Use" to populate the form with a suggestion.

#### Step 3: Edit and Save Query

The query editor form includes:

| Field | Required | Description |
|-------|----------|-------------|
| Query | Yes | The natural language question |
| Expected Answer | No | Reference answer for evaluation |
| Query Type | Yes | Factual, comparison, procedural, etc. |
| Difficulty | Yes | Easy, medium, hard |
| Requires Synthesis | No | Whether answer spans multiple chunks |
| Is Unanswerable | No | Mark as negative example |
| Tags | No | Comma-separated labels |
| Notes | No | Annotation notes |
| Retrieval Filter | No | Optional filter DSL for scoped retrieval |

**Query Types:**

| Type | Description | Example |
|------|-------------|---------|
| `factual` | Direct fact lookup | "What is the capital of France?" |
| `comparison` | Compare two concepts | "How does X differ from Y?" |
| `aggregation` | Summarize multiple items | "List all the features of X" |
| `procedural` | Step-by-step process | "How do I configure X?" |
| `definition` | Define a concept | "What is X?" |
| `causal` | Cause and effect | "Why does X happen?" |
| `temporal` | Time-based queries | "When was X released?" |
| `negation` | What is NOT something | "What does X not support?" |
| `multi_hop` | Requires reasoning chain | "Who wrote the book that introduced X?" |

### Review Tab

The review tab lists all existing queries with:

- Search and filter by type/difficulty
- Edit button to modify queries
- Delete button with confirmation

Editing preserves the original `created_at` timestamp and updates `last_modified`.

## Query Data Model

```python
@dataclass(frozen=True)
class EvalQuery:
    qid: str                           # Unique ID (e.g., "q_20240115_143052_001")
    query: str                         # Question text
    relevant_chunk_ids: frozenset[str] # Ground truth chunks
    expected_answer: str = ""          # Reference answer
    expected_answer_alternatives: tuple[str, ...] = ()
    query_type: QueryType = QueryType.factual
    difficulty: Difficulty = Difficulty.medium
    requires_synthesis: bool = False
    is_unanswerable: bool = False
    unanswerable_reason: str = ""
    tags: frozenset[str] = frozenset()
    notes: str = ""
    created_at: str = ""               # ISO timestamp
    created_by: str = ""               # Author
    metadata: dict = field(default_factory=dict)
```

## Retrieval Filters

Filters allow scoping retrieval to specific documents or metadata values. Use the filter builder UI or raw JSON.

### Supported Filter Types

| Type | Description | Example |
|------|-------------|---------|
| `Eq` | Exact match | `{"type": "Eq", "field": "doc_id", "value": "doc123"}` |
| `In` | Match any in list | `{"type": "In", "field": "tags", "values": ["python", "api"]}` |
| `Contains` | String contains | `{"type": "Contains", "field": "uri", "value": "/docs/"}` |
| `Prefix` | String starts with | `{"type": "Prefix", "field": "uri", "value": "/api/"}` |
| `Range` | Numeric range | `{"type": "Range", "field": "year", "gte": 2020, "lte": 2024}` |
| `And` | All conditions | `{"type": "And", "clauses": [...]}` |
| `Or` | Any condition | `{"type": "Or", "clauses": [...]}` |
| `Not` | Negate condition | `{"type": "Not", "clause": {...}}` |

### Example: Complex Filter

```json
{
  "type": "And",
  "clauses": [
    {"type": "Eq", "field": "doc_id", "value": "architecture.md"},
    {"type": "Or", "clauses": [
      {"type": "Contains", "field": "section", "value": "API"},
      {"type": "Contains", "field": "section", "value": "REST"}
    ]}
  ]
}
```

### Using Filters in Evaluation

Filters are stored in `EvalQuery.metadata["filter"]` and deserialized during evaluation:

```python
query = eval_queries[0]
where = query.get_filter()  # Returns Filter domain object or None
candidates = retriever.retrieve(query.query, top_k=10, where=where)
```

## Storage Format

Queries are stored in JSONL format (one JSON object per line):

```jsonl
{"qid": "q_001", "query": "What is X?", "relevant_chunk_ids": ["c1", "c2"], ...}
{"qid": "q_002", "query": "How does Y work?", "relevant_chunk_ids": ["c3"], ...}
```

The `JsonlEvalStore` adapter handles persistence:

```python
from src.rag.adapters.eval_persistence import JsonlEvalStore

store = JsonlEvalStore()
queries = store.load_queries(Path("queries.jsonl"))
store.append_query(new_query, Path("queries.jsonl"))
```

## Best Practices

### Query Design

1. **Be specific** - Avoid vague queries that could match many chunks
2. **Test coverage** - Include all query types and difficulty levels
3. **Negative examples** - Mark unanswerable queries to test abstention
4. **Multi-chunk queries** - Include queries requiring synthesis across chunks

### Ground Truth Selection

1. **Minimal set** - Only mark chunks that directly answer the query
2. **Complete coverage** - Include all chunks that contain relevant info
3. **Avoid near-duplicates** - If two chunks have the same info, pick one

### Metadata

1. **Use tags** - Group queries by topic, feature, or test category
2. **Add notes** - Document edge cases or why a query is interesting
3. **Accurate difficulty** - Easy = single chunk, obvious. Hard = multi-hop, synthesis

## LLM Query Suggestion

The `OpenAIQuerySuggester` generates query suggestions from chunk content:

```python
from src.rag.adapters.query_suggestion import OpenAIQuerySuggester

suggester = OpenAIQuerySuggester(
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.8,
    max_passage_chars=2000,
)

suggestions = suggester.suggest_queries(chunk, num_suggestions=3)
for s in suggestions:
    print(f"{s.query_type}: {s.query}")
```

The suggester uses a detailed prompt to generate diverse, well-typed queries based on the chunk content.

## See Also

- [Running Evaluations](running_evaluations.md) - Using the evaluation harness
- [Metrics Reference](metrics.md) - Understanding evaluation metrics
