# User Guide

A complete guide to using the Obsidian Vault RAG system, from setup to advanced usage.

## Table of Contents

- [Getting Started](#getting-started)
- [Building an Index](#building-an-index)
- [Querying the System](#querying-the-system)
- [Evaluation](#evaluation)
- [Query Curation UI](#query-curation-ui)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Python >= 3.11
- Conda (recommended)
- OpenAI API key (for production embeddings and generation)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/obsidian-vault-RAG.git
   cd obsidian-vault-RAG
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate rag-obsidian
   ```

3. **Install the package:**
   ```bash
   # For OpenAI support (recommended)
   pip install -e ".[openai]"

   # For local models (Ollama)
   pip install -e ".[ollama]"

   # For the Streamlit UI
   pip install -e ".[ui]"
   ```

4. **Configure API keys:**

   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY='sk-your-api-key-here'
   ```

### Quick Start

```bash
# Build an index from your vault
python scripts/build_index.py --corpus ~/obsidian-vault --index-name my_index

# Ask a question
python scripts/ask.py --index my_index --q "What are the main concepts?"
```

---

## Building an Index

### Basic Usage

Build an index from your document corpus:

```bash
python scripts/build_index.py \
  --corpus ~/obsidian-vault \
  --index-name my_index
```

### Using Dummy Embeddings

For testing without API costs:

```bash
python scripts/build_index.py \
  --corpus ~/obsidian-vault \
  --index-name test_index \
  --use-dummy-embeddings
```

### Custom Chunk Size

Adjust chunking parameters:

```bash
python scripts/build_index.py \
  --corpus ~/obsidian-vault \
  --index-name my_index \
  --chunk-size 1000 \
  --chunk-overlap 150
```

### Index Location

By default, indexes are stored in:
```
artifacts/indexes/{index-name}/
└── chunks.jsonl
```

### Make Commands

For convenience, use the Makefile:

```bash
make help          # Show available commands
make index         # Build index with OpenAI embeddings
make index-dummy   # Build index with dummy embeddings
make clean-index   # Remove index (dangerous!)
```

---

## Querying the System

### Basic Query

```bash
python scripts/ask.py \
  --index my_index \
  --q "What is the main topic?"
```

### Adjusting Retrieval

```bash
python scripts/ask.py \
  --index my_index \
  --q "What is X?" \
  --top-k 10 \      # Retrieve more candidates
  --keep-k 5        # Keep top 5 after reranking
```

### Disabling Reranking

```bash
python scripts/ask.py \
  --index my_index \
  --q "What is X?" \
  --rerank-backend noop
```

### Make Commands

```bash
make ask QUERY="What is the main topic?"
```

### Understanding Output

The system outputs:
1. **Answer**: Generated response with citations like [1], [2]
2. **Citations**: Source files and excerpts used
3. **Trace ID**: Unique identifier for debugging

Example output:
```
Answer:
The main topic is about building RAG systems [1]. Key concepts include
retrieval, chunking, and embedding [2].

Citations:
[1] /docs/intro.md (lines 10-50)
[2] /docs/concepts.md (lines 1-80)

Trace ID: abc123def456
```

---

## Evaluation

### Running Evaluations

Execute the evaluation harness:

```bash
python -m experiments.run_eval \
  --queries experiments/eval_queries.jsonl \
  --run-generation \
  --use-llm-judge \
  --top-k 10 \
  --keep-k 4
```

### Evaluation Options

| Flag | Description |
|------|-------------|
| `--queries` | Path to evaluation queries JSONL file |
| `--run-generation` | Enable answer generation (not just retrieval) |
| `--use-llm-judge` | Use LLM-as-judge for answer quality |
| `--top-k N` | Number of candidates to retrieve |
| `--keep-k N` | Number to keep after reranking |

### Retrieval Metrics

The evaluation reports:

| Metric | Description |
|--------|-------------|
| **Recall@k** | Fraction of relevant chunks in top-k |
| **Precision@k** | Fraction of top-k that are relevant |
| **Hit Rate@k** | Queries with at least one relevant in top-k |
| **MRR** | Mean Reciprocal Rank (position of first relevant) |
| **MAP** | Mean Average Precision |
| **NDCG@k** | Normalized Discounted Cumulative Gain |

### Answer Quality Metrics

When using `--use-llm-judge`:

| Metric | Scale | Description |
|--------|-------|-------------|
| Correctness | 0-5 | Factual accuracy |
| Completeness | 0-5 | Covers all relevant info |
| Relevance | 0-5 | Answers the actual question |
| Hallucination | 0-5 | Lower is better |

### Evaluation Output

Results are saved to:
```
artifacts/eval/
├── run_{run_id}/
│   ├── results.jsonl
│   ├── aggregates.json
│   └── meta.json
```

---

## Query Curation UI

The Streamlit-based query curation tool helps create evaluation datasets.

### Starting the UI

```bash
pip install -e ".[ui]"  # Install streamlit dependency
streamlit run experiments/streamlit_query_curator.py
```

### Features

1. **Chunk Browser**
   - Document tree navigation
   - Searchable chunk list
   - Preview chunk content

2. **Query Creation**
   - Multi-chunk selection for ground truth
   - LLM-generated query suggestions
   - Full EvalQuery field editing

3. **Review Mode**
   - View existing queries
   - Filter by type/difficulty
   - Edit or delete queries

### Workflow

1. **Select Chunks**: Browse and select relevant chunks
2. **Generate Suggestions**: Use LLM to suggest queries
3. **Customize Query**: Edit type, difficulty, expected answer
4. **Save**: Add to evaluation dataset

---

## Troubleshooting

### Common Issues

#### "OPENAI_API_KEY is required but not set"

```bash
# Set the environment variable
export OPENAI_API_KEY='sk-your-key'

# Or create a .env file
echo "OPENAI_API_KEY='sk-your-key'" > .env
```

#### "Missing config file: settings.toml"

```bash
# Copy the example settings
cp settings.example.toml settings.toml
```

#### "Invalid JSON on chunks.jsonl"

The index file may be corrupted. Rebuild the index:
```bash
make clean-index
make index
```

#### Empty Results

1. Check that the index was built successfully
2. Verify the index path matches what you're querying
3. Try increasing `top_k`
4. Check query relevance to indexed content

#### High Latency

1. Consider using `--use-dummy-embeddings` for testing
2. Reduce `top_k` and `keep_k`
3. Enable embedding cache (SQLite)

### Inspecting Logs

View query logs for debugging:

```bash
make tail-logs

# Or manually
tail -f artifacts/logs/queries.jsonl | jq .
```

### Log Fields

| Field | Description |
|-------|-------------|
| `trace_id` | Unique query identifier |
| `query` | Original query text |
| `top_k` | Retrieval count |
| `retrieved` | All retrieved candidates |
| `reranked` | Post-reranking candidates |
| `packed_chunk_ids` | Chunks sent to LLM |
| `latency_ms` | Total execution time |
| `metadata.timing_ms` | Per-stage timing |

### Getting Help

1. Check this documentation
2. Review the [README](../README.md)
3. Open an issue on GitHub

---

## Advanced Usage

### Programmatic Usage

```python
from rag.app.container import build_container, ContainerOverrides
from rag.app.query_runner import run_query
from rag.settings import load_settings

# Load settings and build container
settings = load_settings()
container = build_container(cfg=settings)

# Run a query
result = run_query(
    "What is the main concept?",
    retriever=container.retriever,
    reranker=container.reranker,
    context_builder=container.context_builder,
    generator=container.generator,
    logger=container.logger,
    top_k=8,
    keep_k=4,
    token_budget=1500
)

print(result.answer.text)
print(f"Latency: {result.latency_ms}ms")
```

### Custom Adapters

To use custom adapters:

```python
from rag.app.container import build_container, ContainerOverrides

# Use dummy embeddings
container = build_container(
    overrides=ContainerOverrides(
        embedder_backend="dummy",
        dummy_embed_dim=256
    )
)

# Use in-memory store
container = build_container(
    overrides=ContainerOverrides(
        store_backend="memory"
    )
)
```

### Filter Queries

Filter by metadata during retrieval:

```python
from rag.domain.filters import Eq, And, Prefix

# Filter by source
filter = Eq(field="source", value="filesystem")

# Filter by path prefix
filter = Prefix(field="uri", prefix="/docs/")

# Combine filters
filter = And(clauses=[
    Eq(field="source", value="filesystem"),
    Prefix(field="uri", prefix="/docs/")
])

# Use in retrieval
candidates = retriever.retrieve("query", top_k=10, where=filter)
```

### Batch Processing

Process multiple queries:

```python
queries = ["What is X?", "How does Y work?", "Explain Z"]

results = []
for query in queries:
    result = run_query(
        query,
        retriever=container.retriever,
        # ... other args
    )
    results.append(result)
```

---

## Best Practices

### Chunking

- **800-1200 characters**: Good balance of context and precision
- **10-15% overlap**: Preserves cross-boundary context
- Test different sizes for your corpus

### Retrieval

- **top_k 8-15**: Retrieve enough candidates for reranking
- **keep_k 3-5**: Focus on best matches for generation
- Enable reranking for better precision

### Evaluation

- Create queries at multiple difficulty levels
- Include negative examples (unanswerable queries)
- Use diverse query types (factual, comparison, procedural)
- Review retrieval metrics before answer quality

### Performance

- Use SQLite embedding cache for repeated texts
- Build indexes once, query many times
- Consider dummy embeddings for development
