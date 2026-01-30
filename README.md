# Obsidian Vault RAG

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-minded retrieval-augmented generation (RAG) system for answering questions over document corpora, with a deliberate focus on **retrieval behavior**, **evaluation**, and **failure modes** rather than prompt tuning or UI polish.

## Why This Exists

In practice, most RAG failures come from retrieval issues, not generation. If the right information isn't surfaced, no amount of prompt engineering fixes the result.

This project explores how choices around chunking, embeddings, retrieval, and reranking actually affect downstream answers—and makes those effects visible through comprehensive logging and evaluation.

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/obsidian-vault-RAG.git
cd obsidian-vault-RAG
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[openai]"

# Set your API key
echo "OPENAI_API_KEY='sk-your-key'" > .env

# Build an index
python scripts/build_index.py --corpus ~/obsidian-vault --index-name my_index

# Ask a question
python scripts/ask.py --index my_index --q "What are the main concepts?"
```

---

## Architecture

```
Documents
  → Ingestion & Chunking
    → Embeddings
      → Vector Retrieval
        → (Optional) Reranking
          → Context Building
            → LLM Generation
              → Answer with Citations
```

The system follows **Hexagonal Architecture** (Ports & Adapters), enabling:
- Clean separation between interfaces and implementations
- Easy swapping of components (OpenAI ↔ local models)
- Comprehensive testing through protocol-based interfaces

### Design Principles

- **Observability First**: Every query generates a complete trace with all intermediate results
- **Evaluation as First-Class**: Built-in evaluation framework with retrieval and answer quality metrics
- **Clean Boundaries**: Protocol-based interfaces allow easy component swapping
- **Reproducibility**: Stable IDs for documents and chunks enable deterministic behavior

---

## Project Structure

```
src/rag/
├── domain/           # Core data models (Document, Chunk, Candidate, Answer, QueryTrace)
├── ports/            # Abstract interfaces (Chunker, Embedder, Retriever, Generator, etc.)
├── adapters/         # Concrete implementations
│   ├── chunking/     # Fixed-size chunker
│   ├── embedding/    # OpenAI, dummy embedder, SQLite cache
│   ├── generation/   # OpenAI chat
│   ├── ingestion/    # Filesystem loader, Obsidian markdown loader
│   ├── retrieval/    # Vector retriever
│   ├── vectorstores/ # JSONL store, in-memory store
│   ├── reranking/    # Heuristic reranker, no-op
│   ├── context_building/
│   └── logging/      # JSONL query logger
├── app/              # Pipeline orchestration & dependency injection
├── eval/             # Evaluation harness & metrics
└── settings.py       # Configuration loading

scripts/
├── build_index.py    # Build index from corpus
├── ask.py            # Query the system
└── project_state.py  # Project inspection

experiments/
├── run_eval.py                  # Evaluation runner
├── generate_queries.py
├── curate_queries.py
├── create_starter_set.py
└── streamlit_query_curator.py   # Interactive query curation UI
```

---

## Features

### Ingestion
- **Obsidian-aware loading**: Expands transclusions/embeds (`![[...]]` syntax)
- **Multiple formats**: Markdown (`.md`), plain text (`.txt`)
- **Document tracking**: Stable `doc_id` from content hash

### Chunking
- **Fixed-size chunking**: 800 chars default, 120 char overlap
- **Metadata preservation**: Section headings, paths, language tags
- **Chunk provenance**: Stable `chunk_id` with document offsets

### Embeddings
- **OpenAI backend**: `text-embedding-3-large` (configurable)
- **Dummy embedder**: Random vectors for testing without API costs
- **SQLite caching**: Persistent embedding cache to reduce API calls

### Vector Storage
- **JSONL store**: Human-readable `chunks.jsonl` for inspection
- **In-memory store**: For experiments & testing
- **Cosine similarity search** with metadata filtering

### Reranking
- **Heuristic reranker**: Lexical overlap boost + diversity by `doc_id`
- **No-op reranker**: Baseline (vector similarity only)

### Context Building
- **Token budget enforcement**: Fits chunks within max tokens (~4 chars/token)
- **Deduplication**: Removes near-duplicate chunks

### Generation
- **OpenAI integration**: `gpt-4o-mini` (configurable)
- **Temperature control**: 0.2 default for consistency
- **Abstention detection**: Recognizes "I don't know" responses

### Query Tracing
- **Structured `QueryTrace`**: Documents every retrieval stage
- **JSONL persistence**: One query per line for analysis
- **Timing breakdown**: Per-stage latency metrics

---

## Evaluation System

### Retrieval Metrics
- **Recall@k**, **Precision@k**, **Hit Rate@k**
- **Mean Reciprocal Rank (MRR)**
- **Mean Average Precision (MAP)**
- **NDCG@k**

### Answer Quality (LLM-as-Judge)
- **Correctness**, **Completeness**, **Relevance** (0-5 scale)
- **Hallucination detection**
- **Semantic similarity**

### Running Evaluations

```bash
python -m experiments.run_eval \
  --queries experiments/eval_queries.jsonl \
  --run-generation \
  --use-llm-judge \
  --top-k 10 \
  --keep-k 4
```

### Query Curation UI

Interactive Streamlit tool for creating evaluation datasets:

```bash
pip install -e ".[ui]"
streamlit run experiments/streamlit_query_curator.py
```

Features:
- Chunk browser with document tree navigation
- Multi-chunk selection for synthesis queries
- LLM-generated query suggestions
- Full `EvalQuery` field editing

---

## Installation

### Prerequisites

- Python >= 3.11

### Setup

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with OpenAI support
pip install -e ".[openai]"

# Or for local models
pip install -e ".[ollama]"

# For the Streamlit UI
pip install -e ".[ui]"

# For development
pip install -e ".[dev]"
```

### API Keys

Create a `.env` file:

```bash
OPENAI_API_KEY='sk-your-key-here'
```

---

## Usage

### Building an Index

```bash
# With OpenAI embeddings
python scripts/build_index.py \
  --corpus ~/obsidian-vault \
  --index-name my_index

# With dummy embeddings (free, for testing)
python scripts/build_index.py \
  --corpus ~/obsidian-vault \
  --index-name test_index \
  --use-dummy-embeddings
```

### Querying

```bash
python scripts/ask.py \
  --index my_index \
  --q "What is the main concept?"
```

### Make Commands

```bash
make help          # Show available commands
make index         # Build index with OpenAI embeddings
make index-dummy   # Build index with dummy embeddings
make ask QUERY="your question here"
make tail-logs     # Inspect recent query logs
make clean-index   # Remove index (dangerous)
```

---

## Configuration

Configuration lives in `settings.toml`:

```toml
[paths]
vault_dir = "~/obsidian-vault"
artifacts_dir = "artifacts"

[chunking]
chunk_size = 800
overlap = 120

[embeddings]
backend = "openai"  # or "dummy"
model = "text-embedding-3-large"

[retrieval]
top_k = 8

[rerank]
enabled = true
backend = "heuristic"
keep_k = 4

[llm]
backend = "openai"
model = "gpt-4o-mini"
temperature = 0.2
```

CLI flags can override settings for one-off experiments.

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [User Guide](docs/USER_GUIDE.md) | Step-by-step usage instructions |
| [Configuration](docs/CONFIGURATION.md) | Complete settings reference |
| [Architecture](docs/ARCHITECTURE.md) | System design with diagrams |
| [API Reference](docs/API_REFERENCE.md) | Domain models and ports |
| [Adapters](docs/ADAPTERS.md) | Implementation details |

---

## Programmatic Usage

```python
from rag.app.container import build_container
from rag.app.query_runner import run_query
from rag.settings import load_settings

# Build container with settings
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

---

## Open Questions

- How far reranking scales before cost dominates
- When multi-hop or decomposed retrieval is actually worth it
- How to balance determinism with model-driven ranking

---

## Future Work

### Near-term
- Additional chunking strategies (semantic, header-aware)
- LLM-based reranking implementations
- Larger, more diverse evaluation datasets

### Longer-term
- Multi-hop retrieval
- Query decomposition
- Agent-style orchestration

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Quentin Donnelly**
