# Documentation Index

Welcome to the Obsidian Vault RAG system documentation.

## Quick Links

| Document | Description |
|----------|-------------|
| [README](../README.md) | Project overview and quick start |
| [User Guide](USER_GUIDE.md) | Step-by-step usage instructions |
| [Configuration](CONFIGURATION.md) | Complete settings reference |
| [Architecture](ARCHITECTURE.md) | System design and data flows |
| [API Reference](API_REFERENCE.md) | Domain models and ports |
| [Adapters](ADAPTERS.md) | Implementation details |
| [Evaluation](evaluation/README.md) | Evaluation system and workflows |

---

## Getting Started

1. **New users**: Start with the [User Guide](USER_GUIDE.md)
2. **Configuration**: See [Configuration Reference](CONFIGURATION.md)
3. **Developers**: Read [Architecture](ARCHITECTURE.md) first

---

## Documentation Overview

### [User Guide](USER_GUIDE.md)

Practical guide for using the system:
- Installation and setup
- Building indexes
- Querying the system
- Running evaluations
- Query curation UI
- Troubleshooting

### [Configuration Reference](CONFIGURATION.md)

Complete configuration documentation:
- settings.toml sections
- Environment variables
- CLI overrides
- Example configurations

### [Architecture](ARCHITECTURE.md)

System design and patterns:
- Hexagonal architecture
- Component relationships
- Data flow diagrams (Mermaid)
- Design principles

### [API Reference](API_REFERENCE.md)

Programmatic interface documentation:
- Domain models (Document, Chunk, Answer, etc.)
- Port interfaces (protocols)
- Filter system
- Evaluation schema

### [Adapters Reference](ADAPTERS.md)

Concrete implementation details:
- Chunking adapters
- Embedding adapters
- Vector stores
- Rerankers
- Context builders
- Generators

### [Evaluation System](evaluation/README.md)

Comprehensive evaluation documentation:
- [Query Generation](evaluation/query_generation.md) - Creating eval queries with the UI
- [Running Evaluations](evaluation/running_evaluations.md) - Using the eval harness
- [Metrics Reference](evaluation/metrics.md) - Retrieval and answer metrics
- [Traces and Logging](evaluation/traces_and_logging.md) - Observability and debugging
- [Results Analyzer](evaluation/results_analyzer.md) - Interactive run analysis UI

---

## Quick Reference

### Project Structure

```
src/rag/
├── domain/           # Core data models
├── ports/            # Abstract interfaces
├── adapters/         # Concrete implementations
├── app/              # Pipeline orchestration
├── eval/             # Evaluation framework
└── settings.py       # Configuration loading
```

### Work Item Specs

Implementation-ready specifications for upcoming features:

| Spec | Description | Priority |
|------|-------------|----------|
| [Async OpenAI Embedder](specs/01-async-openai-embedder.md) | Async embedding with batching and connection pooling | P0 |
| [Resilience Patterns](specs/02-resilience-patterns.md) | Retry, circuit breaker, and timeout handling | P0 |
| [Judge Calibration](specs/03-judge-calibration.md) | Fix evaluation judge scoring | P0 |
| [Prometheus Metrics](specs/04-prometheus-metrics.md) | Metrics endpoint and health checks | P1 |
| [Hybrid Search](specs/05-hybrid-search.md) | Vector + keyword search with RRF | P1 |
| [Load Testing](specs/06-load-testing.md) | Locust load tests and benchmarks | P1 |

### Key Commands

```bash
# Build an index
python scripts/build_index.py --corpus ~/vault --index-name my_index

# Query the system
python scripts/ask.py --index my_index --q "What is X?"

# Run evaluation
python -m experiments.run_eval --queries eval_queries.jsonl

# Query curation UI
streamlit run experiments/streamlit_query_curator.py
```

### Configuration Quick Start

```toml
# settings.toml
[paths]
vault_dir = "~/obsidian-vault"

[embeddings]
backend = "openai"
model = "text-embedding-3-large"

[retrieval]
top_k = 8

[rerank]
enabled = true
keep_k = 4
```

---

## Additional Resources

- [FOCUS.md](FOCUS.md) - Current development focus
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) - Known issues and limitations
