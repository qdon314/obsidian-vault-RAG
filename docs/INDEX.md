# Documentation Index

Welcome to the Obsidian Vault RAG system documentation.

## Quick Links

| Document | Description |
|----------|-------------|
| [README](../README.md) | Project overview and quick start |
| [User Guide](USER_GUIDE.md) | Step-by-step usage instructions |
| [Configuration](CONFIGURATION.md) | Complete settings reference |
| [Architecture](ARCHITECTURE.md) | System design and data flows |
| [Distributed Ingestion Runbook](operations/distributed-ingestion.md) | Operator playbook for enumerator/workers |
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

### [Distributed Ingestion Runbook](operations/distributed-ingestion.md)

Operational guidance for distributed ingestion:
- Required config and startup commands
- Queue/lease/task semantics
- Failure modes and recovery steps
- Safe disable/rollback actions

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
| [Regulatory Corpus Ingestion](specs/04-regulatory-corpus-ingestion.md) | Regulatory XML ingestion and normalization | P0 |
| [Dependency Cleanup](specs/05-dependency-cleanup.md) | Remove stale deps and simplify runtime requirements | P1 |
| [Query Changes Enhancement](specs/06-query-changes-enhancement.md) | Better per-query run diff and diagnostics | P1 |
| [Production Regulatory RAG](specs/production-regulatory-rag.md) | Production hardening plan for regulatory workflow | P1 |
| [Agentic Growth System](specs/AGENTIC_GROWTH_SYSTEM_SPEC.md) | Long-horizon capability growth framework | P2 |

### Key Commands

```bash
# Build an index
./scripts/py scripts/build_index.py --corpus ~/vault --index-name my_index

# Query the system
./scripts/py scripts/ask.py --index my_index --q "What is X?"

# Run evaluation
./scripts/py eval/scripts/run_eval.py --queries eval/datasets/curated_queries.jsonl

# Results analyzer UI
make results
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

- [README](../README.md) - Project overview and quick start
- [AGENTS.md](../AGENTS.md) - Repository command discipline and agent guidance
- [Notes: Infrastructure](notes/INFRASTRUCTURE.md) - Deep dive on Docker, CI/CD, Terraform, and AWS architecture
- [Notes: Distributed Ingestion Tools and Techniques](notes/DISTRIBUTED_INGESTION_TOOLS_AND_TECHNIQUES.md) - Implementation patterns and reliability techniques used in Phase 3
