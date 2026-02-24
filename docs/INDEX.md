# Documentation Index

Welcome to the Regulatory Corpus RAG system documentation.

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
| [Deployment](DEPLOYMENT.md) | Docker, CI/CD, and AWS deployment |

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
- Regulatory corpus ingestion (eCFR, NRC case documents)
- Building indexes
- Querying the system
- Running evaluations
- Query curation UI
- Troubleshooting

### [Configuration Reference](CONFIGURATION.md)

Complete configuration documentation:
- settings.toml sections (including NRC ADAMS and case ingestion)
- Environment variables
- CLI overrides
- Example configurations

### [Architecture](ARCHITECTURE.md)

System design and patterns:
- Hexagonal architecture
- Regulatory ingestion pipeline (eCFR XML, ADAMS API)
- Query pipeline (retrieve -> rerank -> context build -> generate -> trace)
- Distributed ingestion architecture
- Component relationships and data flow diagrams (Mermaid)

### [Distributed Ingestion Runbook](operations/distributed-ingestion.md)

Operational guidance for distributed ingestion:
- Required config and startup commands
- Queue/lease/task semantics
- Failure modes and recovery steps
- Safe disable/rollback actions

### [API Reference](API_REFERENCE.md)

Programmatic interface documentation:
- Domain models (Document, Chunk, Answer, CaseDocument, CitationSpan, etc.)
- Port interfaces (protocols)
- Filter system
- Evaluation schema

### [Adapters Reference](ADAPTERS.md)

Concrete implementation details:
- Regulatory ingestion adapters (eCFR parser, normalizer, cross-references, citation extractor)
- Chunking, embedding, vector store, retrieval, reranking, context building, generation adapters

### [Evaluation System](evaluation/README.md)

Comprehensive evaluation documentation:
- [Running Evaluations](evaluation/running_evaluations.md) - Using the eval harness
- [Metrics Reference](evaluation/metrics.md) - Retrieval and answer metrics
- [Traces and Logging](evaluation/traces_and_logging.md) - Observability and debugging
- [Results Analyzer](evaluation/results_analyzer.md) - Interactive run analysis UI
- [Verdict and Release Gating](evaluation/verdict_release_gating.md) - SHIP/BLOCK decision layer

### [Deployment Guide](DEPLOYMENT.md)

Infrastructure and operations:
- Local development with Docker Compose
- CI/CD via GitHub Actions (lint, typecheck, test, eval gate)
- AWS deployment via Terraform (ECS Fargate, ECR, S3, SQS, RDS)
- Scale-to-zero cost management

---

## Quick Reference

### Project Structure

```
src/rag/
├── domain/           # Core data models (Document, Chunk, CaseDocument, CitationSpan, ...)
├── ports/            # Abstract interfaces (Protocol classes)
├── adapters/         # Concrete implementations
│   ├── ingestion/
│   │   ├── regulatory/   # eCFR XML parser, normalizer, cross-references, metadata enrichment
│   │   └── case/         # NRC ADAMS case document fetcher, classifier, citation extractor
│   ├── query_generation/ # Term mapper, case query generator
│   ├── chunking/         # Fixed, structural, proposition-based
│   ├── embedding/        # OpenAI, dummy, SQLite cache
│   ├── retrieval/        # Vector, BM25, hybrid, hydrating
│   ├── reranking/        # Heuristic, no-op
│   ├── context_building/ # Simple, proposition-aware
│   ├── generation/       # OpenAI chat
│   ├── vectorstores/     # JSONL, in-memory, Qdrant
│   ├── chunk_storage/    # S3 + Postgres
│   └── ...
├── app/              # Pipeline orchestration
│   ├── container.py       # Dependency injection
│   ├── query_runner.py    # Full pipeline with tracing
│   ├── regulatory_pipeline.py  # eCFR normalization workflows
│   └── ingestion/         # Distributed ingestion (enumerator, worker)
├── eval/             # Evaluation framework
└── settings.py       # Configuration loading
```

### Key Commands

```bash
# Index regulatory corpus (eCFR)
make index-regulatory REGULATORY_XML=data/ecfr/title-10-part-50.xml REGULATORY_PART=50

# Query the system
make ask QUERY="What are the requirements for ECCS under 10 CFR 50.46?"

# Run evaluation
./scripts/py eval/scripts/run_eval.py --queries eval/datasets/curated_queries.jsonl

# Results analyzer UI
make results

# Release verdict
make verdict
```

### Configuration Quick Start

```toml
# settings.toml
[vectorstore]
backend = "qdrant"
qdrant_collection = "regulatory"
qdrant_url = "http://localhost:6333"

[embeddings]
backend = "openai"
model = "text-embedding-3-large"

[retrieval]
top_k = 10

[rerank]
enabled = true
keep_k = 4
```

---

## Additional Resources

- [README](../README.md) - Project overview and quick start
- [AGENTS.md](../AGENTS.md) - Repository command discipline and agent guidance
- [Notes: Infrastructure](notes/INFRASTRUCTURE.md) - Deep dive on Docker, CI/CD, Terraform, and AWS architecture
- [Notes: Distributed Ingestion Tools and Techniques](notes/DISTRIBUTED_INGESTION_TOOLS_AND_TECHNIQUES.md) - Implementation patterns and reliability techniques
