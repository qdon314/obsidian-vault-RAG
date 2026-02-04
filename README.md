# Obsidian Vault RAG

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **production-minded Retrieval-Augmented Generation (RAG) system** focused on **behavioral correctness, regression detection, and operational safety** of LLM-powered systems.

This project treats LLMs as **non-deterministic production dependencies** and centers on *owning model behavior over time* rather than prompt tuning or UI polish.

---

## Why This Exists

In real systems, most RAG failures are **retrieval failures**, not generation failures.
If the right evidence is not surfaced, generation quality is irrelevant.

More importantly, **LLM failures are often silent**:

* partial grounding
* confidently incorrect answers
* regressions masked by aggregate metrics

This project exists to explore how changes in **chunking, embeddings, retrieval, reranking, and context construction** affect downstream answers — and to make those effects **observable, measurable, and gateable** before they reach users.

---

## What This Project Optimizes For

This system is explicitly designed around **production concerns**:

* **Behavioral regression detection** (not just offline metrics)
* **Evidence-bounded answers** (citations, groundedness, abstention)
* **Go / no-go decisions** for retrieval and model changes
* **Traceability** across every stage of the pipeline
* **Operational repeatability** via containers, CI, and deployment manifests

---

## Production Safety Model

This system treats evaluation as a **release gate**, not a research artifact.

Every change to retrieval, chunking, reranking, or generation is evaluated against a **fixed, versioned evaluation dataset**.

Changes are blocked if they violate defined safety thresholds.

### Example Gates

* Recall@10 must not regress beyond an acceptable margin
* Unsupported claims must not increase
* Groundedness must not regress
* Abstention rate must remain within bounds
* P95 latency must remain within budget

Each evaluation run produces a **human-readable decision summary**:

```
Change: Fixed chunking → proposition-aware chunking

Results:
- Recall@10: +6.1%
- NDCG@10: +4.3%
- Unsupported claims: unchanged
- P95 latency: +9%

Decision: SHIP
Rationale: Gains concentrated in multi-hop queries without new hallucination classes
```

This framing reflects how production teams reason about LLM behavior.

---

## Quick Start

```bash
git clone https://github.com/your-username/obsidian-vault-RAG.git
cd obsidian-vault-RAG

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[openai]"

echo "OPENAI_API_KEY='sk-your-key'" > .env

python scripts/build_index.py --corpus ~/obsidian-vault --index-name my_index
python scripts/ask.py --index my_index --q "What are the main concepts?"
```

---

## Architecture Overview

```
Documents
  → Ingestion & Chunking
    → Embeddings
      → Vector Retrieval
        → (Optional) Reranking
          → Context Building
            → LLM Generation
              → Answer + Citations
              → Query Trace
```

The system follows **Hexagonal Architecture (Ports & Adapters)** to enable:

* Strict separation of interfaces and implementations
* Swappable components (OpenAI ↔ local models)
* Deterministic testing and evaluation
* Clear ownership boundaries

### Core Design Principles

* **Observability First**
  Every query produces a complete trace of retrieval, reranking, context packing, and generation.

* **Evaluation as Infrastructure**
  Evaluation is treated as a production dependency, not an experiment.

* **Behavior Over Outputs**
  The system measures *why* an answer was produced, not just whether it looks correct.

* **Reproducibility**
  Stable document and chunk IDs enable deterministic comparisons across runs.

---

## CI, Deployment, and Operational Guarantees

The system is designed to be **operated**, not demoed.

* Containerized runtime for reproducible execution
* Cloud-ready deployment manifests
* GitHub Actions for:

  * build validation
  * evaluation execution
  * regression gating
* Immutable evaluation artifacts persisted per run

This mirrors how LLM-backed systems are operated in production environments.

---

## Evaluation System

### Retrieval Metrics

* Recall@k, Precision@k, Hit Rate@k
* MRR, MAP
* NDCG@k
* Breakdown by query type and difficulty

### Answer Quality & Safety

* Correctness, completeness, relevance (LLM-as-judge)
* Hallucination detection
* Citation coverage
* Unsupported claim detection
* Abstention behavior

### Running an Evaluation

```bash
python -m experiments.run_eval \
  --queries experiments/eval_queries.jsonl \
  --run-generation \
  --use-llm-judge \
  --top-k 10 \
  --keep-k 4
```

Each run produces:

* aggregate metrics
* per-query breakdowns
* trace-level debugging artifacts
* a ship / block verdict

---

## Query Curation & Dataset Ownership

Evaluation datasets are treated as **first-class assets**.

An interactive Streamlit UI supports:

* browsing chunks in context
* creating single- and multi-hop queries
* selecting ground-truth chunks
* annotating difficulty and failure modes

```bash
pip install -e ".[ui]"
streamlit run experiments/streamlit_query_curator.py
```

This enables continuous evolution of eval datasets alongside the system.

---

## Moving Beyond Personal Notes

While the system supports Obsidian vaults, it is intentionally designed to scale to **harder, adversarial corpora**, including:

* technical documentation + RFCs
* regulatory or policy text
* multi-repository codebases
* time-sensitive or contradictory sources

These domains surface subtle retrieval failures that simpler corpora hide.

---

## Programmatic Usage

```python
from rag.app.container import build_container
from rag.app.query_runner import run_query

container = build_container()

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
```

---

## Open Questions

* When retrieval improvements justify latency tradeoffs
* How eval metrics fail under real user distributions
* Where deterministic heuristics outperform learned rerankers
* How to surface partial grounding failures earlier
* When and how to blend semantic and keyword retrieval

---

## Author

**Quentin Donnelly**
