# [WIP] Production-Minded RAG System

## 1. Project Overview

### What this is

A retrieval-augmented generation (RAG) system for answering questions over a document corpus, with a deliberate focus on retrieval behavior, evaluation, and failure modes rather than prompt tuning or UI polish.

### Why it exists

In practice, most RAG failures come from retrieval issues, not generation. If the right information isn’t surfaced, no amount of prompt engineering fixes the result.

This project is a place to explore how choices around chunking, embeddings, retrieval, and reranking actually affect downstream answers, and to make those effects visible.

---

## 2. System Architecture

### High-level pipeline

```
Documents
  → Ingestion & Chunking
    → Embeddings
      → Vector Retrieval
        → (Optional) Reranking
          → LLM Generation
```

### Design goals

* Components with clear boundaries
* Intermediate results that can be inspected
* Easy to swap or experiment with retrieval and reranking strategies

### Core components

* Ingestion and chunking
* Embedding generation
* Vector search
* Reranking (optional)
* Generation and basic fallback handling

---

## 3. Data & Ingestion

### Document sources

* Currently focused on markdown documents (e.g. Obsidian vaults)
* Additional sources TBD

### Chunking strategies explored

* Fixed-size chunks
* Overlapping chunks
* Header-aware / structure-aware chunking
* Other variants as experiments evolve

### Why chunking matters

Chunking controls the trade-off between recall, precision, and context dilution. Poor chunking decisions tend to show up later as either missed retrievals or irrelevant context being passed to the model.

---

## 4. Retrieval & Embeddings

### Embeddings

* Model choice is treated as a variable rather than a constant
* Experiments focus on how embedding behavior interacts with chunking and retrieval depth

### Similarity search

* Standard vector similarity (cosine / dot product)
* Top-k selection treated as a tunable parameter

### Observed issues

* Semantically similar but irrelevant chunks
* Relevant chunks ranked just outside the top-k window
* Queries where semantic similarity alone is insufficient

---

## 5. Reranking

### Approaches considered

* No reranking (baseline)
* LLM-based reranking
* Cross-encoder reranking (where applicable)

### Trade-offs

* Quality improvements vs added latency
* Cost overhead
* Determinism vs variability in results

A recurring theme is that reranking often improves answer quality more than prompt changes, but introduces non-trivial cost and performance considerations.

---

## 6. Evaluation

### Metrics

* Recall@k
* Mean Reciprocal Rank (MRR)

These are used as proxies for downstream answer quality, with the understanding that they are imperfect but more actionable than purely generative metrics.

### Evaluation setup

* Small, hand-constructed query sets | TODO
* Explicit assumptions about what counts as a “correct” retrieval
* Known limitations documented alongside results | TODO

---

## 7. Experiments

### Chunking comparisons

* Different chunking strategies evaluated against the same query set | TODO

### Reranking impact

* Baseline vs reranked retrieval
* Latency and cost impact
* Examples where reranking fixes otherwise failing queries

### General takeaway

TBD

---

## 8. Failure Analysis

### Common failure modes

* Retrieval miss
* Semantic drift
* Ambiguous or underspecified queries
* Insufficient or fragmented context

### Analysis approach

For failed queries:

* Inspect retrieved chunks
* Compare against expected context
* Examine how failures propagate into generation

Most hallucinations are predictable once retrieval behavior is visible.

---

## 9. Trade-offs & Lessons So Far

### What’s working

* Making retrieval behavior explicit
* Treating evaluation as part of the system instead of a post-op

### What isn’t

* Over-reliance on semantic similarity
* Assuming better prompts fix poor retrieval

### Open questions

* How far reranking scales before cost dominates
* When multi-hop or decomposed retrieval is actually worth it
* How to balance determinism with model-driven ranking

---

## 10. Running the System

### Requirements

* Python
* Conda

If using OpenAI, define the API key in `.env`:

```bash
OPENAI_API_KEY='...'
```

### Setup

```bash
conda env create -f environment.yml
conda activate rag-obsidian
python -m pip install ".[openai | ollama>]"  # choose generator backend
```

### Build an index

```bash
python -m scripts.build_index \
  --corpus <path/to/your/vault> \
  --index-name "my_index" \
  --extensions ".md" \
  [--use-openai-embeddings]
```

### Query the system

```bash
python -m scripts.ask \
  --index my_index \
  --q "my query" \
  [--use-openai-embeddings]
```

---

## 11. Future Work

### Near-term

* Additional chunking strategies
* Structured query logging
* Reranking implementations
* Small, explicit evaluation datasets
* Experiment runner and metric persistence

### Longer-term

* Multi-hop retrieval
* Query decomposition
* Agent-style orchestration
* Better evaluation datasets

---