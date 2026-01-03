# [WIP]Production-Minded RAG System

## 1. Project Overview
**What this is**

* A retrieval-augmented generation (RAG) system designed to answer questions over a document corpus with explicit emphasis on **retrieval quality, evaluation, and failure analysis**.

**Why it exists**

* In practice, most RAG failures stem from **poor retrieval**, not generation.
* This project explores how chunking, embeddings, and reranking affect downstream answer quality.

---

## 2. System Architecture

**High-level pipeline**

```
Documents
  → Ingestion & Chunking
    → Embeddings
      → Vector Retrieval
        → Reranking
          → LLM Generation
```

**Design principles**

* Modular components with clear boundaries
* Observable intermediate outputs
* Easy to swap retrieval or reranking strategies

**Key components**

* Ingestion & chunking
* Embedding generation
* Vector search
* Reranker(s)
* Generation & fallback logic

---

## 3. Data & Ingestion

**Document sources**

* TODO

**Chunking strategies explored**

* Fixed-size chunks
* Overlapping chunks
* Header-aware / semantic chunking
* ...

**Why chunking matters**

* Trade-off between recall, precision, and context dilution
* Chunking decisions directly affect retrieval metrics

---

## 4. Retrieval & Embeddings

**Embedding model(s) used**

* (Name + why chosen)

**Similarity search**

* Metric used (cosine / dot product)
* Top-k selection rationale

**Observed behaviors**

* When semantic similarity fails
* Cases of false positives / false negatives

---

## 5. Reranking Strategies

**Approaches evaluated**

* No reranking (baseline)
* LLM-based reranking
* Cross-encoder reranking (if applicable)

**Trade-offs analyzed**

* Accuracy vs latency
* Cost vs quality
* Determinism vs variability

**Key insight**

* Reranking often improves answer quality more than prompt tuning, but introduces measurable latency and cost overhead.

---

## 6. Evaluation Methodology

**Metrics used**

* Recall@k
* Mean Reciprocal Rank (MRR)

**Why these metrics**

* Correlation with downstream answer correctness
* Limitations of generative metrics for retrieval evaluation

**Evaluation setup**

* Query set construction
* Ground truth assumptions
* Known limitations

---

## 7. Experiments & Results

**Experiment 1: Chunking Strategy Comparison**

* Setup
* Results (summary table or bullets)
* Observations

**Experiment 2: Reranking Impact**

* Baseline vs reranked
* Latency & cost impact
* Failure reduction examples

**Key takeaway**

* Small improvements in retrieval metrics can lead to disproportionately better generation outcomes.

---

## 8. Failure Analysis

**Failure categories identified**

* Retrieval miss
* Semantic drift
* Ambiguous query
* Insufficient context

**Concrete examples**

* Example query
* Retrieved chunks
* Generated answer
* Why it failed

**Why this matters**

* Hallucinations are often *predictable* given retrieval behavior.

---

## 9. Trade-offs & Lessons Learned

**What worked**

* ...
* ...

**What didn’t**

* ...
* ...

**Open questions**

* ...
* ...
* ...

---

## 10. How to Run

**Requirements**

* Python version
* Dependencies

**Setup**
If using OpenAI, be sure to define the API key in .env:
```bash
OPENAI_API_KEY='...'
```
```bash
# install dependencies (must have conda already installed)
conda env create -f environment.yml
conda activate rag-obsidian
python -m pip install ".[openai | ollama>]" # choose either openai or ollama as generator

# ingest documents (assumes targeting obsidian vault for now)
python -m scripts.build_index --corpus <path/to/your/vault> --index-name "my_index" --extensions ".md" [--use-openai-embeddings]

# query the system
python -m scripts.ask --index my_index --q "my query" [--use-openai-embeddings]
```

---

## 11. Future Work
### P1
 - Implement additional chunkers
 - Implement logger
 - Implement reranking
 - Define eval query set (20–50 queries)
 - Define ground-truth assumptions (expected docs/chunks)
 - Add experiment runner script
 - Save metrics as CSV/JSON
 - Add structured query logging schema
 - More...
### P2
* Multi-hop retrieval
* Query decomposition
* Agent-based orchestration
* Improved evaluation datasets

---

