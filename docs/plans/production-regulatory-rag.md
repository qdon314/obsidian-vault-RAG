# Spec: Production-Scale Regulatory RAG Service

## Goal

Turn the current regulatory RAG system (ingestion + retrieval + evaluation) into a **production-shaped service** that demonstrates:

* **Large dataset** handling (cannot reasonably fit on one machine; scale via object storage + distributed index)
* **High traffic readiness** (concurrency, latency budgets, load testing, autoscaling)
* **High availability** (health checks, rolling deploys, graceful degradation)
* **Operational maturity** (observability, runbooks, cost/latency metrics, reproducible evals)
* **Modern AI methods** (hybrid retrieval + reranking + groundedness/abstention gates)

## Non-Goals

* Building a full multi-tenant SaaS product
* UI polish beyond what’s needed to demonstrate operations and debugging
* Fine-tuning custom models (optional later)

---

## System Overview

### Runtime Services

1. **Query API** (stateless)

   * Endpoints: `/query`, `/health`, `/metrics`, `/version`
   * Calls Retriever → (optional) Reranker → ContextBuilder → Generator
   * Enforces groundedness + abstention policy
2. **Ingestion Worker** (async)

   * Pulls corpora from S3, chunks, embeds, writes to vector store + chunk store
3. **Evaluation Runner** (batch)

   * Runs eval suites, produces reports + artifacts; can run on schedule or on-demand

### Data Stores

* **Object store (S3):**

  * Raw corpora
  * Normalized docs
  * Chunk artifacts
  * Eval run artifacts (JSONL, reports)
* **Vector store (distributed / managed):**

  * Qdrant cluster / OpenSearch kNN / Pinecone / Weaviate (pick one; Qdrant is very “infra engineer” friendly)
* **Chunk store (query-time fetch):**

  * Postgres (RDS) *or* DynamoDB
  * Stores chunk text + metadata keyed by `chunk_id`
* **Optional cache:** Redis (ElastiCache) for hot chunk fetches / query results

### Deployment

* ECS + Fargate (you already have tooling)
* ALB in front of Query API
* Autoscaling policies on CPU + request rate
* Blue/green or rolling deployments

---

## Key Requirements (what “production experience” looks like)

### R1 — Corpus scale that exceeds one machine

Demonstrate ingestion and indexing of a corpus large enough that:

* raw + normalized + chunked text is multi-GB and growing
* embedding/index operations are distributed / externalized (not local FS)
* full re-index is reproducible and automated

**Acceptance criteria**

* Ingest at least **10 CFR Title set** (not just one part), or equivalent multi-part corpus
* Store raw+normalized+chunks in S3 with deterministic keys
* Index resides in managed/distributed vector store (not local)

### R2 — High traffic + latency budgets

You must show the service under load with clear SLO-style metrics.

**Target SLOs (initial)**

* p50 latency ≤ 1.5s (retrieval-only)
* p95 latency ≤ 4.0s (with generation)
* Error rate < 1% under target load

**Acceptance criteria**

* k6/Locust load test scripts in repo
* Report showing behavior at **50–200 concurrent users**
* Autoscaling triggers and observed scale-out

### R3 — High availability posture

Prove sane ops defaults.

**Acceptance criteria**

* Health checks + readiness probes
* Multi-AZ where applicable (ALB + RDS Multi-AZ or Dynamo)
* Rolling deploy with no downtime (or documented, bounded)
* Graceful degradation:

  * If LLM unavailable → retrieval-only “evidence mode”
  * If reranker unavailable → fallback to vector + keyword ordering

### R4 — Observability & traceability

Every response must be explainable.

**Acceptance criteria**

* End-to-end trace ID on every request
* Logs include: query, top_k candidates IDs, scores, reranked order, keep_k IDs, citations emitted, abstention reason
* Metrics exported: request rate, latency histograms, token usage, cost estimates, cache hit rate
* “Reproduce this answer” workflow:

  * given `trace_id`, fetch the exact context pack + model inputs

### R5 — Evaluation is a first-class pipeline

Not just ad-hoc.

**Acceptance criteria**

* Eval suite runs in CI or on schedule
* Eval artifacts stored (S3) and browsable via your results analyzer
* Regression gates:

  * fail build if recall@K or groundedness drops > threshold

---

## Architecture Details

### 1) Data Model

**Doc**

* `doc_id`
* `regime`, `instrument`, `part`, `section`, `effective_date`
* `source_url`, `hash`, `ingested_at`

**Chunk**

* `chunk_id`
* `doc_id`
* `citation_key` (canonical)
* `text`
* `span_start`, `span_end`
* `metadata` (headings, subsection labels)

**Embedding Record**

* `chunk_id`
* `embedding_vector`
* `embedding_model`, `dim`
* `index_name` (supports reindexing)

### 2) Ingestion Pipeline

Stages (each stage writes artifacts and is resumable):

1. **Fetch** → store raw docs in S3
2. **Normalize** → canonical citation metadata + stable doc ids
3. **Chunk** → deterministic chunk ids
4. **Embed** → batch embeddings, retryable
5. **Index** → upsert to vector store
6. **Persist** → upsert chunk text/metadata to chunk store

**Operational features**

* Idempotent runs
* Checkpointing (manifest already exists in your system—great)
* Parallelism controls
* Backpressure + rate limits for embedding provider

### 3) Retrieval Policy

Default:

* **Hybrid retrieval** (BM25 + vector), merged candidates
* Reranker optional
* ContextBuilder enforces keep_k budget

**Configuration knobs**

* top_k_vector, top_k_bm25, merge_k
* rerank_k, keep_k
* per-query dynamic weighting (optional v2)

### 4) Answer Policy (Trust Gate)

* If groundedness confidence < threshold → abstain
* All claims must cite chunk ids / canonical citation keys
* “Evidence mode” fallback when generation unavailable

---

## Infrastructure Spec (AWS / ECS)

### Services

* `rag-query-api` (ECS service, autoscaled)
* `rag-ingest-worker` (ECS task, run on demand or scheduled)
* `rag-eval-runner` (ECS task, triggered by GH Actions / manual)

### Networking

* VPC, private subnets for ECS tasks
* ALB public, targets ECS
* Security groups: least privilege

### Storage

* S3 buckets:

  * `rag-raw`
  * `rag-normalized`
  * `rag-chunks`
  * `rag-eval-artifacts`
* RDS Postgres (or DynamoDB) for chunk store + metadata
* Vector store: Qdrant cluster (self-hosted on ECS/EC2) or managed service

  * If you self-host Qdrant: start with 3-node cluster

### Observability

* CloudWatch logs + metrics
* OpenTelemetry traces (recommended)
* Dashboard: latency, error rate, cost estimate, top failure modes

---

## CI/CD

### GitHub Actions

* Lint/test
* Build and push Docker images
* Deploy to ECS (blue/green or rolling)
* Trigger nightly eval job
* Regression thresholds as gates

**Regression gate examples**

* Recall@20 must not drop > 2%
* Groundedness must not drop > 1%
* Abstention correctness must not drop > 2%

---

## Load Testing Plan

### Scripts

* `load/query_readonly.js`: retrieval-only endpoint (fast)
* `load/query_full.js`: full RAG endpoint (LLM + retrieval)

### Scenarios

1. Steady 50 concurrent for 10 min
2. Ramp 0→200 concurrent
3. Spike test 10x for 60s
4. Degraded mode tests (LLM down)

Artifacts:

* latency histogram, p50/p95/p99
* error rate and timeouts
* autoscaling behavior evidence

---

## Deliverables (what you show Kevin / any founder)

1. **Live endpoint** (with auth)
2. **Public “ops README”**:

   * how it’s deployed
   * how to roll back
   * how to run eval
3. **Eval dashboard screenshots**
4. **Load test report**
5. **Postmortem-style writeup**:

   * top 5 failure modes before/after
   * what improved and why
6. **Cost/latency notes** (even rough)

---

## Suggested Milestones (8 weeks, aggressive but realistic)

### Week 1–2: “Service exists”

* ECS Query API deployed behind ALB
* S3-backed corpus artifacts
* Vector store externalized
* Basic traces + metrics

### Week 3–4: “Scale + reliability”

* Full Title ingestion (multi-part)
* Idempotent ingestion pipeline with checkpoints
* Degraded-mode behavior

### Week 5–6: “Eval as a gate”

* Nightly eval job
* Regression thresholds in CI
* Results analyzer reads from S3 artifacts

### Week 7–8: “Proof”

* Load test reports
* Dashboard screenshots
* Writeup + Loom walkthrough

---

## Key Design Choices (explicit tradeoffs)

* **Managed vs self-hosted vector store**

  * Managed reduces ops; self-host proves ops skill
* **RDS vs Dynamo for chunk store**

  * RDS easier querying; Dynamo scales simply
* **Hybrid retrieval**

  * Higher complexity, far less brittleness for regulatory text
* **Strict grounding**

  * Higher abstention, but safer/credible

