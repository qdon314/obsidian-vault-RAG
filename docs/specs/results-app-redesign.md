
# Rebuild `results_analyzer` as an evaluation workbench

The rebuild should not just be “a nicer Streamlit app.” It should become an **evaluation workbench** for your RAG system: a place where you can move cleanly from run-level symptoms to query-level diagnosis to pipeline-stage root cause. That direction fits the architecture and observability priorities of the system overall, where evaluation is first-class and every query can emit a structured `QueryTrace` with retrieval, rerank, context, generation, and metadata details.  

## Recommended product goal

Rebuild `results_analyzer` as a **modular evaluation analysis console** with these top-level goals:

1. **Extensibility**: new metrics, new artifact types, and new analysis views should plug in without rewriting the app shell.
2. **Granular diagnosis**: a single bad aggregate should be traceable down to bad queries, bad chunks, bad reranking, bad context packing, or bad answer behavior.
3. **Cross-artifact analysis**: metrics, results, traces, and verdicts should be joined into one model, not treated as separate screens.
4. **Robustness**: the UI should tolerate missing files, partial runs, schema drift, and optional fields.
5. **Comparative workflow**: the primary user motion should be “what got worse, for whom, and why?”

---

## High-level spec

### 1. Core concept

The rebuilt dashboard should revolve around a canonical **Loaded Evaluation Run** object that unifies:

* run metadata/config
* aggregate metrics
* per-query results
* traces joined by `trace_id`
* verdict artifacts when present
* optional derived tables/caches for exploration

That matches the current data model direction, where a loaded run already contains summary metadata, aggregates, per-query results, traces, and raw metrics. 

### 2. Primary user workflows

The dashboard should support four first-class workflows:

**A. Run triage**
“Is this run healthy, and what got worse?”

**B. Query forensics**
“For this failed query, was the problem retrieval, rerank, context packing, grounding, or generation?”

**C. Experiment comparison**
“Between baseline and candidate, which slices improved, which regressed, and which exact queries changed?”

**D. Trend monitoring**
“Across many runs, are we actually improving, and on which dimensions?”

Your current analyzer already covers the basic forms of these, but the rebuild should make them deeper and more composable. 

---

## Recommended architecture for the rebuild

### A. Separate app shell from analysis engine

Keep Streamlit as the rendering layer, but move almost all nontrivial logic into a reusable analysis package, something like:

* `eval/app_v2/domain/`
* `eval/app_v2/loaders/`
* `eval/app_v2/services/`
* `eval/app_v2/derived/`
* `eval/app_v2/views/`
* `eval/app_v2/widgets/`

The current analyzer already separates domain, ports, adapters, services, and UI. Preserve that instinct, but push it harder so UI components are mostly thin renderers over typed view-models. 

### B. Introduce a normalized analysis model

Create a normalized, schema-tolerant internal model:

* `RunBundle`
* `QueryRecord`
* `TraceRecord`
* `VerdictRecord`
* `MetricSeries`
* `ComparisonRecord`

This layer should normalize naming differences, optional fields, and missing artifacts before anything reaches the UI.

### C. Add a derived-data layer

Do not compute everything ad hoc inside Streamlit callbacks. Precompute:

* query pass/fail labels
* metric deltas
* query regression categories
* retrieval confusion sets
* stage-level timing summaries
* groundedness claim summaries
* verdict failures by rule
* filterable facet values

This makes the app faster, more testable, and much easier to extend.

---

## Specific features I strongly recommend

## 1. Run overview / triage page

This should be the landing page for a selected run.

Show:

* headline KPIs: Recall@K, Precision@K, Hit Rate@K, NDCG@K, MRR, MAP
* answer quality: correctness, completeness, relevance, hallucination severity, citation coverage, quality score
* groundedness: answerable-from-context rate, evidence-bounded rate, supported vs unsupported claims
* latency: avg, p50, p95
* verdict status: `SHIP` / `BLOCK`, with failed checks called out
* run config summary: retriever mode, reranker, chunker, token budget, model names

Those metrics line up directly with your current metrics reference and verdict layer.   

**Key addition:** a “Why should I care?” panel that automatically summarizes:

* the biggest regressions
* the worst slices
* the most costly failures
* whether the block came from retrieval, quality, groundedness, or latency

## 2. Slice analysis

This is one of the biggest missing opportunities.

Your metrics aggregates currently break down by query type and difficulty. Your query schema also has richer fields like `query_type`, `difficulty`, `requires_synthesis`, `is_unanswerable`, tags, and metadata.   

Build a generalized **slice explorer** that can group metrics by:

* query type
* difficulty
* requires synthesis
* answerable vs unanswerable
* dataset tag
* corpus filter / metadata filter
* custom tag families
* run config dimensions in trend mode

This becomes much more powerful than fixed by-type / by-difficulty sections.

## 3. Query forensics page

This should be the centerpiece.

For each query, provide a structured diagnostic layout:

**Header**

* qid
* full query
* type, difficulty, tags
* expected answer / citation metadata if available
* unanswerable flag

**Outcome**

* pass/fail labels by multiple definitions
* retrieval outcome
* answer quality outcome
* verdict-relevant flags
* latency tier

**Retrieval analysis**

* relevant chunk IDs
* retrieved chunk IDs by rank
* matched / missed / extra
* recall, precision, hit rate, ndcg for this query
* optional relevance tier handling

**Pipeline stage drilldown**

* retriever candidates and scores
* reranker reorder and score deltas
* packed chunk IDs vs retrieved IDs
* generated answer and citations
* groundedness claims with support state

Your current analyzer already displays much of this in pieces; the rebuild should make it a coherent forensic view. 

**Crucial addition:** a final section called **Root Cause Hypothesis** with machine-generated but deterministic heuristics like:

* “Relevant chunk existed in top_k but was dropped by reranker.”
* “Relevant chunk survived rerank but was omitted from packed context.”
* “Context contained sufficient evidence; answer still hallucinated.”
* “No relevant chunk retrieved; generation failure is downstream.”
* “Likely chunking/indexing issue: repeated misses on same citation family.”

That turns the dashboard from descriptive to genuinely useful.

## 4. Retrieval diff inspector

The current comparison view already has a retrieval diff table for queries between two runs. Keep that, but upgrade it. 

For a given query across two runs, show:

* rank movement of every relevant chunk
* newly introduced false positives
* chunks lost after rerank
* chunks added to context vs only retrieved
* answer text diff
* groundedness claim diff
* latency diff by stage if possible

This is where you’ll catch “reranker improved aggregate recall but worsened answer grounding on hard synthesis queries” kinds of problems.

## 5. Trace-first analysis

Because your traces contain retrieval, rerank, context, generation, and metadata, and `results.jsonl` includes `trace_id`, the rebuild should treat trace correlation as foundational, not optional. 

Add dedicated trace analyses:

* retrieval score distribution by rank
* rerank score vs original score scatter
* packed-chunk utilization view
* per-stage latency breakdown
* answer cost / estimated cost trend
* redaction-aware rendering when trace text is absent

This is especially important because your `QueryTrace` domain model already includes `estimated_cost_usd` and stage metadata capacity. 

## 6. Verdict analysis page

This is a big missing feature.

Your verdict system already produces `verdict.md` and `verdict.json`, with threshold checks for retrieval, quality, groundedness, latency, and regression rules.  

The dashboard should surface verdicts as first-class artifacts:

* ship/block badge
* failed rules list
* threshold values vs actual values
* regression deltas vs baseline
* which queries contributed most to each failed rule
* historical verdict timeline

This makes the UI much more aligned with CI and release gating instead of feeling like a sidecar explorer.

## 7. Trend analysis beyond line charts

Your current trending mode tracks recall, precision, ndcg, MRR, MAP, quality, and latency over time. Good start. 

Upgrade it with:

* config-aware trend annotations
* change-point markers
* best/worst run identification
* trend by slice, not just overall
* trend by dataset subset
* verdict timeline overlay
* latency/quality Pareto view
* metric correlation matrix across runs

That will help answer: “Did recall improve only because easy factual queries improved?” or “Did lower latency come at the expense of groundedness?”

---

## Features that specifically improve extensibility

## 1. Plugin-style metric registry

Right now the app is shaped around known metrics. Instead, create registries like:

* `aggregate_metric_registry`
* `query_metric_registry`
* `chart_registry`
* `facet_registry`
* `artifact_loader_registry`

Each metric definition should declare:

* metric key
* label
* scope: run / query / comparison / trend
* value type
* render hints
* whether higher is better
* formatting rules

This makes it easy to add future metrics like compression stats, retriever fusion stats, or cost metrics without UI surgery.

## 2. Artifact loader abstraction

The run loader should be able to ingest more than today’s three files.

Today’s run directories revolve around `metrics.json`, `results.jsonl`, and optional `traces.jsonl`; verdicts are generated separately.  

Future-proof by defining loaders for:

* metrics
* results
* traces
* verdict
* query dataset snapshot
* run config snapshot
* future reranker diagnostics
* future compression artifacts
* future ingestion provenance

## 3. Declarative facets and filters

Your filter system in the RAG app already uses typed filter AST concepts like `Eq`, `In`, `Contains`, `Range`, `And`, `Or`, `Not`. That same mindset is useful in the dashboard. 

Instead of hardcoding UI filters, define filter specs declaratively so you can add new facets cheaply.

Example facets:

* `query_type`
* `difficulty`
* `requires_synthesis`
* `is_unanswerable`
* `has_hallucination`
* `evidence_bounded`
* `retrieval_hit@k`
* `latency_bucket`
* `verdict_failure_type`

## 4. Schema-versioned run parsing

You’re going to evolve the eval harness. The analyzer should tolerate that.

Add:

* schema version detection
* graceful fallback for missing fields
* warnings for unknown fields
* compatibility adapters for older run formats

That will save you pain later.

---

## Features that specifically improve granular analysis

## 1. Stage attribution

For each bad query, compute stage labels like:

* retrieval miss
* rerank regression
* context packing omission
* generator unsupported claim
* abstention failure
* citation mismatch
* latency outlier

This is probably the single most useful analytical upgrade.

## 2. Chunk-centric debugging

Add views centered on chunk IDs / citation keys rather than only queries.

Examples:

* most frequently missed relevant chunks
* chunks frequently retrieved as false positives
* citations that correlate with hallucination
* documents overrepresented in false positives
* sections that fail across many queries

Because your chunks and citations have stable IDs and rich provenance, this is an excellent fit. 

## 3. Query clustering / cohorting

Add derived cohorts like:

* repeated failure families
* legal citation lookup vs synthesis queries
* short query vs long query
* exact citation queries vs conceptual questions
* generated-answer queries vs retrieval-only queries

This reveals systematic weaknesses much faster than scrolling rows.

## 4. Groundedness deep dive

You already have supported/unsupported claims and evidence-bounded signals in the metrics model and UI.  

Upgrade this with:

* claim-role grouping if available
* unsupported claim taxonomy
* unsupported claim count distribution
* unsupported claim heatmap by query type/difficulty
* answerable-from-context vs answer-produced mismatch view

---

## Concrete proposed pages

I’d organize the rebuilt dashboard like this:

### 1. Home / Triage

Run picker, verdict, KPI cards, top regressions, worst slices, recent trend mini-charts.

### 2. Runs

Rich run table with sortable columns, config badges, verdict badges, external paths, notes.

### 3. Single Run

Overview, metrics, slices, query table, trace table, raw artifacts.

### 4. Query Forensics

One-query deep inspection with retrieval/rerank/context/generation/groundedness.

### 5. Compare

Run A vs B, aggregate deltas, slice deltas, regressed query list, diff explorer.

### 6. Trends

Time series, config annotations, slice trends, verdict history, Pareto analysis.

### 7. Verdicts

Threshold failures, regression failures, contributor queries, release summary.

### 8. Artifacts

Raw JSON/JSONL viewers, download links, schema warnings.

---

## Concrete data model additions

I’d add these derived models:

* `RunHealthSummary`
* `QueryDiagnostic`
* `SliceMetricTable`
* `StageLatencySummary`
* `VerdictSummary`
* `RegressionCluster`
* `ChunkFailureProfile`

A `QueryDiagnostic` in particular should become the core of the forensic UI.

Suggested fields:

* `qid`
* `query`
* `labels`
* `retrieval_status`
* `rerank_status`
* `packing_status`
* `generation_status`
* `groundedness_status`
* `root_cause`
* `severity`
* `suggested_next_check`

---

## What to prioritize first

If you want this rebuild to matter, I would prioritize in this order:

### Phase 1: foundation

* canonical run loader
* normalized models
* robust artifact parsing
* derived-data layer
* home/triage page

### Phase 2: real debugging power

* query forensics page
* stage attribution
* retrieval diff inspector
* verdict integration

### Phase 3: extensibility

* metric registry
* declarative facets
* plugin loaders
* schema compatibility layer

### Phase 4: advanced analysis

* chunk-centric views
* slice trends
* clustering/cohorting
* cost/latency correlation

---

## My blunt opinion

The biggest risk is rebuilding the UI cosmetically while keeping the underlying analysis model too shallow. Don’t do that.

Your system already has the raw material for a genuinely strong evaluator dashboard:

* structured metrics and aggregates 
* per-query results with groundedness detail 
* trace correlation via `trace_id` 
* comparison and trend concepts already sketched 
* verdict-based release gating tied to CI 

So the right move is to rebuild it as an **analysis system**, not just a Streamlit dashboard.

If you want, I’ll turn this into a concrete implementation-ready spec with page-by-page requirements, domain models, and a phased task breakdown.
