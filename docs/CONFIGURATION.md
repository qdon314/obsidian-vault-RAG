# Configuration Reference

Complete reference for configuration in the RAG system.

## Table of Contents

- [Configuration File](#configuration-file)
- [Configuration Precedence](#configuration-precedence)
- [Environment Variables](#environment-variables)
- [Settings Sections](#settings-sections)
- [Eval Verdict Thresholds](#eval-verdict-thresholds)
- [CLI Overrides](#cli-overrides)

---

## Configuration File

Primary config file: `settings.toml` in the project root.

```toml
# settings.toml - Canonical configuration for the RAG system
# Settings define defaults.
# Containers compose adapters.
# CLI overrides settings via ContainerOverrides.
```

## Configuration Precedence

1. `settings.toml` defaults
2. Environment overrides (`OPENAI_API_KEY`, plus mapped env overrides)
3. CLI flags (single-run overrides)

---

## Environment Variables

| Variable | Description | Required When |
|----------|-------------|---------------|
| `OPENAI_API_KEY` | OpenAI API key | `embeddings.backend="openai"` or `llm.backend="openai"` |

Example:

```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

---

## Settings Sections

### `[paths]`

```toml
[paths]
vault_dir = "/Users/quentindonnelly/Documents/Personal & Professional"
artifacts_dir = "artifacts"
queries_file = "eval/datasets/curated_queries.jsonl"
index_dir = "artifacts/indexes/obsidian"
```

| Option | Type | Default in `settings.toml` | Notes |
|--------|------|-----------------------------|-------|
| `vault_dir` | path | required | Corpus root |
| `artifacts_dir` | path | `artifacts` | Output root |
| `queries_file` | path | `eval/datasets/curated_queries.jsonl` | Eval/query dataset path |
| `index_dir` | path | `artifacts/indexes/obsidian` | Default index location |

`~` and env vars in paths are expanded at load time.

### `[ingestion]`

```toml
[ingestion]
recursive = true
skip_hidden = true
allowed_extensions = [".md", ".txt"]
expand_embeds = true
max_embed_depth = 4
```

| Option | Type | Default in `settings.toml` | Notes |
|--------|------|-----------------------------|-------|
| `recursive` | bool | `true` | Recurse subdirectories |
| `skip_hidden` | bool | `true` | Skip hidden files/directories |
| `allowed_extensions` | list[string] | `[".md", ".txt"]` | File extensions to ingest |
| `expand_embeds` | bool | `true` | Expand Obsidian `![[...]]` embeds |
| `max_embed_depth` | int | `4` | Max recursive embed expansion depth |

### `[chunking]`

```toml
[chunking]
backend = "obsidian_structural" # "fixed" | "obsidian_structural" | "obsidian_proposition"
chunk_size = 800
overlap = 120
target_chars = 4000
hard_max_chars = 5200
overlap_blocks = 1
include_heading_preamble = true
proposition_batch_size = 2
```

| Option | Type | Default in `settings.toml` | Used By |
|--------|------|-----------------------------|---------|
| `backend` | string | `"obsidian_structural"` | All chunking |
| `chunk_size` | int | `800` | `fixed` |
| `overlap` | int | `120` | `fixed` |
| `target_chars` | int | `4000` | `obsidian_structural` |
| `hard_max_chars` | int | `5200` | `obsidian_structural` |
| `overlap_blocks` | int | `1` | `obsidian_structural` |
| `include_heading_preamble` | bool | `true` | `obsidian_structural` |
| `proposition_batch_size` | int | `2` | `obsidian_proposition` |

Note: loader fallback default for `proposition_batch_size` is `8` when omitted.

### `[context]`

```toml
[context]
max_chunks = 5
dedupe = true
include_scores = false
# min_score = 0.5
token_budget = 1500
```

| Option | Type | Default in `settings.toml` | Notes |
|--------|------|-----------------------------|-------|
| `max_chunks` | int | `5` | Max chunks passed to generation |
| `dedupe` | bool | `true` | Remove near-duplicates |
| `include_scores` | bool | `false` | Include retrieval scores in rendered context |
| `min_score` | float \| null | unset | Optional score threshold |
| `token_budget` | int | `1500` | Context token budget |

### `[embeddings]`

```toml
[embeddings]
backend = "openai"         # "openai" | "dummy"
model = "text-embedding-3-large"
timeout = 30.0
max_retries = 3
cache_embeddings = true
cache_db_path = "artifacts/cache/embeddings/embedding_cache.db"
dummy_dim = 128
```

| Option | Type | Default in `settings.toml` | Notes |
|--------|------|-----------------------------|-------|
| `backend` | string | `"openai"` | Provider backend |
| `model` | string | `"text-embedding-3-large"` | Model name |
| `timeout` | float | `30.0` | Request timeout (seconds) |
| `max_retries` | int | `3` | Retry attempts |
| `cache_embeddings` | bool | `true` | Enable SQLite embedding cache |
| `cache_db_path` | path | `artifacts/cache/embeddings/embedding_cache.db` | Cache DB location |
| `dummy_dim` | int | `128` | Only for `backend="dummy"` |

### `[vectorstore]`

```toml
[vectorstore]
backend = "jsonl"         # "memory" | "jsonl" | "qdrant"
jsonl_dir = "artifacts/indexes/obsidian_index"
qdrant_collection = "obsidian"
# qdrant_url = "http://localhost:6333"
qdrant_path = "artifacts/indexes"
# qdrant_api_key = "..."
```

| Option | Type | Default in `settings.toml` | Notes |
|--------|------|-----------------------------|-------|
| `backend` | string | `"jsonl"` | Store backend |
| `jsonl_dir` | path | `artifacts/indexes/obsidian_index` | Used by `jsonl` backend |
| `qdrant_collection` | string | `"obsidian"` | Used by `qdrant` backend |
| `qdrant_url` | string \| null | unset | Remote Qdrant URL |
| `qdrant_path` | path | `artifacts/indexes` | Local Qdrant persistence path |
| `qdrant_api_key` | string \| null | unset | Qdrant Cloud key |

### `[retrieval]`

```toml
[retrieval]
backend = "vector"            # "vector" | "hybrid"
top_k = 8
hybrid_primary_weight = 0.7
hybrid_secondary_weight = 0.3
hybrid_rrf_k = 60
hybrid_bm25_k1 = 1.5
hybrid_bm25_b = 0.75
```

| Option | Type | Default in `settings.toml` | Notes |
|--------|------|-----------------------------|-------|
| `backend` | string | `"vector"` | Retrieval mode |
| `top_k` | int | `8` | Candidate count before rerank |
| `hybrid_primary_weight` | float | `0.7` | Vector component weight |
| `hybrid_secondary_weight` | float | `0.3` | BM25 component weight |
| `hybrid_rrf_k` | int | `60` | RRF fusion constant |
| `hybrid_bm25_k1` | float | `1.5` | BM25 `k1` |
| `hybrid_bm25_b` | float | `0.75` | BM25 `b` |

### `[rerank]`

```toml
[rerank]
enabled = true
backend = "heuristic"      # "heuristic" | "noop"
keep_k = 4
```

| Option | Type | Default in `settings.toml` | Notes |
|--------|------|-----------------------------|-------|
| `enabled` | bool | `true` | Enable reranking |
| `backend` | string | `"heuristic"` | Reranker implementation |
| `keep_k` | int | `4` | Final candidate count |

### `[llm]`

```toml
[llm]
backend = "openai"
model = "gpt-4.1-mini"
temperature = 0.2
max_tokens = 1024
timeout = 60.0
max_retries = 3
```

| Option | Type | Default in `settings.toml` | Notes |
|--------|------|-----------------------------|-------|
| `backend` | string | `"openai"` | Current provider |
| `model` | string | `"gpt-4.1-mini"` | Chat model |
| `temperature` | float | `0.2` | Sampling temperature |
| `max_tokens` | int | `1024` | Max completion tokens |
| `timeout` | float | `60.0` | Request timeout (seconds) |
| `max_retries` | int | `3` | Retry attempts |

---

## Eval Verdict Thresholds

`[eval.verdict]` configures release-gating thresholds for eval verdicts.

```toml
[eval.verdict]
min_recall_at_10 = 0.60
min_ndcg_at_10 = 0.50
min_mrr = 0.40
max_avg_hallucination_severity = 2.5
min_evidence_bounded_rate = 0.70
max_latency_p95_ms = 5000.0
max_unsafe_miss_rate = 0.10
max_abstain_bad_rate = 0.10
max_recall_regression = 0.05
max_quality_regression = 0.10
max_latency_regression_ms = 1000.0
```

| Option | Type | Meaning |
|--------|------|---------|
| `min_recall_at_10` | float | Minimum Recall@10 |
| `min_ndcg_at_10` | float | Minimum nDCG@10 |
| `min_mrr` | float | Minimum MRR |
| `max_avg_hallucination_severity` | float | Maximum average hallucination severity |
| `min_evidence_bounded_rate` | float | Minimum evidence-bounded response rate |
| `max_latency_p95_ms` | float | Maximum p95 latency in ms |
| `max_unsafe_miss_rate` | float | Maximum unsafe miss rate |
| `max_abstain_bad_rate` | float | Maximum bad abstention rate |
| `max_recall_regression` | float | Max allowed recall regression |
| `max_quality_regression` | float | Max allowed quality regression |
| `max_latency_regression_ms` | float | Max allowed latency regression in ms |

Scoped overrides are supported by nested tables such as `[eval.verdict.regulatory]`; scoped values merge on top of base verdict thresholds.

---

## CLI Overrides

Use pinned interpreter wrappers from `AGENTS.md`.

### Build Index

```bash
./scripts/py scripts/build_index.py \
  --corpus /path/to/corpus \
  --index-name my-index \
  --chunk-size 800 \
  --overlap 120
```

Common build overrides:
- `--chunk-size`, `--overlap` for fixed chunking
- `--target-chars`, `--hard-max-chars`, `--overlap-blocks`, `--no-heading-preamble` for structural chunking
- `--use-dummy-embeddings`, `--embed-dim`
- `--cache-embeddings` / `--no-cache-embeddings`
- `--extensions`, `--max-docs`, `--embed-batch-size`, `--no-parallel`

### Ask

```bash
./scripts/py scripts/ask.py \
  --index my-index \
  --q "your question" \
  --top-k 8 \
  --token-budget 1500
```

Common ask overrides:
- `--top-k`, `--token-budget`
- `--use-dummy-embeddings`, `--embed-dim`
- `--cache-embeddings` / `--no-cache-embeddings`
- `--skip-validation`

Note: `ask.py` uses `settings.rerank.keep_k` for reranking and defaults `token_budget` to `1800` if `--token-budget` is not provided.
