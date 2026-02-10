# Traces and Logging

This document describes query traces used for observability and eval debugging.

## Overview

Each query run can emit a `QueryTrace` (see `src/rag/domain/models.py`) containing retrieval, rerank, context, and generation details.

## Logger Implementation

Primary adapter: `rag.adapters.logging.jsonl_logger.JsonlQueryLogger`

```python
from pathlib import Path

from rag.adapters.logging.jsonl_logger import JsonlQueryLogger

logger = JsonlQueryLogger(path=Path("artifacts/logs/traces.jsonl"), redact_text=False)
```

## Default Output Paths

- CLI query flow: `artifacts/logs/traces.jsonl`
- Eval runs: `eval/runs/<run>/traces.jsonl`

## QueryTrace Shape (high level)

- Identity: `trace_id`, `query`, `created_at`
- Retrieval: `top_k`, `retrieved`
- Rerank: `reranked`, `keep_k`, `reranker`
- Context: `token_budget`, `packed_chunk_ids`
- Generation: `model`, `latency_ms`, `estimated_cost_usd`
- Final: `answer`, `metadata`

## Inspecting Traces

```python
import json
from pathlib import Path

path = Path("artifacts/logs/traces.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
print(len(rows))
print(rows[-1]["trace_id"])
```

## Correlating with Eval Results

`results.jsonl` rows include `trace_id`. Use that to join per-query eval outcomes with detailed traces.

## Redaction

When `redact_text=True`, logger redacts content-like fields (`text`, `page_content`, `chunk_text`, `context_text`, `answer`) before writing JSONL.
