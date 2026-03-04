Below is a concrete, implementation-ready *high-level design/spec* for adding **ScaleDown** as a “context compression” stage in your RAG pipeline, and wiring its returned metrics into both **QueryTrace** and **eval aggregates**.

---

## 1) Where ScaleDown fits in your pipeline

Your current `run_query()` stage order is explicitly: retrieval → rerank → context building → generation → logging. 

ScaleDown should slot in **between ContextBuilder and Generator**:

```mermaid
flowchart LR
  A[Retriever] --> B[Reranker]
  B --> C[ContextBuilder\n(ContextPack.rendered_context)]
  C --> D[ScaleDown Compressor\n(optional)]
  D --> E[Generator]
  E --> F[QueryLogger\n(QueryTrace)]
```

This matches your architectural dataflow where `ContextPack` is the final evidence bundle passed into generation. 

---

## 2) New Port + Adapter design

### 2.1 Add a new Port: `PromptCompressor`

Add a new Protocol alongside your existing ports (`ContextBuilder`, `Generator`, etc.). 

**Port responsibilities**

* Accept the “final prompt inputs” (rendered context + query/prompt instructions)
* Return:

  * `compressed_prompt` string to feed into the downstream Generator
  * structured compression metrics (tokens, latency, success flag, etc.)

**Suggested interface (conceptual)**

* `compress(*, context: str, prompt: str, metadata: Mapping[str, Any] | None = None) -> CompressionResult`

### 2.2 Implement `ScaleDownPromptCompressor` adapter

Use the ScaleDown raw compress endpoint shown in their Quickstart (`https://api.scaledown.xyz/compress/raw/`) with header `x-api-key`. ([docs.scaledown.ai][1])

Payload pattern from their docs:

* `context`: your `ContextPack.rendered_context`
* `prompt`: the “question + your answer instructions”
* `scaledown.rate`: `"auto"` or configured rate ([docs.scaledown.ai][1])

**Auth**

* Support env var `SCALEDOWN_API_KEY` if you also adopt their python package conventions. ([docs.scaledown.ai][2])
* Or keep it consistent with your system’s “settings.toml + env override” approach. 

---

## 3) How to build the exact prompt you send to ScaleDown

Today, the generator stage is “Generate answer using LLM”, fed by `ContextPack` (wh_context`) and the query. 

To integrate ScaleDown cleanly, standardize a **single “user prompt body”** that ScaleDown returns as `compressed_prompt`:

* `context` = `context_:contentReference[oaicite:9]{index=9}pt` = a stable template such as:

```
QUESTION:
{query}

Answer clearly and cite chunk numbers like [1], [2] where relevant.
Use only the provided CONTEXT. If the answer cannot be found in the CONTEXT, say you don't know.
```

Then treat `compressed_prompt` as the **entire user message** you pass to the LLM (system message remains unchanged).

This is consistent with ScaleDown’s positioning: “compress a prompt… ready to be used with your AI model.” ([docs.scaledown.ai][1])

---

## 4) QueryTrace integration (per-query metrics)

Your `QueryTrace` already has a flexible `metadata: Mapping[str, Any]` specifically for attaching extra structured fields. 

### 4.1 Store ScaleDown metrics in `QueryTrace.metadata`

Add a nested object so it’s easy to query in JSONL and safe to extend:

```json
{
  "scaledown": {
    "successful": true,
    "latency_ms": 2341,
    "or:contentReference[oaicite:12]{index=12}   "compressed_prompt_tokens": 65,
    "token_savings": 85,
    "savings_pct": 0.5667,
    "request_metadata": {
      "compression_time_ms": 2341,
      "compression_rate": "auto",
      "prompt_length": 425,
      "compressed_prompt_length": 189
    }
  }
}
```

This is compatible with your trace model and logging flow (logger records `QueryTrace` after generation). 

### 4.2 Failure behavior: “fail open” by default

If ScaleDown errors or returns `successful=false`:

* fall back to the original (uncompressed) user prompt body
* set `metadata.scaledown.successful=false` and att `status_code`, etc.)
* still run generation (unless you explicitly want “fail closed” in CI)

---

## 5) Eval metrics integration (run-level aggregates + UI)

Your eval system already aggregates retrieval/quality/latency into `metrics.json`. 
And your results analyzer reads trace “raw_data” for per-query detail. 

### 5.1 Add a new top-level metrics section: `compression`

Compute from traces across the run:

**Required ess_rate`

* `original_prompt_tokens_avg`, `p50`, `p95`
* `compressed_promp:contentReference[oaicite:18]{index=18} `savings_pct_avg`, `p50`, `p95`
* `scaledown_latency_ms_avg`, `p50`, `p95`

This aligns with how you already treat latency as a first-class aggregate. 

### 5.2 Results Analyzer display updates

Add to:

* **Run Configuration Expander**: show “Compression enabled, rate, target_model” (similar to how you show generator/embedder model names). a “Compression” block beneath latency
* **Query Explorer Table**: optional extra columns:

  * `Savings%`
  * `Compressed tokens`
  * `ScaleDown latency`
* **Trace Viewer**: show `scaledown` n (you already have a “raw data” trace model that can show extra fields). 

---

## 6) Configuration spec

Add a new `[compression]` section in `settings.toml` (and support env overrides using your existing convention). 

**Proposed settings**

* `enabled: b:contentReference[oaicite:25]{index=25}ledown" | "noop" = "scaledown"`
* `scaledown_api_url: str = "https://api.scaledown.xyz/compress/raw/"`
* `scaledown_api_key_env: str = "SCALEDOWN_:contentReference[oaicite:26]{index=26} = "auto"` ([docs.scaledown.ai][1])
* `timeout_s: float = 10.0`
* `max_retries: int = 2`
* `fail_open: bool = true`
* `min_expected_savings_pct: float | None` (optional sanity check)
* `target_model: str | None` (if you adopt their SDK concepts; otherwise omit) ([docs.scaledown.ai][2])

---

## 7) Verdict/release gating (optional but powerful)

Your verdict layer already gates on p95 latency and other thresholds. 

If you want ScaleDown to be “production-real”, add optional thresholds under `[eval.verdict]`:

* `max_scaledown_latency_p95_ms`
* `min_scaledown_success_rate`
* (optional) `min_prompt_savings_pct_avg`

This makes compression a **first-class regression surface** (not just a cost optimization).

---

## 8) Implementable slices

1. **Domain + Port**

* Add `CompressionResult` dataclass
* Add `PromptCompressor` Protocol

2. **ScaleDown adapter**

* Implement HTTP client call per Quickstart (URL + `x-api-key` + payload shape). ([docs.scaledown.ai][1])
* Parse response into `CompressionResult` (your provided response shape)

3. **Pipeline wiring**

* Extend `run_query()` signature to accept `compressor: PromptCompressor | None`
* Apply compressor between ContextBuilder and Generator 
* Record metrics in `QueryTrace.metadata.scaledown` 

4. **Eval aggregation**

* Update metrics computation to read `scaledown` from traces and compute the run-level aggregates

5. **Results analyzer**

* Add “Compression” section + optional columns

---

[1]: https://docs.scaledown.ai/quickstart?utm_source=chatgpt.com "Quickstart - ScaleDown"
[2]: https://docs.scaledown.ai/Documentation?utm_source=chatgpt.com "Documentation"
