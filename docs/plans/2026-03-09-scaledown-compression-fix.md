# ScaleDown Compression Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two bugs in the ScaleDown compressor (nested response access, full-prompt compression instead of context-only), and add a before/after context viewer in the eval app's Forensics page.

**Architecture:** Remove the instruction preamble from `ContextBuilder._render_context` (it's redundant with the Generator's system prompt and was causing the compressor to receive instructional text it shouldn't touch). Fix the ScaleDown adapter to read from `response["results"]` and use the API-reported token counts. Store `context_before`/`context_after` in the compression trace metadata, then render them in the Forensics page.

**Tech Stack:** Python frozen dataclasses, httpx (ScaleDown HTTP), Streamlit (eval app), pytest.

---

### Task 1: Remove preamble from SimpleContextBuilder

**Files:**
- Modify: `src/rag/adapters/context_building/simple_context_builder.py:106-127`
- Test: `tests/adapters/test_simple_context_builder.py`

**Context:** `_render_context` currently prepends two instruction lines before `CONTEXT:\n`. These duplicate the Generator's system prompt and pollute what the compressor sees. Remove them; keep the `CONTEXT:\n` section label and all chunk formatting unchanged.

**Step 1: Run the existing test that will need updating**

```bash
./scripts/py -m pytest tests/adapters/test_simple_context_builder.py::TestSimpleContextBuilderRenderedContext::test_rendered_context_has_instructions -v
```

Expected: PASS (it currently checks for `"CONTEXT"` — confirm it does pass before touching code).

**Step 2: Update the test to remove the instruction-prose assertion**

In `tests/adapters/test_simple_context_builder.py`, update `test_rendered_context_has_instructions`:

```python
def test_rendered_context_has_instructions(self):
    """Rendered context includes the CONTEXT section header."""
    builder = SimpleContextBuilder()
    candidates = [make_candidate()]

    pack = builder.build("query", candidates, token_budget=10000)

    assert "CONTEXT" in pack.rendered_context
    assert "[1]" in pack.rendered_context
```

Remove: the `assert "context" in pack.rendered_context.lower()` line (the section header "CONTEXT:" satisfies the uppercase assert; the instruction prose is intentionally gone).

**Step 3: Run the updated test to confirm it still passes (instruction lines still present)**

```bash
./scripts/py -m pytest tests/adapters/test_simple_context_builder.py::TestSimpleContextBuilderRenderedContext::test_rendered_context_has_instructions -v
```

Expected: PASS (nothing changed in production code yet).

**Step 4: Remove the two instruction lines from `_render_context`**

In `src/rag/adapters/context_building/simple_context_builder.py`, `_render_context` method:

```python
def _render_context(
    self, chunks: Sequence[Chunk], ordered_scores: Sequence[Candidate] | None = None
) -> str:
    lines: list[str] = []
    lines.append("CONTEXT:\n")  # Keep section label; instruction prose moved to Generator

    for i, ch in enumerate(chunks, start=1):
        lines.append(f"[{i}]")
        title = ch.metadata.get("title")
        uri = ch.metadata.get("uri") or ch.metadata.get("source_uri")
        if title or uri:
            lines.append(f"Source: {title or ''} {uri or ''}".strip())
        lines.append(ch.text.strip())
        lines.append("")

    return "\n".join(lines).strip() + "\n"
```

The two lines removed:
```python
# REMOVE these two:
lines.append("You are given CONTEXT chunks from a document corpus. Answer the QUESTION using only the CONTEXT.\n")
lines.append("If the answer is not supported by the CONTEXT, say you don't know.\n")
```

**Step 5: Run full context builder tests**

```bash
./scripts/py -m pytest tests/adapters/test_simple_context_builder.py -v
```

Expected: all PASS.

**Step 6: Commit**

```bash
git add src/rag/adapters/context_building/simple_context_builder.py \
        tests/adapters/test_simple_context_builder.py
git commit -m "refactor(context-builder): move instruction preamble out of rendered_context

Preamble was redundant with the Generator system prompt and caused the
compressor to receive instructional text. Keep CONTEXT: section label only."
```

---

### Task 2: Remove preamble from PropositionAwareContextBuilder

**Files:**
- Modify: `src/rag/adapters/context_building/propositional_context_builder.py:249-287`

**Context:** `PropositionAwareContextBuilder._render_context` has the same two instruction lines as SimpleContextBuilder. Same fix.

**Step 1: Apply the removal**

In `src/rag/adapters/context_building/propositional_context_builder.py`, update `_render_context`:

```python
def _render_context(
    self,
    chunks: Sequence[Chunk],
    ordered: Sequence[Candidate],
    *,
    token_budget: int,
) -> str:
    """Render the final context string sent to the LLM."""
    lines: list[str] = []
    lines.append("CONTEXT:\n")  # Instruction prose lives in the Generator system prompt

    score_by_id: dict[str, float] = {}
    if self.include_scores:
        def candidate_key(c: Candidate) -> float:
            return c.rerank_score if c.rerank_score is not None else c.score
        for c in ordered:
            score_by_id[c.chunk.chunk_id] = candidate_key(c)

    for i, ch in enumerate(chunks, start=1):
        header = f"[{i}]"
        if self.include_scores and ch.chunk_id in score_by_id:
            header += f" score={score_by_id[ch.chunk_id]:.4f}"
        lines.append(header)

        title = ch.metadata.get("title")
        uri = ch.metadata.get("uri") or ch.metadata.get("source_uri")
        if title or uri:
            lines.append(f"Source: {title or ''} {uri or ''}".strip())

        lines.append(self._render_chunk_text(ch))
        lines.append("")

    return "\n".join(lines).strip() + "\n"
```

**Step 2: Run all tests to confirm nothing broken**

```bash
./scripts/py -m pytest tests/ -v --tb=short
```

Expected: all PASS.

**Step 3: Commit**

```bash
git add src/rag/adapters/context_building/propositional_context_builder.py
git commit -m "refactor(context-builder): remove instruction preamble from PropositionAwareContextBuilder"
```

---

### Task 3: Fix ScaleDown adapter — response access and token counts

**Files:**
- Modify: `src/rag/adapters/compression/scaledown.py:66-104`

**Context:** Two bugs in `compress()`:
1. `data.get("compressed_prompt")` reads from the top-level dict, but the API nests results under `data["results"]`.
2. Token counts come from our local estimate. Use the API-reported values (`original_prompt_tokens`, `compressed_prompt_tokens`, `compression_ratio`) for accuracy. `savings_pct = 1 - compression_ratio`.

The API response shape (reference):
```python
{
  "results": {
    "success": True,
    "compressed_prompt": "CONTEXT:\n[1]...",  # compressed chunks
    "original_prompt": "...",                  # the query we sent as "prompt"
    "original_prompt_tokens": 852,
    "compressed_prompt_tokens": 794,
    "compression_ratio": 0.931,               # compressed/original; savings = 1 - ratio
  },
  "model_used": "gpt-4-turbo",
  "total_original_tokens": 852,
  "total_compressed_tokens": 794,
  ...
}
```

**Step 1: Write a unit test for the fixed parsing**

Create `tests/adapters/test_scaledown_compressor.py`:

```python
"""Tests for ScaleDownCompressor response parsing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.adapters.compression.scaledown import ScaleDownCompressor
from rag.domain.models import ContextPack


def _make_pack(text: str = "CONTEXT:\n[1]\nsome chunk text\n") -> ContextPack:
    return ContextPack(
        query="test query",
        chunks=(),
        rendered_context=text,
        citations=(),
        token_budget=2000,
        metadata={"tokens_used_est": 852},
    )


SAMPLE_API_RESPONSE = {
    "results": {
        "success": True,
        "compressed_prompt": "CONTEXT:\n[1]\ncompressed chunk text\n",
        "original_prompt": "test query",
        "original_prompt_tokens": 852,
        "compressed_prompt_tokens": 794,
        "compression_ratio": 0.9319,
    },
    "model_used": "gpt-4-turbo",
    "total_original_tokens": 852,
    "total_compressed_tokens": 794,
    "num_pairs_processed": 1,
    "successful": True,
    "latency_ms": 367,
}


class TestScaleDownCompressorParsing:
    def _compressor(self) -> ScaleDownCompressor:
        return ScaleDownCompressor(api_key="test-key")

    def _mock_response(self, data: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = data
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_reads_compressed_text_from_results(self):
        """compressed_prompt is read from response["results"], not top-level."""
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", return_value=self._mock_response(SAMPLE_API_RESPONSE)):
            result = compressor.compress(pack, query="test query")

        assert result.successful is True
        assert result.context_pack.rendered_context == "CONTEXT:\n[1]\ncompressed chunk text\n"

    def test_uses_api_token_counts(self):
        """tokens_before and tokens_after come from the API response."""
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", return_value=self._mock_response(SAMPLE_API_RESPONSE)):
            result = compressor.compress(pack, query="test query")

        assert result.tokens_before == 852
        assert result.tokens_after == 794

    def test_savings_pct_derived_from_compression_ratio(self):
        """savings_pct = 1 - compression_ratio."""
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", return_value=self._mock_response(SAMPLE_API_RESPONSE)):
            result = compressor.compress(pack, query="test query")

        expected_savings = 1.0 - 0.9319
        assert abs(result.savings_pct - expected_savings) < 1e-4

    def test_compression_ratio_in_extra(self):
        """compression_ratio is surfaced in extra for observability."""
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", return_value=self._mock_response(SAMPLE_API_RESPONSE)):
            result = compressor.compress(pack, query="test query")

        assert "compression_ratio" in result.extra
        assert abs(result.extra["compression_ratio"] - 0.9319) < 1e-4

    def test_fail_open_on_error(self):
        """On network error, returns original pack with successful=False."""
        import httpx
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            result = compressor.compress(pack, query="test query")

        assert result.successful is False
        assert result.context_pack is pack
        assert result.tokens_before == result.tokens_after
```

**Step 2: Run tests to confirm they fail**

```bash
./scripts/py -m pytest tests/adapters/test_scaledown_compressor.py -v
```

Expected: FAIL (current code reads from wrong dict level).

**Step 3: Fix the `compress` method**

Replace the success-path extraction block in `src/rag/adapters/compression/scaledown.py` (lines 66-105):

```python
data: dict = response.json()
latency_ms = int((time.perf_counter() - t0) * 1000)

results: dict = data.get("results", {})
compressed_text: str = results.get("compressed_prompt", "")
tokens_after = int(results.get("compressed_prompt_tokens", tokens_before))
original_tokens = int(results.get("original_prompt_tokens", tokens_before))
compression_ratio = float(results.get("compression_ratio", 1.0))
savings_pct = 1.0 - compression_ratio

updated_pack = replace(
    context_pack,
    rendered_context=compressed_text,
    metadata={**dict(context_pack.metadata), "tokens_used_est": tokens_after},
)

# Capture remaining response fields as extra metadata
known_results_keys = {
    "compressed_prompt",
    "original_prompt",
    "original_prompt_tokens",
    "compressed_prompt_tokens",
    "compression_ratio",
    "success",
}
extra: dict = {
    **{k: v for k, v in results.items() if k not in known_results_keys},
    **{k: v for k, v in data.items() if k != "results"},
    "compression_ratio": compression_ratio,  # always surfaced for observability
}

return CompressionResult(
    context_pack=updated_pack,
    successful=True,
    tokens_before=original_tokens,
    tokens_after=tokens_after,
    savings_pct=savings_pct,
    latency_ms=latency_ms,
    adapter="scaledown",
    extra=extra,
)
```

Note: also move `latency_ms` assignment above the extraction block (currently it's after the `httpx.post` call — move it to right after `response.json()`).

**Step 4: Run new tests to confirm they pass**

```bash
./scripts/py -m pytest tests/adapters/test_scaledown_compressor.py -v
```

Expected: all PASS.

**Step 5: Run full test suite**

```bash
./scripts/py -m pytest tests/ -v --tb=short
```

Expected: all PASS.

**Step 6: Commit**

```bash
git add src/rag/adapters/compression/scaledown.py \
        tests/adapters/test_scaledown_compressor.py
git commit -m "fix(scaledown): read from response[results] and use API-reported token counts

- Access compressed_prompt, token counts, and compression_ratio from
  response['results'] (was reading from top-level dict)
- Use original_prompt_tokens / compressed_prompt_tokens from API for
  accurate before/after counts
- Derive savings_pct from compression_ratio (1 - ratio)
- Surface compression_ratio in extra for observability"
```

---

### Task 4: Store context_before / context_after in trace metadata

**Files:**
- Modify: `src/rag/app/query_runner.py:87-94`

**Context:** The forensics page needs the pre- and post-compression context strings. Store them in `compression_meta` so they flow into `QueryTrace.metadata["compression"]` and get serialized to JSONL automatically. No domain model changes required.

**Step 1: Write a test**

Add to `tests/adapters/test_scaledown_compressor.py` (or a new `tests/app/test_query_runner_compression.py` if you prefer isolation):

```python
def test_compression_meta_includes_context_texts():
    """query_runner stores context_before and context_after in compression_meta."""
    # This is an integration-style test; mock the compressor directly.
    from unittest.mock import MagicMock
    from rag.domain.models import CompressionResult, ContextPack

    original_text = "CONTEXT:\n[1]\noriginal chunk\n"
    compressed_text = "CONTEXT:\n[1]\ncompressed chunk\n"

    original_pack = ContextPack(
        query="q", chunks=(), rendered_context=original_text,
        citations=(), token_budget=1000, metadata={"tokens_used_est": 100},
    )
    compressed_pack = ContextPack(
        query="q", chunks=(), rendered_context=compressed_text,
        citations=(), token_budget=1000, metadata={"tokens_used_est": 80},
    )
    mock_result = CompressionResult(
        context_pack=compressed_pack,
        successful=True,
        tokens_before=100,
        tokens_after=80,
        savings_pct=0.2,
        latency_ms=100,
        adapter="scaledown",
    )

    mock_compressor = MagicMock()
    mock_compressor.compress.return_value = mock_result

    # Call the relevant section of run_query directly via helper
    from rag.app.query_runner import _apply_compression  # we'll extract this
    meta = _apply_compression(original_pack, mock_compressor, query="q")

    assert meta["context_before"] == original_text
    assert meta["context_after"] == compressed_text
```

NOTE: if you don't want to extract a helper, test this indirectly by inspecting `QueryTrace.metadata["compression"]` after a full `run_query` call with a mock compressor.

**Step 2: Run to confirm failure**

```bash
./scripts/py -m pytest tests/ -k "context_before" -v
```

Expected: FAIL (fields not yet stored).

**Step 3: Update `query_runner.py` compression block**

In `src/rag/app/query_runner.py`, update lines 87-94:

```python
# Compression (optional — skipped when no compressor is provided)
t_compress_ms = 0
compression_meta: dict[str, Any] | None = None
if compressor is not None:
    context_before = context.rendered_context          # capture before overwrite
    compression_result = compressor.compress(context, query=query, metadata=metadata)
    t_compress_ms = compression_result.latency_ms
    context = compression_result.context_pack
    compression_meta = compression_result.to_metadata_dict()
    compression_meta["context_before"] = context_before
    compression_meta["context_after"] = context.rendered_context
```

**Step 4: Run tests**

```bash
./scripts/py -m pytest tests/ -v --tb=short
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/rag/app/query_runner.py
git commit -m "feat(query-runner): store context_before/after in compression trace metadata"
```

---

### Task 5: Add context window viewer to Forensics page

**Files:**
- Modify: `eval/app_v2/ui/pages/forensics.py`

**Context:** Add an expander "Context window (compression)" below the chunk viewer. It appears only when `trace.raw_data["metadata"]["compression"]` contains `context_before`/`context_after`. Show two tabs: Before / After, each a `st.text_area`.

**Step 1: Write the widget function**

Add to `eval/app_v2/ui/pages/forensics.py` (before `render()`):

```python
def _render_compression_context(trace_raw_data: dict, *, key_prefix: str = "") -> None:
    """Render before/after context text areas when compression metadata is present."""
    compression = (trace_raw_data.get("metadata") or {}).get("compression") or {}
    context_before: str | None = compression.get("context_before")
    context_after: str | None = compression.get("context_after")

    if not context_before and not context_after:
        return

    with st.expander("Context window (compression)", expanded=False):
        tab_before, tab_after = st.tabs(["Before", "After"])
        with tab_before:
            st.text_area(
                "",
                value=context_before or "— not captured —",
                height=300,
                disabled=True,
                key=f"{key_prefix}_ctx_before",
                label_visibility="collapsed",
            )
        with tab_after:
            st.text_area(
                "",
                value=context_after or "— not captured —",
                height=300,
                disabled=True,
                key=f"{key_prefix}_ctx_after",
                label_visibility="collapsed",
            )
```

**Step 2: Call it from `render()`**

In `render()`, after the `render_retrieved_chunks(...)` call (line 56), add:

```python
# Compression context viewer (only shown when compression ran and text was captured)
if r.trace is not None:
    _render_compression_context(r.trace.raw_data, key_prefix=qid)
```

**Step 3: Verify import — `QueryTrace` is already imported via `AnalyzedQuery`; no new imports needed.**

**Step 4: Smoke-test the app manually**

```bash
make results
```

Navigate to Forensics page. For a run without compression: no "Context window" expander appears. For a run with compression and the new trace metadata: expander appears with Before/After tabs.

(Automated Streamlit UI tests are out of scope — manual smoke test is sufficient here.)

**Step 5: Commit**

```bash
git add eval/app_v2/ui/pages/forensics.py
git commit -m "feat(eval-app): add before/after context viewer to Forensics page

Shows compression context window expander when trace metadata contains
context_before/context_after from ScaleDown compression run."
```

---

### Task 6: Final validation

**Step 1: Run full test suite + lint**

```bash
./scripts/py -m pytest tests/ -v --tb=short
make lint
make typecheck
```

Expected: all PASS, no type errors.

**Step 2: Check test coverage for new compressor tests**

```bash
./scripts/py -m pytest tests/adapters/test_scaledown_compressor.py -v
```

All 5 tests should pass.

**Step 3: Review the diff end-to-end**

```bash
git diff main...HEAD --stat
```

Expected changed files:
- `src/rag/adapters/context_building/simple_context_builder.py`
- `src/rag/adapters/context_building/propositional_context_builder.py`
- `src/rag/adapters/compression/scaledown.py`
- `src/rag/app/query_runner.py`
- `eval/app_v2/ui/pages/forensics.py`
- `tests/adapters/test_simple_context_builder.py`
- `tests/adapters/test_scaledown_compressor.py` (new)
