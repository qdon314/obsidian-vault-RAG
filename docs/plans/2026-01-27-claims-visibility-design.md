# Claims Visibility in Streamlit Eval App

## Problem

The groundedness judge computes detailed per-claim data (`AnswerClaim` objects with claim text, supported status, chunk_id, quote, and note), but only aggregate counts (`supported_claims`, `unsupported_claims`) are retained in `AnswerQualityMetrics`. The rich detail is discarded, making it impossible to:

- Debug why a specific answer hallucinated
- Identify patterns in unsupported claims
- Trace claims back to source chunks

## Solution

Persist the full `GroundednessJudgeResult` on `EvalResult` and surface claims in the Streamlit Query Explorer.

## Design

### 1. Data Model Changes

**File:** `src/rag/eval/harness.py`

Add field to `EvalResult`:

```python
@dataclass(frozen=True, slots=True)
class EvalResult(DataClassJsonMixin):
    # ... existing fields ...
    answer_metrics: AnswerQualityMetrics | None = None
    groundedness_result: GroundednessJudgeResult | None = None  # NEW
```

Backwards compatible: existing runs without this field load with `None`.

### 2. Harness Changes

**File:** `src/rag/eval/harness.py`

Update `evaluate_answer_quality()` to return both `AnswerQualityMetrics` and `GroundednessJudgeResult`:

```python
groundedness = groundedness_judge(answer, context)
metrics = AnswerQualityMetrics(...)
return metrics, groundedness  # Return both
```

Update caller in `run_full_eval()` to populate both fields on `EvalResult`.

### 3. Data Loading

**File:** `eval/app/results/adapters/filesystem_loader.py`

Import `GroundednessJudgeResult` and `AnswerClaim` from `src/rag/eval/judges` (avoid duplication, keep types in sync).

Handle deserialization:

```python
groundedness_data = data.get("groundedness_result")
groundedness_result = (
    GroundednessJudgeResult.from_dict(groundedness_data)
    if groundedness_data else None
)
```

### 4. Streamlit UI

**File:** `eval/app/results/ui/query_explorer.py`

Add "Claims" tab when drilling into a query result:

```python
with tabs["Claims"]:
    if result.groundedness_result and result.groundedness_result.claims:
        for claim in result.groundedness_result.claims:
            with st.expander(f"{'✓' if claim.supported else '✗'} {claim.claim[:80]}..."):
                st.markdown(f"**Supported:** {claim.supported}")
                if claim.chunk_id:
                    st.markdown(f"**Chunk ID:** `{claim.chunk_id}`")
                if claim.quote:
                    st.markdown(f"**Quote:** _{claim.quote}_")
                if claim.note:
                    st.markdown(f"**Note:** {claim.note}")
    else:
        st.info("No claims data available for this query.")
```

## Files to Modify

| File | Change |
|------|--------|
| `src/rag/eval/harness.py` | Add `groundedness_result` to `EvalResult`; update `evaluate_answer_quality()` return; update `run_full_eval()` |
| `eval/app/results/adapters/filesystem_loader.py` | Import judge types; deserialize `groundedness_result` |
| `eval/app/results/ui/query_explorer.py` | Add Claims tab with expanders |

## No Changes Needed

- `GroundednessJudgeResult` and `AnswerClaim` already have `DataClassJsonMixin`
- `AnswerQualityMetrics` unchanged (derived scores remain useful)
- Existing eval runs load fine (new field defaults to `None`)