# Verdict Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address three issues from code review of the eval verdict changeset (spec 03): add a `latest` symlink in `run_eval.py`, skip threshold checks when data is absent instead of defaulting to 0.0, and document the `./scripts/py` vs bare `python` CI ambiguity for later resolution.

**Architecture:** Minimal surgical edits to existing files. The `latest` symlink is created in `save_run()`. The verdict check builder becomes a list that conditionally appends checks. A `TODO` comment in `ci.yml` captures the scripts/py question.

**Tech Stack:** Python 3.11, pathlib, frozen dataclasses

---

### Task 1: Add `latest` symlink to `save_run()`

**Files:**
- Modify: `src/rag/eval/harness.py:476-480` (end of `save_run`)
- Test: `tests/eval/test_verdict.py` (no new test needed — symlink is filesystem plumbing)

**Step 1: Write the failing test**

```python
# tests/eval/test_save_run_latest_symlink.py
def test_save_run_creates_latest_symlink(tmp_path):
    """save_run should create a 'latest' symlink pointing to the run directory."""
```

Actually — this is pure filesystem plumbing (create symlink after save). A unit test would just test `pathlib.symlink_to`. Instead we verify manually and rely on the existing `save_run` integration behavior.

**Step 1: Add symlink creation at end of `save_run` in `harness.py`**

At the end of `save_run()`, after writing `metrics.json` and building the artifacts dict, add:

```python
    # Maintain a stable "latest" symlink so verdict scripts can use
    # --current eval/runs/latest/ without knowing the timestamp.
    latest = output_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(output_dir.name)
```

Key details:
- `output_dir.parent` is `eval/runs/`, `output_dir.name` is `run_2026_02_08T...`
- Uses a **relative** symlink target (`output_dir.name` not `output_dir`) so the symlink works if the repo is moved
- `unlink()` before `symlink_to()` handles re-runs safely
- Placed before the `return replace(...)` line

**Step 2: Remove the `resolve_run_dir` fallback from `eval/scripts/verdict.py`**

The `resolve_run_dir` function currently has a compatibility fallback that picks the lexicographically-last `run_*` directory when `latest` doesn't exist. Now that `save_run` creates the symlink, this fallback is unnecessary and its implicit sorting assumption is fragile. Simplify to:

```python
def resolve_run_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Run directory not found: {candidate}")
```

**Step 3: Run tests**

Run: `./scripts/py -m pytest tests/eval/ -v`
Expected: All existing tests still pass (none depend on `resolve_run_dir` fallback)

**Step 4: Suggested commit**

```
fix(eval): add latest symlink in save_run, remove fragile fallback

save_run() now creates eval/runs/latest -> run_{timestamp} so
verdict scripts resolve --current eval/runs/latest/ reliably.

Removes the lexicographic-sort fallback from eval/scripts/verdict.py
that could pick the wrong directory with non-timestamp run names.

Files: src/rag/eval/harness.py, eval/scripts/verdict.py
```

---

### Task 2: Skip threshold checks when data is absent (not default to 0.0)

**Files:**
- Modify: `src/rag/eval/verdict.py:121-189` (`compute_verdict` checks tuple)
- Modify: `tests/eval/test_verdict.py` (update `_make_run` helper, add new test)

**Step 1: Write the failing test**

Add to `tests/eval/test_verdict.py`:

```python
def test_verdict_skips_checks_for_absent_data() -> None:
    """Retrieval-only runs with no answer_quality/latency should omit those checks."""
    results = tuple(
        _make_result(f"q{i+1}", None) for i in range(5)
    )
    current = EvalRun(
        meta=EvalRunMeta(
            run_id="retrieval-only",
            started_at=datetime.now(UTC),
            queries_path="eval/datasets/curated_queries.jsonl",
        ),
        results=results,
        aggregates=EvalAggregates(
            overall=RetrievalSummary(
                num_queries=5,
                avg_retrieved=2.0,
                recall_at_k={10: 0.80},
                ndcg_at_k={10: 0.70},
                mrr=0.60,
                map=0.0,
            ),
            answer_quality=None,
            latency_ms=None,
        ),
    )
    verdict = compute_verdict(current, None, VerdictThresholds())

    assert verdict.decision == Decision.SHIP
    check_names = {c.name for c in verdict.checks}
    # Retrieval checks must be present
    assert "recall@10 >= min_recall_at_10" in check_names
    assert "ndcg@10 >= min_ndcg_at_10" in check_names
    assert "mrr >= min_mrr" in check_names
    # Answer quality and latency checks must be absent
    assert "avg_hallucination_severity <= max_avg_hallucination_severity" not in check_names
    assert "evidence_bounded_rate >= min_evidence_bounded_rate" not in check_names
    assert "latency_p95_ms <= max_latency_p95_ms" not in check_names
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/eval/test_verdict.py::test_verdict_skips_checks_for_absent_data -v`
Expected: FAIL — currently the function always emits 8 checks and the hallucination/evidence/latency checks would use 0.0 defaults.

**Step 3: Modify `compute_verdict` in `src/rag/eval/verdict.py`**

Change the checks construction from a static tuple to a list that conditionally appends. The three retrieval checks (recall, ndcg, mrr) are always present. The three answer-quality checks (hallucination, evidence_bounded, latency) are only present when the underlying data exists. The two behavioral-rate checks (unsafe_miss, abstain_bad) are only present when outcome labels exist.

Replace lines 121-189 (the `checks = (...)` block) with:

```python
    # --- Build checks list: always include retrieval, conditionally include answer/latency ---
    checks: list[ThresholdCheck] = [
        ThresholdCheck(
            name="recall@10 >= min_recall_at_10",
            passed=_safe_value(overall.recall_at_k.get(10)) >= thresholds.min_recall_at_10,
            current=_safe_value(overall.recall_at_k.get(10)),
            threshold=thresholds.min_recall_at_10,
            baseline=(
                _safe_value(baseline_overall.recall_at_k.get(10))
                if baseline_overall
                else None
            ),
        ),
        ThresholdCheck(
            name="ndcg@10 >= min_ndcg_at_10",
            passed=_safe_value(overall.ndcg_at_k.get(10)) >= thresholds.min_ndcg_at_10,
            current=_safe_value(overall.ndcg_at_k.get(10)),
            threshold=thresholds.min_ndcg_at_10,
            baseline=(_safe_value(baseline_overall.ndcg_at_k.get(10)) if baseline_overall else None),
        ),
        ThresholdCheck(
            name="mrr >= min_mrr",
            passed=_safe_value(overall.mrr) >= thresholds.min_mrr,
            current=_safe_value(overall.mrr),
            threshold=thresholds.min_mrr,
            baseline=_safe_value(baseline_overall.mrr) if baseline_overall else None,
        ),
    ]

    # Answer quality checks: only when judge data is present.
    halluc_val = answer_quality.get("avg_hallucination_severity_0_5")
    if halluc_val is not None:
        checks.append(
            ThresholdCheck(
                name="avg_hallucination_severity <= max_avg_hallucination_severity",
                passed=float(halluc_val) <= thresholds.max_avg_hallucination_severity,
                current=float(halluc_val),
                threshold=thresholds.max_avg_hallucination_severity,
                baseline=_safe_value(baseline_answer_quality.get("avg_hallucination_severity_0_5"))
                if baseline
                else None,
            )
        )

    eb_val = answer_quality.get("evidence_bounded_rate")
    if eb_val is not None:
        checks.append(
            ThresholdCheck(
                name="evidence_bounded_rate >= min_evidence_bounded_rate",
                passed=float(eb_val) >= thresholds.min_evidence_bounded_rate,
                current=float(eb_val),
                threshold=thresholds.min_evidence_bounded_rate,
                baseline=_safe_value(baseline_answer_quality.get("evidence_bounded_rate"))
                if baseline
                else None,
            )
        )

    # Latency check: only when pipeline latency was measured.
    p95_val = latency.get("p95")
    if p95_val is not None:
        checks.append(
            ThresholdCheck(
                name="latency_p95_ms <= max_latency_p95_ms",
                passed=float(p95_val) <= thresholds.max_latency_p95_ms,
                current=float(p95_val),
                threshold=thresholds.max_latency_p95_ms,
                baseline=_safe_value(baseline_latency.get("p95")) if baseline else None,
            )
        )

    # Behavioral outcome checks: only when outcome labels were computed.
    has_outcomes = any(r.outcome_label is not None for r in current.results)
    if has_outcomes:
        checks.append(
            ThresholdCheck(
                name="unsafe_miss_rate <= max_unsafe_miss_rate",
                passed=unsafe_miss_rate <= thresholds.max_unsafe_miss_rate,
                current=unsafe_miss_rate,
                threshold=thresholds.max_unsafe_miss_rate,
                baseline=None,
            )
        )
        checks.append(
            ThresholdCheck(
                name="abstain_bad_rate <= max_abstain_bad_rate",
                passed=abstain_bad_rate <= thresholds.max_abstain_bad_rate,
                current=abstain_bad_rate,
                threshold=thresholds.max_abstain_bad_rate,
                baseline=None,
            )
        )
```

Then update the references below from `checks` (was tuple) — the `failed_checks` line and the `Verdict(checks=checks, ...)` line need `tuple(checks)`:

Change:
```python
    failed_checks = [check for check in checks if not check.passed]
```
stays the same (works on list).

Change:
```python
        checks=checks,
```
to:
```python
        checks=tuple(checks),
```

**Step 4: Update existing test `test_verdict_ship_when_all_pass` to verify check count is 8 for full runs**

The existing tests create runs with `answer_quality` and `latency_ms` populated, and with `OutcomeLabel.SUCCESS_GROUNDED` labels, so they should still produce 8 checks. Add an assertion to `test_verdict_ship_when_all_pass`:

```python
    assert len(verdict.checks) == 8  # 3 retrieval + 2 answer quality + 1 latency + 2 behavioral
```

**Step 5: Run all verdict tests**

Run: `./scripts/py -m pytest tests/eval/test_verdict.py -v`
Expected: All 11 tests pass

**Step 6: Run mypy**

Run: `./scripts/py -m mypy src/rag/eval/verdict.py`
Expected: Success

**Step 7: Suggested commit**

```
fix(eval): skip verdict checks when underlying data is absent

Retrieval-only runs (no answer_quality, no latency_ms, no outcome
labels) now omit the corresponding threshold checks instead of
silently passing them with a 0.0 default. This prevents data
collection bugs from hiding behind a default-pass.

Files: src/rag/eval/verdict.py, tests/eval/test_verdict.py
```

---

### Task 3: Document CI `./scripts/py` ambiguity

**Files:**
- Modify: `.github/workflows/ci.yml` (add TODO comment at top of file)

**Step 1: Add a TODO block**

At the top of `ci.yml`, inside the existing comment area or right after `name: CI`, add:

```yaml
# TODO(ci-python-path): Both jobs use bare `python` / `pip` because
# setup-python puts the interpreter on $PATH directly, while
# ./scripts/py hardcodes .venv/bin/python for local conda isolation.
# Consider making ./scripts/py fall back to $PATH when .venv doesn't
# exist, so CI and local use the same entry point.
# See: CLAUDE.md "Python Environment" section.
```

Place this right above the `jobs:` key.

**Step 2: Suggested commit**

```
docs(ci): add TODO documenting scripts/py vs bare python ambiguity

Files: .github/workflows/ci.yml
```
