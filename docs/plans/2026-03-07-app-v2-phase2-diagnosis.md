# Results Analyzer v2 — Phase 2: Deterministic Diagnosis (already in Phase 1)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** Phase 2 tasks (Tasks 12–15: `stage_attribution`, `diagnostics`, `health`, `slices`) are included in the Phase 1 plan because they are dependencies of `build_bundle()`.

See [2026-03-07-app-v2-phase1-engine-backbone.md](2026-03-07-app-v2-phase1-engine-backbone.md), Tasks 12–15.

**Phase 2 exit criterion:** The engine is useful without Streamlit. Run this to confirm:

```bash
./scripts/py -c "
from pathlib import Path
from eval.app_v2.engine.loaders.bundle import build_bundle
from eval.app_v2.engine.domain.enums import Severity

bundle = build_bundle(Path('eval/runs/run_2026_02_12T20-40'))
print(f'Queries: {len(bundle.queries)}')
print(f'Recall@10: {bundle.health.headline_recall_at_10:.3f}')
print(f'Dominant failure: {bundle.health.dominant_failure_mode}')
print(f'Critical queries: {bundle.health.severity_counts.get(Severity.CRITICAL, 0)}')
"
```

Expected output: readable summary with at least one dominant failure mode shown.

## Additional Phase 2 Task: `engine/derived/__init__.py` exports

After Tasks 12–15 pass, ensure the derived package exposes clean public imports:

```python
# eval/app_v2/engine/derived/__init__.py
from eval.app_v2.engine.derived.diagnostics import analyze_queries, build_query_diagnostic
from eval.app_v2.engine.derived.health import build_health
from eval.app_v2.engine.derived.slices import build_slice_table
from eval.app_v2.engine.derived.stage_attribution import classify_query

__all__ = [
    "analyze_queries",
    "build_query_diagnostic",
    "build_health",
    "build_slice_table",
    "classify_query",
]
```

```bash
git add eval/app_v2/engine/derived/__init__.py
git commit -m "chore(app-v2): export derived-layer public API"
```
