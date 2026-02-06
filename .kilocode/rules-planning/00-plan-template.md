# Planning / Design Output Contract

This mode produces actionable plans/designs. No code edits.

## Required output sections
1) **Goal**
2) **Non-goals**
3) **Assumptions** (explicit; keep short)
4) **Proposed approach**
   - architecture sketch (mermaid if helpful)
   - key data flows / state transitions
5) **Work breakdown**
   - milestones or phases
   - each item: scope + files/areas touched + owner type (backend/ui/infra)
6) **Interfaces & contracts**
   - APIs, schemas, config knobs, CLI flags, storage changes
7) **Risks & mitigations**
8) **Observability**
   - logs/metrics/traces; success signals
9) **Rollout & rollback**
10) **Open questions** (only if genuinely unresolved)

## Guardrails
- Do not write diffs or code.
- Prefer minimal viable design; avoid over-abstraction.
- If multiple options exist, present 2 options max and recommend one.
