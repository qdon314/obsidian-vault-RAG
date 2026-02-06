---
description: Produce an actionable implementation plan/design (no code edits).
arguments:
  - topic
mode: planning
---

You are in Planning / Design mode.

Topic / request (may be empty): $ARGUMENTS

Produce an implementation plan using this structure:

1) Goal
2) Non-goals
3) Assumptions (explicit; keep short)
4) Proposed approach
   - architecture sketch (mermaid if helpful)
   - key flows / state transitions
5) Work breakdown
   - phases/milestones
   - each item: scope + files/areas touched + dependencies
6) Interfaces & contracts
   - APIs, schemas, config knobs, CLI flags, migrations/backfills
7) Risks & mitigations
8) Observability
   - logs/metrics/traces; success signals
9) Rollout & rollback
10) Open questions (only if truly blocking)

Constraints:
- Do NOT write code or diffs.
- If multiple options exist, present at most 2 and recommend one.
- State assumptions instead of asking questions unless truly blocking.

Proceed.
