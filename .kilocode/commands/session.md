---
description: Start an execution task with the repo session contract (defaults to implementation).
arguments:
  - task
mode: implementation
---

Session contract for this repo.

Default mode: **Implementation Executor**.
This command is for work where direction is already clear.

If the task requires planning, design, or spec-writing, stop and use:
- `/plan` for implementation plans/designs
- `/spec` for work item specs/tickets
- `/review` to critique a plan or spec

Mode selection (override only if needed):
- Architecture questions or refactors → Architecture / Systems Review
- Failures, regressions, or unclear behavior → Debug / Forensics
- Metrics, evals, judges, retrieval quality → Evaluation / Signal Analysis
- Docs only → Documentation / Explanation

Workflow:
- Start with a short plan (file-level checklist).
- Batch edits; minimize file hopping and focus changes.
- Keep diffs minimal; no drive-by refactors or formatting.

Commands:
- Prefer `make <target>`.
- Otherwise use `./scripts/py ...` or `./scripts/pip ...`.

Git:
- Do not auto-commit.
- No co-author lines.
- Propose suggested commits (message + files) at the end.

Now:
1) Restate the task in 1–2 sentences.
2) Confirm or switch the mode explicitly.
3) Proceed according to that mode.

Current task (may be empty): $ARGUMENTS
