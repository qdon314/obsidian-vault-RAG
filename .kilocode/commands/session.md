---
description: Start a task with the repo session contract and pick the right mode.
arguments:
  - task
mode: implementation
---

Session contract for this repo:

Mode selection:
- Architecture questions or refactors → Architecture / Systems Review
- Failures, regressions, or unclear behavior → Debug / Forensics
- Clear, agreed execution → Implementation Executor
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
2) Choose the best mode explicitly (switch if needed).
3) Proceed according to that mode.

Current task (may be empty): $ARGUMENTS
