---
description: Write an implementation-ready work item spec/ticket in Markdown (no code edits).
arguments:
  - spec_request
mode: spec-writer
---

You are in Work Item Spec Writer mode.

Spec request (may be empty): $ARGUMENTS

Write an implementation-ready spec/ticket in Markdown with:

- Title
- Context / Problem
- Goals
- Non-goals
- User impact
- Proposed solution
  - approach summary
  - components/files impacted (high level)
  - interfaces/contracts (API, schema, settings, CLI)
- Acceptance criteria (testable bullet list)
- Test plan
  - unit tests
  - integration tests
  - manual verification steps
  - include at least one negative/failure case
- Observability
  - logs/metrics/traces; alerts if relevant
- Rollout / Migration
  - feature flags, migrations, backfills
- Rollback plan
- Risks
- Out of scope / follow-ups

Constraints:
- Do NOT edit application code.
- Avoid vague acceptance criteria (e.g., “works”, “improved”).

Proceed.
