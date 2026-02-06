# Work Item Spec Contract

This mode outputs implementation-ready tickets/specs.

## Required sections (Markdown)
- **Title**
- **Context / Problem**
- **Goals**
- **Non-goals**
- **User impact**
- **Proposed solution**
  - approach summary
  - components/files impacted (high level)
  - interfaces/contracts (API, schema, settings)
- **Acceptance criteria** (bullet list; testable)
- **Test plan**
  - unit tests
  - integration tests
  - manual verification steps
- **Observability**
  - logs/metrics/traces; alerts if relevant
- **Rollout / Migration**
  - feature flags, backfills, migrations
- **Rollback plan**
- **Risks**
- **Out of scope / follow-ups**

## Guardrails
- Do not change application code.
- Avoid vague acceptance criteria (e.g. “works”, “improved”).
- Include at least one negative test / failure case in Test plan.
