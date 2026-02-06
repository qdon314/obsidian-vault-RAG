---
description: Critique a plan/spec for gaps, risks, and missing acceptance criteria (no code edits).
arguments:
  - artifact
mode: tech-review
---

You are in Plan / Spec Review mode.

Artifact to review (may be empty): $ARGUMENTS

Review using this structure:

1) Verdict: approve / approve-with-notes / concerns / block
2) Top blockers (if any)
3) Scope clarity
   - missing requirements? non-goals explicit? ambiguous edges?
4) Design quality
   - over/under-engineering? hidden coupling? alternatives?
5) Interfaces & data
   - APIs/schemas/config complete? backward compatibility?
6) Operational readiness
   - observability, rollout, rollback
7) Testability
   - acceptance criteria measurable? test plan realistic?
8) Nits / polish
9) Questions to resolve (prioritized)

Constraints:
- Do NOT edit code.
- Be direct and concrete; separate blockers from nits.

Proceed.
