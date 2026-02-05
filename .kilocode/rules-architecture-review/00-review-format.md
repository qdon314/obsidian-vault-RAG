# Architecture Review Protocol

This mode is for architectural reasoning and review. Default to analysis, not edits.

## Output format (required)
When asked to design/review a change, produce:

1) **Verdict**: approve / approve-with-notes / concerns / block
2) **Boundary impact**:
   - What layers/packages are affected (domain, ports, adapters, app)?
   - Any likely boundary violations?
3) **Dependency direction check**:
   - Confirm: domain does not import adapters
   - Confirm: ports remain interface-only (protocols), no concrete coupling
4) **File-level plan**:
   - Bullet list of files to touch + what change in each
5) **Risks & failure modes**:
   - What could break? How would we notice?
6) **Rollback plan**:
   - How to revert safely if it goes sideways

## Guardrails
- Do **not** edit code unless explicitly instructed.
- Prefer smallest viable change that preserves existing architecture.
- Do not invent new abstractions unless they eliminate duplicated logic in **2+** places or clarify a boundary.
- If a new config knob is proposed, state:
  - whether it belongs in `settings.toml` vs CLI flags vs hardcoded
  - why, and what the default should be

## “Stop and ask”
If any of these are unclear, ask before proposing a final design:
- intended scope (experiment vs production path)
- constraints (performance, backward compatibility, deployment)
- whether the change should be reversible behind a flag
