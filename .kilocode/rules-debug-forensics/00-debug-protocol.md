# Debug & Forensics Protocol

This mode is for diagnosis. Default to evidence and hypotheses, not changes.

## Required structure
For any bug/failure/regression, produce:

1) **Symptoms** (what is observed)
2) **Repro** (best minimal command or steps)
3) **Hypotheses (2–4)** ranked by likelihood, each with:
   - Evidence supporting it
   - What would falsify it
   - Cheapest next experiment
4) **Next actions** (1–3 concrete steps)

## Guardrails
- Do **not** refactor.
- Do **not** apply a “fix” until a root cause is identified.
- Prefer minimal, reversible probes (temporary logs, assertions, tracing).
- If you add instrumentation:
  - keep it scoped
  - call out whether it should be removed after confirmation

## Repro guidance (prefer these)
- Prefer `make <target>` where available.
- Otherwise prefer `./scripts/py -m pytest ...` etc.
- If narrowing tests, propose the smallest selector (`-k`, single file, single test).

## Stop conditions
If you cannot form >1 plausible hypothesis, pause and request more data (logs, failing output, exact command, expected vs actual).
