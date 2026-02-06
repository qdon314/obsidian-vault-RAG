# Documentation Accuracy & Style

This mode edits docs only. Keep it grounded and consistent.

## Hard accuracy rules
- Never claim a feature exists unless verified in the repository.
- If uncertain, phrase as “planned” or “future work,” or ask to confirm.
- Prefer describing current behavior over aspirational behavior.

## Style conventions
- Prefer **mermaid** for diagrams (flows, architecture).
 - mermaid code blocks must be wrapped in ```mermaid
- Keep “How to run” sections aligned with repo commands:
  - prefer `make <target>` and `./scripts/py` / `./scripts/pip`

## Scope discipline
- Do not edit code.
- Do not broaden docs beyond what changed.
- Add a brief “What changed” note when updating docs after code edits.

## Suggested structure (when relevant)
- Overview
- How to run (commands)
- Architecture (1 diagram + short bullets)
- Evaluation (how to run, what metrics mean)
- Troubleshooting (common failure modes)
