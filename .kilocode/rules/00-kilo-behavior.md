# Kilo Code — Project Rules (Behavioral)

These rules exist to reduce churn and keep work reviewable. Follow them unless explicitly instructed otherwise.

## 1) Batch edits (reduce focus/tab churn)
- **Plan first**, then apply changes in as few passes as possible.
- Avoid hopping between files repeatedly.
- Prefer making a cohesive set of edits per file before moving on.
- If a change touches many files, group them and do them in a deliberate sequence.

## 2) Command discipline (prefer repo wrappers)
When running Python, tests, linting, formatting, typechecking, or the UI:
- Prefer `make <target>` for common workflows.
- Otherwise use:
  - `./scripts/py ...` for ad-hoc Python commands
  - `./scripts/pip ...` for dependency management
- Avoid running `python`, `pip`, `pytest`, `ruff`, or `streamlit` directly unless I explicitly ask.

## 3) Git behavior
- Do **not** auto-commit.
- Do **not** add “Co-authored-by” or similar attribution lines.
- At the end of work, provide **suggested commits** (message + files included) so I can apply them.
