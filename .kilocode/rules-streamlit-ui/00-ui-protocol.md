# Streamlit UI/UX Protocol

This mode is for UI/UX work in the Streamlit evaluation app.

## Default output style
When asked to improve UI/UX:
1) **User goal**: what the user is trying to accomplish
2) **Current friction**: what’s confusing / slow / error-prone
3) **Proposed change**: minimal set of UI changes
4) **Acceptance criteria**: how we know it’s better
5) **Implementation plan**: files touched + key functions

## Guardrails
- Do not modify core RAG pipeline logic unless explicitly requested.
- Prefer small diffs; no drive-by formatting or refactors.
- Provide good empty-states (no data, no selection, no results).
- Validate inputs early; surface errors in the UI with actionable messages.

## UI quality checklist (use implicitly)
- Clear page title + short description of purpose
- Primary actions obvious (buttons near inputs, consistent placement)
- Use forms for multi-input actions; avoid rerun confusion
- Avoid excessive scrolling: use tabs/expanders wisely
- Show progress / spinners for slow operations
- Make “what happened” obvious (status messages, success/failure)
