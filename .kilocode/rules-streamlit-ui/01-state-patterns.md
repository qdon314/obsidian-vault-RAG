# Streamlit State & Flow Patterns

## State management
- Store long-lived UI state in `st.session_state` with explicit keys.
- Initialize session state keys once (guard with `if key not in st.session_state`).
- Prefer pure helper functions to compute derived values.

## Rerun control
- Use `st.form` + `form_submit_button` for actions that should run once.
- Avoid accidental multi-run of expensive calls.
- Use caching (`st.cache_data` / `st.cache_resource`) for stable expensive computations where safe.

## Result presentation
- Prefer: summary first → details expandable
- Provide "copy" affordances (code blocks for JSON, concise tables)
- Make errors readable: show cause + next step

## Commands
- Prefer repo wrappers (e.g., `make results`) when running the UI.
