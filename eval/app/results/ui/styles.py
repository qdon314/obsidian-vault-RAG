import streamlit as st


def inject_global_css(max_width_px: int = 1200) -> None:
    st.markdown(
        f"""
<style>
/* --- Layout: less empty space, more “app-like” --- */
.block-container {{
  padding-top: 2rem;
  padding-bottom: 2.5rem;
  max-width: {max_width_px}px;
}}

/* Sidebar spacing */
section[data-testid="stSidebar"] .block-container {{
  padding-top: 1.5rem;
}}

/* --- Typography: calmer headings, better reading --- */
h1, h2, h3 {{
  letter-spacing: -0.02em;
  line-height: 1.15;
}}
h1 {{ margin-bottom: 0.6rem; }}
h2 {{ margin-top: 1.4rem; margin-bottom: 0.5rem; }}
h3 {{ margin-top: 1.0rem; margin-bottom: 0.4rem; }}

/* Make body text a touch more readable without changing theme font */
html, body, [class*="css"] {{
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}}

/* --- Inputs: nicer focus ring (subtle) --- */
:focus-visible {{
  outline: 2px solid rgba(74, 158, 255, 0.55);
  outline-offset: 2px;
  border-radius: 6px;
}}

/* --- Buttons: slightly less “default Streamlit” --- */
button[kind="primary"] {{
  border-radius: 10px !important;
  font-weight: 600 !important;
}}
button[kind="secondary"] {{
  border-radius: 10px !important;
}}

/* --- Cards/containers: unify rounded corners + soft border --- */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stExpander"]) {{
  border-radius: 12px;
}}

/* Expanders: less harsh borders, more “panel” feel */
details[data-testid="stExpander"] {{
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.08);
  overflow: hidden;
}}
details[data-testid="stExpander"] > summary {{
  padding: 0.75rem 1rem;
}}
details[data-testid="stExpander"] > div {{
  padding: 0.25rem 1rem 0.75rem 1rem;
}}

/* --- Code blocks: smaller, nicer corners, less shouty --- */
code {{
  font-size: 0.88rem;
}}
pre {{
  border-radius: 10px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
}}

/* --- Dataframes/tables: subtle borders, tighter rows --- */
div[data-testid="stDataFrame"] {{
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
}}

/* --- Alerts: reduce visual noise --- */
div[data-testid="stAlert"] {{
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.08);
}}

/* --- Metric cards: align styling with rest --- */
div[data-testid="stMetric"] {{
  padding: 0.5rem 0.75rem;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.08);
}}

/* --- Horizontal rules: less harsh --- */
hr {{
  border: none;
  border-top: 1px solid rgba(255,255,255,0.08);
  margin: 1.25rem 0;
}}
</style>
        """,
        unsafe_allow_html=True,
    )
