# Results Analyzer UX Analysis

## Current State Overview

The results analyzer is a Streamlit app for analyzing RAG evaluation runs. It has three main views:
- **Single Run Analysis**: View metrics, charts, query explorer, traces, and raw data for one run
- **Compare Two Runs**: Side-by-side comparison with delta highlighting
- **Multi-Run Trending**: Time series analysis across multiple runs

## User Goals

Based on the codebase analysis, users want to:

1. **Quickly assess run quality** - See if a run performed well at a glance
2. **Compare experiments** - Understand what changed between two runs
3. **Track progress over time** - See if metrics are improving across runs
4. **Debug failing queries** - Drill down into specific queries that didn't perform well
5. **Understand pipeline behavior** - Examine traces to see retrieval, reranking, generation

## Current Friction Points

### 1. Navigation & Wayfinding

**Issue**: The view switcher (single/comparison/trending) is a radio button in the sidebar. Once you're in a view, it's not always clear where you are or how to get back.

```python
# Current: Simple radio button
view = st.radio(
    "View Mode",
    options=["single", "comparison", "trending"],
    format_func=lambda v: {...}[v],
    key="view_mode",
)
```

**Impact**: Users may get lost when switching between views, especially since each view has its own sub-navigation (tabs within single run view).

### 2. Deeply Nested UI in Single Run View

**Issue**: The single run view has tabs (Metrics, Charts, Query Explorer, Traces, Raw Data) plus expandable sections within each tab. This creates deep nesting.

```python
# Current: Tabs within tabs within expanders
selected_tab = st.radio("View", tab_names, ...)  # Top level tabs
if selected_tab == "Metrics":
    render_metrics_table(...)  # Has expanders for by_type, by_difficulty
elif selected_tab == "Query Explorer":
    render_query_explorer(...)  # Has filters, table, detail view
```

**Impact**: Users have to click multiple times to reach the data they want. No way to link directly to a specific query's trace.

### 3. No Persistent State for Filters

**Issue**: Filters in Query Explorer reset when switching tabs or views.

```python
# Current: Filters are local to render function
selected_types = st.multiselect("Query Type", ..., key="explorer_query_types")
```

**Impact**: Users lose their place when exploring data. Can't bookmark/share a specific filtered view.

### 4. Comparison View - No Side-by-Side Query Detail

**Issue**: In comparison view, you can see delta tables and charts, but drilling into a specific query requires mental mapping between runs.

```python
# Current: Shows summary stats, then lists improved/regressed queries
render_summary_metrics(comparison)
render_query_changes_summary(comparison)
# Each query shows: Recall A → Recall B
```

**Impact**: Hard to understand *why* a query changed - need to see the actual retrieved chunks side by side.

### 5. Trending View - Limited Metric Selection

**Issue**: Can only view one metric at a time in the main chart, then a separate multi-metric chart below.

```python
# Current: Two separate chart areas
metric_choice = st.selectbox("Metric", [...])  # Single metric
render_trend_chart(trend, metric=metric_choice, k=k_choice)
# Then separate multi-metric
render_multi_metric_trend(trend, metrics=[...], k=10)
```

**Impact**: Can't customize which metrics appear in the multi-metric view.

### 6. Run Selection - No Search/Filter

**Issue**: Run selector is a simple dropdown that can get unwieldy with many runs.

```python
# Current: Basic selectbox
selected = st.selectbox(label, options=[None, *options.keys()], ...)
```

**Impact**: Hard to find specific runs when there are dozens of evaluation runs.

### 7. Trace Viewer - Buried in Tabs

**Issue**: Traces are only accessible through the "Traces" tab in single run view, or via Query Explorer drill-down.

**Impact**: Traces are a powerful debugging tool but they're not discoverable.

### 8. No Keyboard Shortcuts

**Issue**: Everything requires mouse interaction.

**Impact**: Power users can't navigate quickly.

### 9. Mobile/Responsive Issues

**Issue**: Multi-column layouts don't adapt well to smaller screens.

```python
# Current: Fixed column layouts
col1, col2, col3 = st.columns(3)  # Gets squashed on mobile
```

### 10. Loading States

**Issue**: Some operations show spinners, but there's no progressive loading for large runs.

```python
# Current: All-or-nothing loading
with st.spinner("Loading run data..."):
    loaded_run = repo.get_run(selected_id)
```

## Proposed UX Improvements

### High Priority

1. **Persistent Navigation Bar**
   - Move view switcher to top horizontal nav
   - Show breadcrumbs for nested views
   - Highlight current location

2. **Unified Filter Bar**
   - Global filters that persist across views (date range, query types)
   - URL-synced filters for shareable links
   - Save/load filter presets

3. **Improved Run Selector**
   - Searchable dropdown with typeahead
   - Recent runs section
   - Filter by date, model, metrics

4. **Side-by-Side Query Comparison**
   - When comparing runs, show query details side by side
   - Highlight chunk differences visually
   - Link to traces from comparison view

### Medium Priority

5. **Customizable Dashboard**
   - Let users pin metrics/charts to a dashboard
   - Drag-and-drop layout
   - Save custom views

6. **Keyboard Shortcuts**
   - `Cmd+K` for command palette
   - `←/→` for navigating queries
   - `1/2/3` for switching main views

7. **Progressive Loading**
   - Load summary first, details on demand
   - Virtual scrolling for large query tables
   - Background loading for traces

### Lower Priority

8. **Mobile Responsiveness**
   - Collapsible sidebar
   - Stacked layouts on small screens
   - Touch-friendly controls

9. **Empty States**
   - Better guidance when no runs exist
   - Suggested next steps
   - Example data option

10. **Export/Share**
    - One-click export of charts
    - Share links to specific views
    - PDF report generation

## Implementation Approach

Rather than a full architectural refactor, we can make incremental UX improvements:

### Phase 1: Navigation & State
- Add persistent navigation component
- Implement URL state sync for filters
- Create improved run selector with search

### Phase 2: Comparison & Query Explorer
- Redesign comparison view with side-by-side layout
- Add query diff visualization
- Link traces to query explorer

### Phase 3: Dashboard & Customization
- Add customizable dashboard view
- Implement keyboard shortcuts
- Add command palette

This approach delivers value faster and allows for user feedback before major architectural changes.
