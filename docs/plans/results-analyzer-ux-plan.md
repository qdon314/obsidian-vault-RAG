# Results Analyzer UX Implementation Plan

## Overview

This plan focuses on incremental UX improvements to the results analyzer before any major architectural refactoring. The goal is to deliver immediate value while establishing patterns that can be carried forward.

## Phase 1: Navigation & State Management

### 1.1 Top Navigation Bar

**Goal**: Make view switching more discoverable and provide breadcrumbs.

**Changes**:
- Replace sidebar radio buttons with horizontal nav bar at top of page
- Add visual indicator for current view
- Include app logo/title on left, view switcher in center, help/actions on right

**Files**:
- `eval/app/results/ui/navigation.py` (new)
- `eval/app/results_analyzer.py` - replace sidebar view selector

**Implementation**:
```python
# eval/app/results/ui/navigation.py
def render_top_nav(current_view: str) -> str:
    """Render horizontal navigation bar."""
    cols = st.columns([1, 3, 1])
    
    with cols[0]:
        st.markdown("**📊 Results Analyzer**")
    
    with cols[1]:
        views = ["single", "comparison", "trending"]
        labels = ["Single Run", "Compare Runs", "Trends"]
        
        # Custom styled buttons or segmented control
        selected = st.segmented_control(
            "view",
            options=views,
            format_func=lambda x: labels[views.index(x)],
            default=current_view,
        )
    
    with cols[2]:
        if st.button("? Help"):
            show_help_dialog()
    
    return selected or current_view
```

### 1.2 URL State Sync

**Goal**: Make filters and selections shareable via URL.

**Changes**:
- Sync run selection, filters, and view state to URL query parameters
- Parse URL on load to restore state
- Update URL when state changes

**Files**:
- `eval/app/results/ui/state_sync.py` (new)
- `eval/app/results_analyzer.py` - integrate at top of main()

**Implementation**:
```python
# eval/app/results/ui/state_sync.py
def sync_state_to_url() -> None:
    """Sync current session state to URL query params."""
    params = {}
    
    if "view_mode" in st.session_state:
        params["view"] = st.session_state.view_mode
    
    if "single_run_selector" in st.session_state:
        params["run"] = st.session_state.single_run_selector
    
    # Add filter state
    for key in st.session_state:
        if key.startswith("filter_"):
            params[key] = st.session_state[key]
    
    st.query_params.update(params)

def restore_state_from_url() -> None:
    """Restore session state from URL query params."""
    params = st.query_params
    
    if "view" in params:
        st.session_state.view_mode = params["view"]
    
    if "run" in params:
        st.session_state.single_run_selector = params["run"]
```

### 1.3 Improved Run Selector

**Goal**: Make it easier to find runs when there are many.

**Changes**:
- Add search/filter to run dropdown
- Show recent runs at top
- Display key metrics inline
- Group by date

**Files**:
- `eval/app/results/ui/run_selector.py` - enhance existing

**Implementation**:
```python
# eval/app/results/ui/run_selector.py
def render_enhanced_run_selector(
    runs: list[RunSummary],
    key: str,
    multi: bool = False,
) -> str | list[str] | None:
    """Enhanced run selector with search and filtering."""
    
    # Filter/search UI
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("Search runs", key=f"{key}_search")
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (newest)", "Date (oldest)", "Recall@10 (high)", "Recall@10 (low)"],
            key=f"{key}_sort",
        )
    
    # Apply filters
    filtered_runs = _filter_runs(runs, search, sort_by)
    
    # Group by date
    grouped = _group_runs_by_date(filtered_runs)
    
    # Build options with rich formatting
    return st.selectbox(
        "Select run",
        options=[r.run_id for r in filtered_runs],
        format_func=lambda rid: _format_run_option(next(r for r in runs if r.run_id == rid)),
        key=key,
    )
```

## Phase 2: Single Run View Improvements

### 2.1 Persistent Filter Bar

**Goal**: Filters that persist when switching between tabs.

**Changes**:
- Extract filter UI into reusable component
- Store filter state at view level, not tab level
- Apply filters across all tabs (metrics, charts, query explorer)

**Files**:
- `eval/app/results/ui/filter_bar.py` (new)
- `eval/app/results_analyzer.py` - render_single_run_view()

### 2.2 Query Explorer Enhancements

**Goal**: Better query exploration with saved views.

**Changes**:
- Add "Save View" button to save current filters
- Show saved views as quick-select chips
- Add query ID search
- Improve table with sorting and column selection

**Files**:
- `eval/app/results/ui/query_explorer.py` - enhance existing

### 2.3 Inline Trace Access

**Goal**: Make traces more discoverable.

**Changes**:
- Add "View Trace" button next to each query in explorer
- Show trace preview inline instead of separate tab
- Link from query detail to full trace view

**Files**:
- `eval/app/results/ui/query_explorer.py`
- `eval/app/results/ui/trace_viewer.py` - add compact mode

## Phase 3: Comparison View Improvements

### 3.1 Side-by-Side Query Detail

**Goal**: Better understand why queries changed.

**Changes**:
- When clicking a changed query, show both runs side by side
- Highlight chunk differences (gained/lost/moved)
- Show answer comparison if available

**Files**:
- `eval/app/results/ui/comparison_detail.py` (new)
- `eval/app/results_analyzer.py` - render_comparison_view()

**Implementation**:
```python
# eval/app/results/ui/comparison_detail.py
def render_query_comparison_detail(
    result_a: EvalResult,
    result_b: EvalResult,
    trace_a: QueryTrace | None,
    trace_b: QueryTrace | None,
) -> None:
    """Render side-by-side comparison of a single query."""
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"**Run A: {result_a.run_name}**")
        _render_query_summary(result_a)
        _render_retrieval_chunks(result_a, highlight_diff=True)
    
    with col_b:
        st.markdown(f"**Run B: {result_b.run_name}**")
        _render_query_summary(result_b)
        _render_retrieval_chunks(result_b, highlight_diff=True)
    
    # Show diff summary
    diff = compute_retrieval_diff(result_a, result_b)
    _render_diff_summary(diff)
```

### 3.2 Improved Delta Visualization

**Goal**: Make metric changes more visually apparent.

**Changes**:
- Color-code delta values (green for improvement, red for regression)
- Add sparkline showing trend if more than 2 runs selected
- Show statistical significance indicators

**Files**:
- `eval/app/results/ui/delta_table.py` - enhance

## Phase 4: Trending View Improvements

### 4.1 Customizable Multi-Metric Chart

**Goal**: Let users choose which metrics to compare.

**Changes**:
- Add metric multi-select
- Allow custom colors per metric
- Add annotations for significant events

**Files**:
- `eval/app/results/ui/trend_chart.py` - enhance

### 4.2 Run Annotation

**Goal**: Add context to trend charts.

**Changes**:
- Allow adding notes/annotations to specific runs
- Show annotations as markers on charts
- Display annotation list below chart

**Files**:
- `eval/app/results/ui/trend_chart.py`
- `eval/app/results/domain/models.py` - add annotation model

## Phase 5: Keyboard Shortcuts & Command Palette

### 5.1 Command Palette

**Goal**: Quick access to all actions.

**Changes**:
- Add Cmd+K shortcut to open command palette
- Search across runs, queries, views
- Quick actions (export, refresh, etc.)

**Files**:
- `eval/app/results/ui/command_palette.py` (new)
- `eval/app/results/ui/keyboard_shortcuts.py` (new)

**Implementation**:
```python
# eval/app/results/ui/command_palette.py
def render_command_palette() -> None:
    """Render command palette dialog."""
    
    with st.dialog("Command Palette"):
        query = st.text_input("Type a command or search...", key="cmd_palette_input")
        
        if query:
            commands = _search_commands(query)
            for cmd in commands:
                if st.button(cmd.label, key=f"cmd_{cmd.id}"):
                    cmd.execute()
                    st.rerun()
```

### 5.2 Keyboard Shortcuts

**Goal**: Faster navigation for power users.

**Changes**:
- `1/2/3` - Switch views
- `←/→` - Navigate runs (in single view)
- `f` - Focus filter bar
- `r` - Refresh runs
- `?` - Show keyboard help

**Files**:
- `eval/app/results/ui/keyboard_shortcuts.py`
- `eval/app/results_analyzer.py` - integrate at top level

## Phase 6: Empty States & Onboarding

### 6.1 Better Empty States

**Goal**: Help users when there's no data.

**Changes**:
- Show illustration + clear message when no runs exist
- Provide "Run Evaluation" CTA button
- Link to documentation

**Files**:
- `eval/app/results/ui/empty_states.py` (new)
- `eval/app/results_analyzer.py` - use in main views

### 6.2 First-Run Experience

**Goal**: Guide new users.

**Changes**:
- Show welcome modal on first visit
- Highlight key features
- Offer sample data option

**Files**:
- `eval/app/results/ui/onboarding.py` (new)

## Implementation Order

### Week 1: Navigation & State
1. Top navigation bar
2. URL state sync
3. Improved run selector

### Week 2: Single Run Improvements
4. Persistent filter bar
5. Query explorer enhancements
6. Inline trace access

### Week 3: Comparison & Trends
7. Side-by-side query detail
8. Customizable multi-metric chart

### Week 4: Polish
9. Command palette
10. Keyboard shortcuts
11. Empty states

## Testing Strategy

- Manual testing of each feature
- Test with real evaluation data
- Verify URL state persistence across browser refresh
- Test keyboard shortcuts on different OS/browsers

## Success Metrics

- Reduced time to find specific runs (measure via user feedback)
- Increased use of comparison view (track via analytics if available)
- Fewer support questions about navigation
- Positive user feedback on new features
