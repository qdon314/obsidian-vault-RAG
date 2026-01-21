# Screenshots Directory

This directory contains screenshots for the Results Analyzer documentation.

## Required Screenshots

To complete the documentation, capture the following screenshots and place them in this directory:

### Application Layout
- `results-analyzer-layout.png` - Full application showing sidebar and main content
- `results-analyzer-sidebar.png` - Sidebar with view mode selection and controls

### Single Run Analysis View
- `single-run-selector.png` - Run selector dropdown showing available runs
- `single-run-config.png` - Expanded run configuration panel
- `single-run-metrics.png` - Metrics tab showing overall retrieval metrics table
- `single-run-metrics-by-type.png` - Metrics by query type expandable sections
- `single-run-charts-metrics.png` - Bar chart showing Recall, Precision, NDCG by K value
- `single-run-charts-by-type.png` - Bar chart showing Recall@10 by query type
- `single-run-raw-data.png` - Raw data tab with expandable JSON sections

### Query Explorer
- `query-explorer-filters.png` - Filter controls row (type, difficulty, pass/fail, search)
- `query-explorer-table.png` - Results table with status indicators
- `query-explorer-detail.png` - Query detail view showing retrieval breakdown
- `query-explorer-chunks.png` - Matched/missed/extra chunks display

### Trace Viewer
- `trace-viewer.png` - Trace viewer showing pipeline stages tabs
- `trace-viewer-retrieval.png` - Expanded retrieval stage showing candidates

### Comparison View
- `comparison-run-selection.png` - Two-column run selection for comparison
- `comparison-summary-metrics.png` - Summary metrics row with delta indicators
- `comparison-query-changes-summary.png` - Improved/regressed/unchanged counts
- `comparison-chart-recall.png` - Recall@K comparison bar chart
- `comparison-chart-global.png` - Global metrics (MRR, MAP) comparison chart
- `comparison-delta-table.png` - Full delta table showing all metric comparisons
- `comparison-improved-queries.png` - Improved queries list with recall changes
- `comparison-regressed-queries.png` - Regressed queries list

### Trending View
- `trending-run-selection.png` - Multi-select run picker
- `trending-metric-selector.png` - Metric and K value selectors
- `trending-single-metric.png` - Single metric trend line chart
- `trending-multi-metric.png` - Multi-metric trend chart with three lines
- `trending-summary-table.png` - Run summary table

## Screenshot Guidelines

1. **Resolution**: Use a consistent browser width (e.g., 1440px)
2. **Theme**: Use light theme for better readability in documentation
3. **Data**: Use representative sample data with meaningful values
4. **Crop**: Focus on the relevant UI element, but include enough context
5. **Annotations**: If needed, use subtle arrows or highlights to draw attention

## Capturing Screenshots

1. Launch the application: `streamlit run eval/app/results_analyzer.py`
2. Navigate to the relevant view
3. Use browser dev tools or a screenshot tool to capture
4. Save as PNG with the exact filename from the list above
