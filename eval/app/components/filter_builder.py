"""
Interactive filter builder component for constructing retrieval filters.
"""

from __future__ import annotations

import json

import streamlit as st

from rag.domain.filter_serde import deserialize_filter

# Filter types available in the builder
FILTER_TYPES = ["None", "Eq", "In", "Contains", "Prefix", "Range", "And", "Or", "Not"]
SIMPLE_FILTER_TYPES = ["Eq", "In", "Contains", "Prefix", "Range"]

# Common field names for suggestions
COMMON_FIELDS = ["doc_id", "uri", "tags"]


def _build_simple_filter(filter_type: str, key_prefix: str) -> dict | None:
    """Build a simple (non-composite) filter from form inputs."""
    field = st.text_input(
        "Field",
        key=f"{key_prefix}_field",
        placeholder="e.g., doc_id, uri, tags",
        help=f"Common fields: {', '.join(COMMON_FIELDS)}",
    )

    if not field:
        return None

    if filter_type == "Eq":
        value = st.text_input(
            "Value",
            key=f"{key_prefix}_value",
            placeholder="Exact value to match",
        )
        if value:
            return {"type": "Eq", "field": field, "value": value}

    elif filter_type == "In":
        values_input = st.text_input(
            "Values (comma-separated)",
            key=f"{key_prefix}_values",
            placeholder="value1, value2, value3",
        )
        if values_input:
            values = [v.strip() for v in values_input.split(",") if v.strip()]
            if values:
                return {"type": "In", "field": field, "values": values}

    elif filter_type == "Contains":
        value = st.text_input(
            "Value",
            key=f"{key_prefix}_contains_value",
            placeholder="Value that field should contain",
        )
        if value:
            return {"type": "Contains", "field": field, "value": value}

    elif filter_type == "Prefix":
        prefix = st.text_input(
            "Prefix",
            key=f"{key_prefix}_prefix",
            placeholder="String prefix to match",
        )
        if prefix:
            return {"type": "Prefix", "field": field, "prefix": prefix}

    elif filter_type == "Range":
        st.caption("At least one bound required")
        col1, col2 = st.columns(2)
        with col1:
            gte = st.text_input("Greater than or equal (gte)", key=f"{key_prefix}_gte")
            gt = st.text_input("Greater than (gt)", key=f"{key_prefix}_gt")
        with col2:
            lte = st.text_input("Less than or equal (lte)", key=f"{key_prefix}_lte")
            lt = st.text_input("Less than (lt)", key=f"{key_prefix}_lt")

        range_filter: dict = {"type": "Range", "field": field}
        if gte:
            range_filter["gte"] = gte
        if gt:
            range_filter["gt"] = gt
        if lte:
            range_filter["lte"] = lte
        if lt:
            range_filter["lt"] = lt

        if len(range_filter) > 2:  # Has at least one bound beyond type and field
            return range_filter

    return None


def _build_sub_filter(key_prefix: str, depth: int = 0) -> dict | None:
    """Build a filter, potentially nested. Limits depth to prevent infinite recursion."""
    max_depth = 2

    # At max depth, only allow simple filters
    available_types = ["None", *SIMPLE_FILTER_TYPES] if depth >= max_depth else FILTER_TYPES

    filter_type = st.selectbox(
        "Filter type",
        options=available_types,
        key=f"{key_prefix}_type",
        help="Select the type of filter to create",
    )

    if filter_type == "None":
        return None

    if filter_type in SIMPLE_FILTER_TYPES:
        return _build_simple_filter(filter_type, key_prefix)

    elif filter_type == "And":
        return _build_composite_filter("And", key_prefix, depth)

    elif filter_type == "Or":
        return _build_composite_filter("Or", key_prefix, depth)

    elif filter_type == "Not":
        st.caption("Negated filter:")
        with st.container():
            inner = _build_sub_filter(f"{key_prefix}_not", depth + 1)
            if inner:
                return {"type": "Not", "clause": inner}

    return None


def _build_composite_filter(composite_type: str, key_prefix: str, depth: int) -> dict | None:
    """Build an And or Or composite filter with multiple clauses."""
    # Track number of clauses in session state
    count_key = f"{key_prefix}_{composite_type.lower()}_count"
    if count_key not in st.session_state:
        st.session_state[count_key] = 2

    clause_count = st.session_state[count_key]

    st.caption(f"{composite_type} filter clauses:")

    clauses = []
    for i in range(clause_count):
        with st.expander(f"Clause {i + 1}", expanded=True):
            clause = _build_sub_filter(f"{key_prefix}_{composite_type.lower()}_{i}", depth + 1)
            if clause:
                clauses.append(clause)

    # Add/remove clause buttons (outside form to allow dynamic updates)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("+ Add clause", key=f"{key_prefix}_add"):
            st.session_state[count_key] = clause_count + 1
            st.rerun()
    with col2:
        if clause_count > 2 and st.button("- Remove last", key=f"{key_prefix}_remove"):
            st.session_state[count_key] = clause_count - 1
            st.rerun()

    if len(clauses) >= 2:
        return {"type": composite_type, "clauses": clauses}
    elif len(clauses) == 1:
        st.warning(f"{composite_type} requires at least 2 clauses")

    return None


def _init_builder_from_filter(filter_dict: dict, key_prefix: str) -> None:
    """Initialize session state from an existing filter for editing."""
    if not filter_dict:
        return

    filter_type = filter_dict.get("type")
    if not filter_type:
        return

    # Set the filter type
    type_key = f"{key_prefix}_type"
    st.session_state[type_key] = filter_type

    # Initialize fields based on type
    if filter_type == "Eq":
        st.session_state[f"{key_prefix}_field"] = filter_dict.get("field", "")
        st.session_state[f"{key_prefix}_value"] = filter_dict.get("value", "")

    elif filter_type == "In":
        st.session_state[f"{key_prefix}_field"] = filter_dict.get("field", "")
        values = filter_dict.get("values", [])
        st.session_state[f"{key_prefix}_values"] = ", ".join(str(v) for v in values)

    elif filter_type == "Contains":
        st.session_state[f"{key_prefix}_field"] = filter_dict.get("field", "")
        st.session_state[f"{key_prefix}_contains_value"] = filter_dict.get("value", "")

    elif filter_type == "Prefix":
        st.session_state[f"{key_prefix}_field"] = filter_dict.get("field", "")
        st.session_state[f"{key_prefix}_prefix"] = filter_dict.get("prefix", "")

    elif filter_type == "Range":
        st.session_state[f"{key_prefix}_field"] = filter_dict.get("field", "")
        st.session_state[f"{key_prefix}_gte"] = filter_dict.get("gte", "")
        st.session_state[f"{key_prefix}_gt"] = filter_dict.get("gt", "")
        st.session_state[f"{key_prefix}_lte"] = filter_dict.get("lte", "")
        st.session_state[f"{key_prefix}_lt"] = filter_dict.get("lt", "")

    elif filter_type in ("And", "Or"):
        clauses = filter_dict.get("clauses", [])
        count_key = f"{key_prefix}_{filter_type.lower()}_count"
        st.session_state[count_key] = max(2, len(clauses))
        for i, clause in enumerate(clauses):
            _init_builder_from_filter(clause, f"{key_prefix}_{filter_type.lower()}_{i}")

    elif filter_type == "Not":
        inner = filter_dict.get("clause")
        if inner:
            _init_builder_from_filter(inner, f"{key_prefix}_not")


def render_filter_builder(key_prefix: str, existing_filter: dict | None = None) -> dict | None:
    """
    Render an interactive filter builder.

    Args:
        key_prefix: Unique prefix for Streamlit widget keys
        existing_filter: Optional existing filter to pre-populate the builder

    Returns:
        The filter dict if valid, None if no filter or invalid
    """
    st.markdown("**Retrieval Filter (optional)**")

    # Mode toggle: builder vs raw JSON
    mode_key = f"{key_prefix}_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "builder"

    # Initialize from existing filter on first render
    init_key = f"{key_prefix}_initialized"
    if existing_filter and init_key not in st.session_state:
        _init_builder_from_filter(existing_filter, f"{key_prefix}_builder")
        st.session_state[init_key] = True

    mode = st.radio(
        "Edit mode",
        options=["builder", "raw"],
        format_func=lambda x: "Filter Builder" if x == "builder" else "Raw JSON",
        key=mode_key,
        horizontal=True,
        help="Use the builder for guided filter creation, or raw JSON for advanced editing",
    )

    result_filter = None

    if mode == "builder":
        result_filter = _build_sub_filter(f"{key_prefix}_builder")

        # Show preview
        if result_filter:
            st.caption("Generated filter:")
            st.code(json.dumps(result_filter, indent=2), language="json")

            # Validate
            try:
                deserialize_filter(result_filter)
                st.success("Filter is valid", icon="✅")
            except (ValueError, KeyError, TypeError) as e:
                st.error(f"Invalid filter: {e}")
                result_filter = None
    else:
        # Raw JSON mode
        default_json = ""
        if existing_filter:
            default_json = json.dumps(existing_filter, indent=2)

        filter_json = st.text_area(
            "Filter JSON",
            value=default_json,
            height=120,
            key=f"{key_prefix}_raw_json",
            placeholder='{"type": "Eq", "field": "doc_id", "value": "my-doc"}',
            help=(
                "Enter filter as JSON. "
                "Types: Eq, In, Contains, Prefix, Range, And, Or, Not."
            ),
        )

        if filter_json.strip():
            try:
                result_filter = json.loads(filter_json.strip())
                deserialize_filter(result_filter)
                st.success("Filter is valid", icon="\u2713")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                result_filter = None
            except (ValueError, KeyError, TypeError) as e:
                st.error(f"Invalid filter structure: {e}")
                result_filter = None

    return result_filter
