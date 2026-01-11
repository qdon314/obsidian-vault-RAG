"""
Session state management for the query curation UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from rag.adapters.chunk_loading import JsonlChunkLoader
from rag.adapters.eval_persistence import JsonlEvalStore
from rag.adapters.query_suggestion import OpenAIQuerySuggester
from rag.domain.models import Chunk
from rag.eval.schema import QuerySuggestion
from rag.settings import load_settings


@dataclass
class CurationState:
    """State for the query curation wizard."""

    # Configuration
    chunks_path: Path = field(
        default_factory=lambda: Path("artifacts/indexes/obsidian_index/chunks.jsonl")
    )
    output_path: Path = field(
        default_factory=lambda: Path("experiments/curated_queries.jsonl")
    )

    # Loaded data
    chunks: list[Chunk] = field(default_factory=list)
    chunks_by_doc: dict[str, list[Chunk]] = field(default_factory=dict)

    # Selection state
    selected_doc_id: str | None = None
    selected_chunk: Chunk | None = None  # Currently browsing chunk
    selected_chunks: list[Chunk] = field(default_factory=list)  # All chunks for this query

    # Generation state
    suggestions: list[QuerySuggestion] = field(default_factory=list)
    selected_suggestion_idx: int | None = None

    # Query being edited
    query_text: str = ""
    expected_answer: str = ""

    # Stats
    query_counter: int = 0


def init_state() -> None:
    """Initialize session state if not exists."""
    if "curation" not in st.session_state:
        st.session_state.curation = CurationState()

    if "services_initialized" not in st.session_state:
        st.session_state.services_initialized = False


def init_services() -> None:
    """Initialize services (chunk loader, eval store, query suggester)."""
    if st.session_state.services_initialized:
        return

    settings = load_settings()

    st.session_state.chunk_loader = JsonlChunkLoader()
    st.session_state.eval_store = JsonlEvalStore()

    api_key = settings.secrets.openai_api_key
    if api_key:
        st.session_state.query_suggester = OpenAIQuerySuggester(api_key=api_key)
    else:
        st.session_state.query_suggester = None

    st.session_state.services_initialized = True


def get_state() -> CurationState:
    """Get the current curation state."""
    init_state()
    return st.session_state.curation


def get_chunk_loader() -> JsonlChunkLoader:
    """Get the chunk loader service."""
    init_services()
    return st.session_state.chunk_loader


def get_eval_store() -> JsonlEvalStore:
    """Get the eval store service."""
    init_services()
    return st.session_state.eval_store


def get_query_suggester() -> OpenAIQuerySuggester | None:
    """Get the query suggester service (may be None if no API key)."""
    init_services()
    return st.session_state.query_suggester


def load_chunks() -> None:
    """Load chunks from the configured path."""
    state = get_state()
    loader = get_chunk_loader()

    state.chunks = loader.load_all(state.chunks_path)
    state.chunks_by_doc = loader.get_chunks_by_doc(state.chunks_path)


def reset_selection() -> None:
    """Reset selection state for next query."""
    state = get_state()
    state.selected_chunk = None
    state.selected_chunks = []
    state.suggestions = []
    state.selected_suggestion_idx = None
    state.query_text = ""
    state.expected_answer = ""


def add_chunk_to_selection(chunk: Chunk) -> None:
    """Add a chunk to the current selection (if not already added)."""
    state = get_state()
    if chunk.chunk_id not in {c.chunk_id for c in state.selected_chunks}:
        state.selected_chunks.append(chunk)


def remove_chunk_from_selection(chunk_id: str) -> None:
    """Remove a chunk from the current selection."""
    state = get_state()
    state.selected_chunks = [c for c in state.selected_chunks if c.chunk_id != chunk_id]


def increment_counter() -> None:
    """Increment the query counter."""
    state = get_state()
    state.query_counter += 1
