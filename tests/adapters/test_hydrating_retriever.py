"""Tests for the HydratingRetriever wrapper.

Verifies that the wrapper transparently hydrates thin chunks (empty text)
from a ChunkStore while passing through fat chunks untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag.adapters.retrieval.hydrating_retriever import HydratingRetriever
from rag.domain.filters import Where
from rag.domain.models import Candidate, Chunk
from tests.conftest import make_candidate, make_chunk

# =============================================================================
# Fakes
# =============================================================================


class _FakeRetriever:
    """Retriever that returns preconfigured candidates."""

    def __init__(self, candidates: list[Candidate]) -> None:
        self._candidates = candidates

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        return self._candidates[:top_k]


class _FakeChunkStore:
    """ChunkStore backed by a dict for testing."""

    def __init__(self, chunks: dict[str, Chunk]) -> None:
        self._chunks = chunks
        self.get_chunks_calls: list[list[str]] = []

    def get_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, Chunk]:
        self.get_chunks_calls.append(list(chunk_ids))
        return {cid: self._chunks[cid] for cid in chunk_ids if cid in self._chunks}

    def store_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        for ch in chunks:
            self._chunks[ch.chunk_id] = ch

    def list_all_chunk_ids(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[str]:
        return sorted(self._chunks)


# =============================================================================
# Helpers
# =============================================================================


def _thin_chunk(chunk_id: str = "c1", doc_id: str = "d1") -> Chunk:
    """Create a chunk with empty text (thin payload stub)."""
    return make_chunk(chunk_id=chunk_id, doc_id=doc_id, text="")


def _fat_chunk(chunk_id: str = "c1", doc_id: str = "d1", text: str = "Hello world") -> Chunk:
    """Create a chunk with text (fat payload)."""
    return make_chunk(chunk_id=chunk_id, doc_id=doc_id, text=text)


# =============================================================================
# Tests
# =============================================================================


class TestPassthrough:
    """When all candidates have text, the ChunkStore is never called."""

    def test_fat_candidates_returned_unchanged(self) -> None:
        """Candidates with text pass through without hydration."""
        cands = [
            make_candidate(chunk=_fat_chunk("c1", text="alpha"), score=0.9),
            make_candidate(chunk=_fat_chunk("c2", text="beta"), score=0.8),
        ]
        store = _FakeChunkStore({})
        wrapper = HydratingRetriever(
            retriever=_FakeRetriever(cands),
            chunk_store=store,
        )

        result = wrapper.retrieve("q", top_k=2)

        assert len(result) == 2
        assert result[0].chunk.text == "alpha"
        assert result[1].chunk.text == "beta"
        assert store.get_chunks_calls == []

    def test_empty_candidates_list(self) -> None:
        """Empty result from inner retriever returns empty without error."""
        store = _FakeChunkStore({})
        wrapper = HydratingRetriever(
            retriever=_FakeRetriever([]),
            chunk_store=store,
        )

        result = wrapper.retrieve("q", top_k=5)

        assert result == []
        assert store.get_chunks_calls == []


class TestHydration:
    """When candidates have empty text, the wrapper hydrates from ChunkStore."""

    def test_thin_candidates_are_hydrated(self) -> None:
        """Candidates with empty text are replaced with full chunks."""
        thin = _thin_chunk("c1")
        full = _fat_chunk("c1", text="Full content from S3")
        cands = [make_candidate(chunk=thin, score=0.95)]
        store = _FakeChunkStore({"c1": full})
        wrapper = HydratingRetriever(
            retriever=_FakeRetriever(cands),
            chunk_store=store,
        )

        result = wrapper.retrieve("q", top_k=1)

        assert len(result) == 1
        assert result[0].chunk.text == "Full content from S3"
        assert result[0].score == 0.95
        assert store.get_chunks_calls == [["c1"]]

    def test_mixed_thin_and_fat(self) -> None:
        """Only thin candidates are hydrated; fat ones are untouched."""
        fat = _fat_chunk("c1", text="Already have text")
        thin = _thin_chunk("c2")
        full_c2 = _fat_chunk("c2", text="Hydrated from store")
        cands = [
            make_candidate(chunk=fat, score=0.9),
            make_candidate(chunk=thin, score=0.8),
        ]
        store = _FakeChunkStore({"c2": full_c2})
        wrapper = HydratingRetriever(
            retriever=_FakeRetriever(cands),
            chunk_store=store,
        )

        result = wrapper.retrieve("q", top_k=2)

        assert result[0].chunk.text == "Already have text"
        assert result[1].chunk.text == "Hydrated from store"
        # Only thin chunk requested from store
        assert store.get_chunks_calls == [["c2"]]

    def test_preserves_order_and_scores(self) -> None:
        """Hydration preserves candidate order and scores."""
        thin1 = _thin_chunk("c1")
        thin2 = _thin_chunk("c2")
        full1 = _fat_chunk("c1", text="First")
        full2 = _fat_chunk("c2", text="Second")
        cands = [
            make_candidate(chunk=thin1, score=0.95),
            make_candidate(chunk=thin2, score=0.70),
        ]
        store = _FakeChunkStore({"c1": full1, "c2": full2})
        wrapper = HydratingRetriever(
            retriever=_FakeRetriever(cands),
            chunk_store=store,
        )

        result = wrapper.retrieve("q", top_k=2)

        assert result[0].score == 0.95
        assert result[0].chunk.chunk_id == "c1"
        assert result[1].score == 0.70
        assert result[1].chunk.chunk_id == "c2"


class TestMissingChunks:
    """When a chunk is not found in the store, the stub is preserved."""

    def test_missing_chunk_keeps_stub(self) -> None:
        """If ChunkStore doesn't have the chunk, the empty-text stub remains."""
        thin = _thin_chunk("c-missing")
        cands = [make_candidate(chunk=thin, score=0.9)]
        store = _FakeChunkStore({})  # empty store
        wrapper = HydratingRetriever(
            retriever=_FakeRetriever(cands),
            chunk_store=store,
        )

        result = wrapper.retrieve("q", top_k=1)

        assert len(result) == 1
        assert result[0].chunk.text == ""
        assert result[0].chunk.chunk_id == "c-missing"

    def test_partial_hydration(self) -> None:
        """Some chunks hydrated, others missing — partial results returned."""
        thin1 = _thin_chunk("c1")
        thin2 = _thin_chunk("c2")
        full1 = _fat_chunk("c1", text="Found this one")
        cands = [
            make_candidate(chunk=thin1, score=0.9),
            make_candidate(chunk=thin2, score=0.8),
        ]
        store = _FakeChunkStore({"c1": full1})  # c2 missing
        wrapper = HydratingRetriever(
            retriever=_FakeRetriever(cands),
            chunk_store=store,
        )

        result = wrapper.retrieve("q", top_k=2)

        assert result[0].chunk.text == "Found this one"
        assert result[1].chunk.text == ""  # stub preserved
