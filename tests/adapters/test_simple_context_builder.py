"""Tests for SimpleContextBuilder adapter."""

from __future__ import annotations

from rag.adapters.context_building.simple_context_builder import SimpleContextBuilder
from rag.ports import ContextBuilder
from tests.conftest import make_candidate, make_chunk


class TestSimpleContextBuilderBasics:
    """Basic context building behavior."""

    def test_build_empty_candidates(self, simple_context_builder: ContextBuilder):
        """Building from empty candidates returns empty context."""
        pack = simple_context_builder.build("test query", [], token_budget=1000)

        assert len(pack.chunks) == 0
        assert len(pack.citations) == 0
        assert pack.query == "test query"

    def test_build_creates_context_pack(self, simple_context_builder: ContextBuilder):
        """Build returns a ContextPack with chunks and citations."""
        candidates = [
            make_candidate(
                score=0.9, chunk=make_chunk(text="First chunk text.")
            ),  # Needs to have different text to avoid dedupe
            make_candidate(chunk=make_chunk(chunk_id="c2"), score=0.8),
        ]

        pack = simple_context_builder.build("query", candidates, token_budget=10000)

        assert len(pack.chunks) == 2
        assert len(pack.citations) == 2
        assert pack.rendered_context != ""

    def test_build_includes_query_in_pack(self, simple_context_builder: ContextBuilder):
        """ContextPack includes the original query."""
        pack = simple_context_builder.build(
            "my search query",
            [make_candidate()],
            token_budget=10000,
        )

        assert pack.query == "my search query"

    def test_build_includes_token_budget_in_pack(self, simple_context_builder: ContextBuilder):
        """ContextPack includes the token budget."""
        pack = simple_context_builder.build(
            "query",
            [make_candidate()],
            token_budget=5000,
        )

        assert pack.token_budget == 5000


class TestSimpleContextBuilderTokenBudget:
    """Token budget enforcement."""

    def test_respects_token_budget(self):
        """Build stops adding chunks when token budget is reached."""
        builder = SimpleContextBuilder()
        # Create candidates with long text
        long_text = "A" * 1000
        candidates = [
            make_candidate(chunk=make_chunk(chunk_id=f"c{i}", text=long_text), score=0.9 - i * 0.1)
            for i in range(5)
        ]

        # Small budget should only fit 1-2 chunks
        pack = builder.build("query", candidates, token_budget=300)

        assert len(pack.chunks) < 5
        tokens_used = pack.metadata.get("tokens_used_est", 0)
        assert tokens_used <= 300

    def test_single_chunk_fits_budget(self):
        """Single small chunk should fit any reasonable budget."""
        builder = SimpleContextBuilder()
        candidates = [make_candidate(chunk=make_chunk(text="Short text"))]

        pack = builder.build("query", candidates, token_budget=1000)

        assert len(pack.chunks) == 1


class TestSimpleContextBuilderDeduplication:
    """Deduplication behavior."""

    def test_deduplicates_identical_chunks(self):
        """Deduplication removes chunks with identical text."""
        builder = SimpleContextBuilder(dedupe=True)
        same_text = "This is identical text content."
        candidates = [
            make_candidate(chunk=make_chunk(chunk_id="1", text=same_text), score=0.9),
            make_candidate(chunk=make_chunk(chunk_id="2", text=same_text), score=0.8),
            make_candidate(chunk=make_chunk(chunk_id="3", text="Different content"), score=0.7),
        ]

        pack = builder.build("query", candidates, token_budget=10000)

        # Should only include first occurrence of duplicate + the different one
        assert len(pack.chunks) == 2

    def test_no_deduplication_when_disabled(self):
        """With dedupe=False, duplicates are kept."""
        builder = SimpleContextBuilder(dedupe=False)
        same_text = "Duplicate text."
        candidates = [
            make_candidate(chunk=make_chunk(chunk_id="1", text=same_text), score=0.9),
            make_candidate(chunk=make_chunk(chunk_id="2", text=same_text), score=0.8),
        ]

        pack = builder.build("query", candidates, token_budget=10000)

        assert len(pack.chunks) == 2

    def test_dedupe_normalizes_whitespace(self):
        """Deduplication normalizes whitespace when comparing."""
        builder = SimpleContextBuilder(dedupe=True)
        candidates = [
            make_candidate(chunk=make_chunk(chunk_id="1", text="hello world"), score=0.9),
            make_candidate(chunk=make_chunk(chunk_id="2", text="hello  world"), score=0.8),
            make_candidate(chunk=make_chunk(chunk_id="3", text="HELLO WORLD"), score=0.7),
        ]

        pack = builder.build("query", candidates, token_budget=10000)

        # Should dedupe the normalized versions (all same after lowercase + whitespace normalize)
        assert len(pack.chunks) == 1


class TestSimpleContextBuilderScoring:
    """Score handling and ordering."""

    def test_filters_by_min_score(self):
        """min_score threshold filters out low-scoring candidates."""
        builder = SimpleContextBuilder(min_score=0.8)
        candidates = [
            make_candidate(chunk=make_chunk(text="1"), score=0.9),
            make_candidate(chunk=make_chunk(text="2"), score=0.75),  # Below threshold
            make_candidate(chunk=make_chunk(text="3"), score=0.85),
        ]

        pack = builder.build("query", candidates, token_budget=10000)

        assert len(pack.chunks) == 2
        chunk_ids = [c.chunk_id for c in pack.chunks]
        assert "2" not in chunk_ids

    def test_prefers_rerank_score_when_present(self):
        """Candidates are ordered by rerank_score when present."""
        builder = SimpleContextBuilder()
        candidates = [
            make_candidate(chunk=make_chunk(chunk_id="high-base"), score=0.9, rerank_score=0.5),
            make_candidate(chunk=make_chunk(chunk_id="high-rerank"), score=0.6, rerank_score=0.95),
        ]

        pack = builder.build("query", candidates, token_budget=10000)

        # Should prioritize by rerank_score
        assert pack.chunks[0].chunk_id == "high-rerank"

    def test_uses_base_score_without_rerank(self):
        """Without rerank_score, uses base score for ordering."""
        builder = SimpleContextBuilder()
        candidates = [
            make_candidate(chunk=make_chunk(chunk_id="low"), score=0.5),
            make_candidate(chunk=make_chunk(chunk_id="high"), score=0.9),
        ]

        pack = builder.build("query", candidates, token_budget=10000)

        assert pack.chunks[0].chunk_id == "high"


class TestSimpleContextBuilderMaxChunks:
    """Max chunks limiting."""

    def test_respects_max_chunks(self):
        """max_chunks limits number of chunks even with budget."""
        builder = SimpleContextBuilder(max_chunks=2)
        candidates = [
            make_candidate(chunk=make_chunk(chunk_id=f"c{i}", text=f"{i}"), score=0.9 - i * 0.1)
            for i in range(5)
        ]

        pack = builder.build("query", candidates, token_budget=100000)

        assert len(pack.chunks) == 2


class TestSimpleContextBuilderCitations:
    """Citation generation."""

    def test_creates_citations_for_each_chunk(self, simple_context_builder: ContextBuilder):
        """Each included chunk gets a citation."""
        candidates = [
            make_candidate(chunk=make_chunk(chunk_id="c1"), score=0.9),
            make_candidate(chunk=make_chunk(chunk_id="c2"), score=0.8),
        ]

        pack = simple_context_builder.build("query", candidates, token_budget=10000)

        assert len(pack.citations) == len(pack.chunks)

    def test_citations_include_chunk_metadata(self):
        """Citations include chunk metadata like URI."""
        builder = SimpleContextBuilder()
        chunk = make_chunk(
            chunk_id="test-chunk",
            doc_id="test-doc",
            metadata={"uri": "/docs/test.md", "title": "Test Doc"},
        )
        candidates = [make_candidate(chunk=chunk, score=0.9)]

        pack = builder.build("query", candidates, token_budget=10000)

        assert len(pack.citations) == 1
        citation = pack.citations[0]
        assert citation.chunk_id == "test-chunk"
        assert citation.doc_id == "test-doc"
        assert citation.uri == "/docs/test.md"

    def test_citations_include_quote(self):
        """Citations include a quote from the chunk text."""
        builder = SimpleContextBuilder()
        chunk = make_chunk(text="This is the chunk content for quoting.")
        candidates = [make_candidate(chunk=chunk, score=0.9)]

        pack = builder.build("query", candidates, token_budget=10000)

        assert pack.citations[0].quote is not None
        assert "chunk content" in pack.citations[0].quote


class TestSimpleContextBuilderRenderedContext:
    """Rendered context string."""

    def test_rendered_context_includes_chunk_text(self):
        """Rendered context contains the actual chunk text."""
        builder = SimpleContextBuilder()
        chunk_text = "This is important context."
        candidates = [make_candidate(chunk=make_chunk(text=chunk_text), score=0.9)]

        pack = builder.build("query", candidates, token_budget=10000)

        assert chunk_text in pack.rendered_context

    def test_rendered_context_has_instructions(self):
        """Rendered context includes the CONTEXT section header."""
        builder = SimpleContextBuilder()
        candidates = [make_candidate()]

        pack = builder.build("query", candidates, token_budget=10000)

        assert "CONTEXT" in pack.rendered_context
        assert "[1]" in pack.rendered_context

    def test_rendered_context_numbers_chunks(self):
        """Rendered context numbers the chunks."""
        builder = SimpleContextBuilder()
        candidates = [
            make_candidate(chunk=make_chunk(chunk_id="c1", text="First chunk text."), score=0.9),
            make_candidate(chunk=make_chunk(chunk_id="c2", text="Second chunk text."), score=0.8),
        ]

        pack = builder.build("query", candidates, token_budget=10000)

        assert "[1]" in pack.rendered_context
        assert "[2]" in pack.rendered_context
