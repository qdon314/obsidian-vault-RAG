from __future__ import annotations

import pytest

from rag.adapters.chunking.obsidian_structural import ObsidianStructuralChunker
from tests.conftest import make_document


def _make_chunker(
    *,
    target_chars: int = 80,
    hard_max_chars: int = 160,
    overlap_blocks: int = 0,
    include_heading_preamble: bool = True,
) -> ObsidianStructuralChunker:
    return ObsidianStructuralChunker(
        target_chars=target_chars,
        hard_max_chars=hard_max_chars,
        overlap_blocks=overlap_blocks,
        include_heading_preamble=include_heading_preamble,
    )


# =============================================================================
# Basic Chunking Behavior
# =============================================================================


class TestObsidianStructuralChunkerBasics:
    """Fundamental chunking behavior tests."""

    def test_empty_document_returns_no_chunks(self) -> None:
        """Empty or whitespace-only documents should produce no chunks."""
        chunker = _make_chunker()

        for text in ["", "   ", "\n\n\n", "\t\t"]:
            doc = make_document(text=text)
            chunks = chunker.chunk(doc)
            assert chunks == [], f"Expected no chunks for text: {text!r}"

    def test_single_paragraph_produces_one_chunk(self) -> None:
        """A simple paragraph without headings should produce one chunk."""
        doc = make_document(text="This is a simple paragraph of text.")
        chunker = _make_chunker(target_chars=200)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert "This is a simple paragraph of text." in chunks[0].text
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata.get("chunk_kind") == "para"

    def test_chunk_index_increments_correctly(self) -> None:
        """Chunk indices should start at 0 and increment sequentially."""
        doc = make_document(
            text="""# Section One
First content.

# Section Two
Second content.

# Section Three
Third content.
"""
        )
        chunker = _make_chunker(target_chars=50, hard_max_chars=100)

        chunks = chunker.chunk(doc)

        indices = [ch.chunk_index for ch in chunks]
        assert indices == list(range(len(chunks)))

    def test_doc_id_propagates_to_all_chunks(self) -> None:
        """All chunks should reference the source document ID."""
        doc = make_document(doc_id="my-unique-doc-id", text="# Heading\nContent here.")
        chunker = _make_chunker()

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.doc_id == "my-unique-doc-id"


class TestObsidianStructuralChunkerCodeBlocks:
    """Code fence handling and language inference."""

    def test_language_and_fences_preserved(self) -> None:
        """Single fenced block should keep both fences and detect language."""
        doc = make_document(
            text="""```python
print('hi')
```
""",
        )
        chunker = _make_chunker(target_chars=40, hard_max_chars=120)

        chunks = chunker.chunk(doc)
        code_chunks = [ch for ch in chunks if ch.metadata.get("chunk_kind") == "code"]

        assert len(code_chunks) == 1
        first_code = code_chunks[0]

        assert first_code.language == "python"
        _, _, rendered_body = first_code.text.partition("\n\n")
        assert rendered_body.startswith("```python")
        assert rendered_body.rstrip().endswith("```")

    def test_multiple_code_blocks_emit_separate_chunks(self) -> None:
        """Each fenced block should render as its own chunk with language preserved."""
        doc = make_document(
            text="""```python
print('alpha')
```

```javascript
console.log('beta')
```
""",
        )
        chunker = _make_chunker(target_chars=40, hard_max_chars=80)

        chunks = chunker.chunk(doc)
        code_chunks = [ch for ch in chunks if ch.metadata.get("chunk_kind") == "code"]

        assert [ch.language for ch in code_chunks] == ["python", "javascript"]
        for chunk in code_chunks:
            _, _, rendered_body = chunk.text.partition("\n\n")
            body = rendered_body.strip()
            assert body.count("```") == 2


class TestObsidianStructuralChunkerSections:
    """Heading and section metadata behavior."""

    def test_child_section_path_in_preamble_and_metadata(self) -> None:
        """Chunks should include section path metadata and rendered preamble."""
        doc = make_document(
            text="""# Parent
Intro text under parent.

## Child
Paragraph in child.

```python
print('child')
```
""",
        )
        chunker = _make_chunker(target_chars=200, hard_max_chars=400)

        chunks = chunker.chunk(doc)
        child_chunks = [ch for ch in chunks if ch.section_heading == "Child"]

        assert child_chunks, "Expected at least one chunk scoped to the child heading"
        child_chunk = child_chunks[0]

        assert child_chunk.section_path == "Parent > Child"
        assert child_chunk.metadata.get("chunk_kind") == "mixed"
        assert "Path: Parent > Child" in child_chunk.text
        assert "Paragraph in child." in child_chunk.text

    def test_deeply_nested_sections_build_correct_path(self) -> None:
        """Deeply nested headings should produce full section paths."""
        doc = make_document(
            text="""# Level 1
## Level 2
### Level 3
Content at level 3.
"""
        )
        chunker = _make_chunker(target_chars=200)

        chunks = chunker.chunk(doc)
        level3_chunks = [ch for ch in chunks if ch.section_heading == "Level 3"]

        assert len(level3_chunks) == 1
        assert level3_chunks[0].section_path == "Level 1 > Level 2 > Level 3"

    def test_sibling_sections_reset_path_correctly(self) -> None:
        """Sibling headings at the same level should not inherit from each other."""
        doc = make_document(
            text="""# Parent
## Child A
Content A.

## Child B
Content B.
"""
        )
        chunker = _make_chunker(target_chars=200)

        chunks = chunker.chunk(doc)
        child_a = [ch for ch in chunks if ch.section_heading == "Child A"]
        child_b = [ch for ch in chunks if ch.section_heading == "Child B"]

        assert child_a[0].section_path == "Parent > Child A"
        assert child_b[0].section_path == "Parent > Child B"

    def test_content_before_first_heading_has_no_section_path(self) -> None:
        """Content before any heading should have no section path."""
        doc = make_document(
            text="""Some preamble content before headings.

# First Heading
Content under heading.
"""
        )
        chunker = _make_chunker(target_chars=200)

        chunks = chunker.chunk(doc)
        preamble_chunks = [ch for ch in chunks if ch.section_heading is None]

        assert len(preamble_chunks) >= 1
        assert preamble_chunks[0].section_path is None


# =============================================================================
# Block Type Detection
# =============================================================================


class TestObsidianStructuralChunkerBlockTypes:
    """Block type detection for lists, tables, callouts."""

    def test_unordered_list_detected_as_list_kind(self) -> None:
        """Unordered lists should be detected as 'list' chunk kind."""
        doc = make_document(
            text="""# Lists
- Item one
- Item two
- Item three
"""
        )
        chunker = _make_chunker(target_chars=200)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.get("chunk_kind") == "list"

    def test_ordered_list_detected_as_list_kind(self) -> None:
        """Ordered lists should be detected as 'list' chunk kind."""
        doc = make_document(
            text="""# Numbered
1. First
2. Second
3. Third
"""
        )
        chunker = _make_chunker(target_chars=200)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.get("chunk_kind") == "list"

    def test_table_detected_as_table_kind(self) -> None:
        """Markdown tables should be detected as 'table' chunk kind."""
        doc = make_document(
            text="""# Data
| Col A | Col B |
|-------|-------|
| val1  | val2  |
| val3  | val4  |
"""
        )
        chunker = _make_chunker(target_chars=200)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.get("chunk_kind") == "table"

    def test_blockquote_detected_as_callout_kind(self) -> None:
        """Blockquotes should be detected as 'callout' chunk kind."""
        doc = make_document(
            text="""# Quote
> This is a blockquote
> spanning multiple lines.
"""
        )
        chunker = _make_chunker(target_chars=200)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.get("chunk_kind") == "callout"

    def test_obsidian_callout_detected_as_callout_kind(self) -> None:
        """Obsidian callouts (> [!note]) should be detected as 'callout' chunk kind."""
        doc = make_document(
            text="""# Notes
> [!note] Important
> This is an Obsidian callout.
"""
        )
        chunker = _make_chunker(target_chars=200)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.get("chunk_kind") == "callout"

    def test_mixed_block_types_produce_mixed_kind(self) -> None:
        """Multiple block types in one chunk should produce 'mixed' kind."""
        doc = make_document(
            text="""# Mixed Content
Some paragraph text.

- A list item
- Another item
"""
        )
        chunker = _make_chunker(target_chars=500)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.get("chunk_kind") == "mixed"


# =============================================================================
# Chunk Size Boundaries
# =============================================================================


class TestObsidianStructuralChunkerSizeBoundaries:
    """Tests for target_chars and hard_max_chars behavior."""

    def test_content_under_target_stays_in_one_chunk(self) -> None:
        """Content smaller than target_chars should stay in a single chunk."""
        doc = make_document(text="# Short\nBrief content.")
        chunker = _make_chunker(target_chars=500, hard_max_chars=1000)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1

    def test_content_exceeding_target_splits_at_block_boundary(self) -> None:
        """Content exceeding target_chars should split at block boundaries."""
        doc = make_document(
            text="""# Section
First paragraph with some text.

Second paragraph with more text.

Third paragraph with even more.
"""
        )
        # Small target to force splitting
        chunker = _make_chunker(target_chars=50, hard_max_chars=200)

        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        # Each chunk should be a complete block, not mid-sentence
        for chunk in chunks:
            body = chunk.text.split("\n\n", 1)[-1]  # skip preamble
            # Should not end mid-word (rough check)
            assert body.strip()

    def test_hard_max_forces_flush_even_mid_section(self) -> None:
        """Hard max should force a flush even if target hasn't been reached."""
        # Create blocks that individually fit under hard_max but together exceed it
        doc = make_document(
            text="""# Section
Block one with text.

Block two with text.

Block three with text.

Block four with text.
"""
        )
        chunker = _make_chunker(target_chars=500, hard_max_chars=80)

        chunks = chunker.chunk(doc)

        # Should have split due to hard_max, not target
        assert len(chunks) > 1


# =============================================================================
# Overlap Behavior
# =============================================================================


class TestObsidianStructuralChunkerOverlap:
    """Tests for overlap_blocks parameter."""

    def test_no_overlap_when_zero(self) -> None:
        """With overlap_blocks=0, chunks should not share content."""
        doc = make_document(
            text="""# Section
Block A content.

Block B content.

Block C content.
"""
        )
        chunker = _make_chunker(target_chars=40, hard_max_chars=80, overlap_blocks=0)

        chunks = chunker.chunk(doc)

        # Check that consecutive chunks don't share block content
        for i in range(len(chunks) - 1):
            body_i = chunks[i].text.split("\n\n", 1)[-1].strip()
            body_next = chunks[i + 1].text.split("\n\n", 1)[-1].strip()
            # The last line of chunk i should not appear at start of chunk i+1
            last_line_i = body_i.split("\n")[-1]
            if last_line_i:
                assert not body_next.startswith(last_line_i)

    def test_overlap_includes_previous_block(self) -> None:
        """With overlap_blocks=1, each chunk should start with the last block of the previous."""
        doc = make_document(
            text="""# Section
Alpha block.

Beta block.

Gamma block.
"""
        )
        chunker = _make_chunker(target_chars=30, hard_max_chars=60, overlap_blocks=1)

        chunks = chunker.chunk(doc)

        # With overlap, we should see some content repeated
        if len(chunks) >= 2:
            # The content "Beta block" should appear in consecutive chunks
            texts = [ch.text for ch in chunks]
            found_overlap = False
            for i in range(len(texts) - 1):
                for word in ["Alpha", "Beta", "Gamma"]:
                    if word in texts[i] and word in texts[i + 1]:
                        found_overlap = True
                        break
            assert found_overlap, "Expected overlapping content between chunks"


# =============================================================================
# Oversize Paragraph Splitting
# =============================================================================


class TestObsidianStructuralChunkerOversizeParagraph:
    """Tests for oversize paragraph handling."""

    def test_oversize_paragraph_splits_into_multiple_chunks(self) -> None:
        """Paragraphs exceeding hard_max_chars should be split."""
        long_para = "Word " * 100  # ~500 chars
        doc = make_document(text=f"# Section\n{long_para}")
        chunker = _make_chunker(target_chars=100, hard_max_chars=150)

        chunks = chunker.chunk(doc)

        assert len(chunks) > 1, "Oversize paragraph should split into multiple chunks"

    def test_oversize_paragraph_marked_with_split_reason(self) -> None:
        """Split paragraphs should have metadata indicating the split reason."""
        long_para = "X" * 300
        doc = make_document(text=f"# Section\n{long_para}")
        chunker = _make_chunker(target_chars=80, hard_max_chars=100)

        chunks = chunker.chunk(doc)

        split_chunks = [
            ch for ch in chunks if ch.metadata.get("split_reason") == "oversize_paragraph"
        ]
        assert len(split_chunks) > 0, "Expected chunks with split_reason metadata"

    def test_code_blocks_never_split_mid_block(self) -> None:
        """Code blocks should never be split, even if they exceed hard_max."""
        long_code = "x = 1\n" * 50  # Long code block
        doc = make_document(
            text=f"""# Code
```python
{long_code}
```
"""
        )
        chunker = _make_chunker(target_chars=50, hard_max_chars=100)

        chunks = chunker.chunk(doc)
        code_chunks = [ch for ch in chunks if ch.metadata.get("chunk_kind") == "code"]

        # Code block should remain intact (not split)
        for chunk in code_chunks:
            body = chunk.text.split("\n\n", 1)[-1]
            fence_count = body.count("```")
            assert fence_count == 2, "Code block should have opening and closing fences"


# =============================================================================
# Metadata Handling
# =============================================================================


class TestObsidianStructuralChunkerMetadata:
    """Tests for metadata propagation and merging."""

    def test_document_metadata_propagates_to_chunks(self) -> None:
        """Document metadata should be included in chunk metadata."""
        doc = make_document(
            text="# Test\nContent.",
            metadata={"author": "test", "category": "notes"},
        )
        chunker = _make_chunker()

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.metadata.get("author") == "test"
            assert chunk.metadata.get("category") == "notes"

    def test_caller_metadata_overrides_document_metadata(self) -> None:
        """Metadata passed to chunk() should override document metadata."""
        doc = make_document(
            text="# Test\nContent.",
            metadata={"source": "original", "extra": "keep"},
        )
        chunker = _make_chunker()

        chunks = chunker.chunk(doc, metadata={"source": "override"})

        for chunk in chunks:
            assert chunk.metadata.get("source") == "override"
            assert chunk.metadata.get("extra") == "keep"

    def test_chunk_strategy_in_metadata(self) -> None:
        """All chunks should have chunk_strategy in metadata."""
        doc = make_document(text="# Test\nContent.")
        chunker = _make_chunker()

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.metadata.get("chunk_strategy") == "obsidian_structural_v1"


# =============================================================================
# Stable Chunk IDs
# =============================================================================


class TestObsidianStructuralChunkerStableIds:
    """Tests for chunk ID stability and uniqueness."""

    def test_same_input_produces_same_chunk_ids(self) -> None:
        """Identical documents should produce identical chunk IDs."""
        text = "# Heading\nSome content here."
        doc1 = make_document(doc_id="doc-1", text=text)
        doc2 = make_document(doc_id="doc-1", text=text)
        chunker = _make_chunker()

        chunks1 = chunker.chunk(doc1)
        chunks2 = chunker.chunk(doc2)

        ids1 = [ch.chunk_id for ch in chunks1]
        ids2 = [ch.chunk_id for ch in chunks2]
        assert ids1 == ids2

    def test_different_doc_ids_produce_different_chunk_ids(self) -> None:
        """Same content with different doc_id should produce different chunk IDs."""
        text = "# Heading\nContent."
        doc1 = make_document(doc_id="doc-A", text=text)
        doc2 = make_document(doc_id="doc-B", text=text)
        chunker = _make_chunker()

        chunks1 = chunker.chunk(doc1)
        chunks2 = chunker.chunk(doc2)

        ids1 = {ch.chunk_id for ch in chunks1}
        ids2 = {ch.chunk_id for ch in chunks2}
        assert ids1.isdisjoint(ids2), "Different doc_ids should produce different chunk_ids"

    def test_all_chunk_ids_unique_within_document(self) -> None:
        """All chunks from a single document should have unique IDs."""
        doc = make_document(
            text="""# One
Content one.

# Two
Content two.

# Three
Content three.
"""
        )
        chunker = _make_chunker(target_chars=30, hard_max_chars=60)

        chunks = chunker.chunk(doc)

        ids = [ch.chunk_id for ch in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"


# =============================================================================
# Preamble Behavior
# =============================================================================


class TestObsidianStructuralChunkerPreamble:
    """Tests for heading preamble generation."""

    def test_preamble_includes_title_from_metadata(self) -> None:
        """Preamble should use title from metadata if available."""
        doc = make_document(
            text="# Section\nContent.",
            metadata={"title": "My Note Title"},
        )
        chunker = _make_chunker(include_heading_preamble=True)

        chunks = chunker.chunk(doc)

        assert "Title: My Note Title" in chunks[0].text

    def test_preamble_falls_back_to_uri(self) -> None:
        """Preamble should use URI as title fallback."""
        doc = make_document(
            text="# Section\nContent.",
            uri="/vault/my-note.md",
            metadata={},
        )
        chunker = _make_chunker(include_heading_preamble=True)

        chunks = chunker.chunk(doc)

        assert "Title: /vault/my-note.md" in chunks[0].text

    def test_no_preamble_when_disabled(self) -> None:
        """Preamble should not be included when include_heading_preamble=False."""
        doc = make_document(
            text="# Section\nContent here.",
            metadata={"title": "Should Not Appear"},
        )
        chunker = _make_chunker(include_heading_preamble=False)

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert "Title:" not in chunk.text
            assert "Path:" not in chunk.text

    def test_preamble_includes_section_path(self) -> None:
        """Preamble should include the full section path."""
        doc = make_document(
            text="""# Parent
## Child
Content under child.
""",
            metadata={"title": "Test Doc"},
        )
        chunker = _make_chunker(include_heading_preamble=True)

        chunks = chunker.chunk(doc)
        child_chunks = [ch for ch in chunks if ch.section_heading == "Child"]

        assert len(child_chunks) >= 1
        assert "Path: Parent > Child" in child_chunks[0].text


# =============================================================================
# Character Offset Tracking
# =============================================================================


class TestObsidianStructuralChunkerOffsets:
    """Tests for start_char and end_char accuracy."""

    def test_start_and_end_char_within_document_bounds(self) -> None:
        """Chunk offsets should be within document text bounds."""
        text = "# Heading\n\nParagraph one.\n\nParagraph two."
        doc = make_document(text=text)
        chunker = _make_chunker(target_chars=30, hard_max_chars=60)

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(text)
            assert chunk.start_char < chunk.end_char

    def test_offsets_do_not_overlap_incorrectly(self) -> None:
        """Without overlap, chunk end should not exceed next chunk's start."""
        doc = make_document(
            text="""# Section
Block one.

Block two.

Block three.
"""
        )
        chunker = _make_chunker(target_chars=30, hard_max_chars=50, overlap_blocks=0)

        chunks = chunker.chunk(doc)

        for i in range(len(chunks) - 1):
            # End of current should not exceed start of next (no overlap case)
            # Note: sections may have gaps, so we just verify ordering
            assert chunks[i].start_char < chunks[i + 1].start_char


# =============================================================================
# Edge Cases
# =============================================================================


class TestObsidianStructuralChunkerEdgeCases:
    """Edge case and regression tests."""

    def test_heading_only_document(self) -> None:
        """Document with only headings and no content should produce no chunks."""
        doc = make_document(
            text="""# Heading One
## Heading Two
### Heading Three
"""
        )
        chunker = _make_chunker()

        chunks = chunker.chunk(doc)

        # Empty sections should be dropped
        assert chunks == []

    def test_code_block_without_language(self) -> None:
        """Code blocks without language specifier should have language=None."""
        doc = make_document(
            text="""```
plain code
```
"""
        )
        chunker = _make_chunker()

        chunks = chunker.chunk(doc)
        code_chunks = [ch for ch in chunks if ch.metadata.get("chunk_kind") == "code"]

        assert len(code_chunks) == 1
        assert code_chunks[0].language is None

    def test_nested_list_stays_together(self) -> None:
        """Nested/indented list items should stay with their parent."""
        doc = make_document(
            text="""# List
- Parent item
  - Nested item
  - Another nested
- Second parent
"""
        )
        chunker = _make_chunker(target_chars=500)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert "Nested item" in chunks[0].text
        assert "Second parent" in chunks[0].text

    @pytest.mark.parametrize(
        "list_marker",
        ["-", "*", "+", "1.", "2)", "10."],
    )
    def test_various_list_markers_detected(self, list_marker: str) -> None:
        """Different list markers should all be detected as lists."""
        doc = make_document(text=f"# List\n{list_marker} Item one\n{list_marker} Item two\n")
        chunker = _make_chunker(target_chars=500)

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.get("chunk_kind") == "list"
