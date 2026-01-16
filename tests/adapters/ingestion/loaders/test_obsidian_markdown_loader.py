"""Tests for ObsidianMarkdownLoader and related utilities."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import (
    CODE_BLOCK_NOTE,
    MOC_NOTE_CONTENT,
    NOTE_WITH_FRONTMATTER,
    NOTE_WITH_INLINE_TAGS,
    NOTE_WITH_WIKILINKS,
    SIMPLE_NOTE_CONTENT,
    write_note,
)

from rag.adapters.ingestion.loaders.obsidian_markdown_loader import (
    ObsidianMarkdownLoader,
    _extract_inline_tags,
    _normalize_tags,
    _split_fenced_code_blocks,
    _strip_wikilinks_outside_code,
    classify_note,
    split_obsidian_frontmatter,
)


class TestSplitObsidianFrontmatter:
    """Tests for split_obsidian_frontmatter function."""

    def test_no_frontmatter(self):
        text = "Just some content\nwithout frontmatter."
        fm, content = split_obsidian_frontmatter(text)
        assert fm == {}
        assert content == text

    def test_empty_frontmatter(self):
        text = "---\n---\nContent after empty frontmatter."
        fm, content = split_obsidian_frontmatter(text)
        assert fm == {}
        assert content == "Content after empty frontmatter."

    def test_simple_frontmatter(self):
        text = """---
title: My Note
author: Test Author
---
Body content here."""
        fm, content = split_obsidian_frontmatter(text)
        assert fm["title"] == "My Note"
        assert fm["author"] == "Test Author"
        assert "Body content here." in content

    def test_frontmatter_with_tags_list(self):
        text = """---
tags: [python, testing, rag]
---
Content."""
        fm, _ = split_obsidian_frontmatter(text)
        assert fm["tags"] == ["python", "testing", "rag"]

    def test_frontmatter_with_quoted_values(self):
        text = """---
title: 'Quoted Title'
description: "Double quoted"
---
Content."""
        fm, _ = split_obsidian_frontmatter(text)
        assert fm["title"] == "Quoted Title"
        assert fm["description"] == "Double quoted"

    def test_unclosed_frontmatter(self):
        text = "---\ntitle: Missing close\nNo closing delimiter"
        fm, content = split_obsidian_frontmatter(text)
        assert fm == {}
        assert content == text

    def test_frontmatter_with_dashes_in_content(self):
        text = """---
title: Test
---
Content with --- dashes in it."""
        fm, content = split_obsidian_frontmatter(text)
        assert fm["title"] == "Test"
        assert "--- dashes" in content

    def test_only_opening_delimiter(self):
        text = "---"
        fm, content = split_obsidian_frontmatter(text)
        assert fm == {}
        assert content == text

    def test_too_few_lines(self):
        text = "---\n---"
        fm, content = split_obsidian_frontmatter(text)
        assert fm == {}
        assert content == text


class TestExtractInlineTags:
    """Tests for _extract_inline_tags function."""

    def test_simple_tags(self):
        text = "Some text #tag1 and #tag2 here."
        tags = _extract_inline_tags(text)
        assert tags == ["tag1", "tag2"]

    def test_nested_tags(self):
        text = "A note with #parent/child tag."
        tags = _extract_inline_tags(text)
        assert tags == ["parent/child"]

    def test_tags_with_numbers(self):
        text = "Version #v2 and #release123"
        tags = _extract_inline_tags(text)
        assert tags == ["release123", "v2"]

    def test_tags_with_dashes_underscores(self):
        text = "#my-tag and #my_tag here"
        tags = _extract_inline_tags(text)
        assert tags == ["my-tag", "my_tag"]

    def test_no_tags(self):
        text = "Plain text without any hashtags."
        tags = _extract_inline_tags(text)
        assert tags == []

    def test_deduplicates_tags(self):
        text = "#dup #other #dup again"
        tags = _extract_inline_tags(text)
        assert tags == ["dup", "other"]

    def test_ignores_hex_colors(self):
        # Tags require a word boundary before them
        text = "Color #fff or word#attached"
        tags = _extract_inline_tags(text)
        assert tags == ["fff"]  # #fff is preceded by space

    def test_tag_at_line_start(self):
        text = "#starttag\nMore text"
        tags = _extract_inline_tags(text)
        assert tags == ["starttag"]


class TestNormalizeTags:
    """Tests for _normalize_tags function."""

    def test_none_value(self):
        assert _normalize_tags(None) == []

    def test_single_string(self):
        assert _normalize_tags("single") == ["single"]

    def test_comma_separated_string(self):
        assert _normalize_tags("a, b, c") == ["a", "b", "c"]

    def test_list_of_strings(self):
        assert _normalize_tags(["one", "two", "three"]) == ["one", "two", "three"]

    def test_list_with_whitespace(self):
        assert _normalize_tags(["  padded  ", "normal"]) == ["padded", "normal"]

    def test_empty_string(self):
        assert _normalize_tags("") == []

    def test_empty_list(self):
        assert _normalize_tags([]) == []

    def test_list_with_empty_strings(self):
        assert _normalize_tags(["valid", "", "  ", "also_valid"]) == ["valid", "also_valid"]

    def test_non_string_returns_empty(self):
        assert _normalize_tags(123) == []
        assert _normalize_tags({"key": "value"}) == []


class TestClassifyNote:
    """Tests for classify_note function."""

    def test_moc_in_filename(self):
        assert classify_note("My MOC.md", {}, []) == "moc"
        assert classify_note("moc-index.md", {}, []) == "moc"

    def test_moc_in_frontmatter_type(self):
        assert classify_note("regular.md", {"type": "moc"}, []) == "moc"
        assert classify_note("regular.md", {"type": "MOC"}, []) == "moc"

    def test_moc_in_tags(self):
        assert classify_note("regular.md", {}, ["moc"]) == "moc"
        assert classify_note("regular.md", {}, ["other", "MOC", "tags"]) == "moc"

    def test_regular_note(self):
        assert classify_note("regular.md", {}, []) == "note"
        assert classify_note("notes.md", {"type": "article"}, ["python"]) == "note"


class TestSplitFencedCodeBlocks:
    """Tests for _split_fenced_code_blocks function."""

    def test_no_code_blocks(self):
        text = "Plain text\nwith multiple lines."
        segments = _split_fenced_code_blocks(text)
        assert len(segments) == 1
        assert segments[0][1] is False  # not code

    def test_single_code_block(self):
        text = "Before\n```python\ncode here\n```\nAfter"
        segments = _split_fenced_code_blocks(text)
        # Should have: before, code fence start, code content, code fence end, after
        code_segments = [s for s in segments if s[1] is True]
        non_code_segments = [s for s in segments if s[1] is False]
        assert len(code_segments) >= 1
        assert len(non_code_segments) >= 2
        assert any("code here" in s[0] for s in code_segments)

    def test_multiple_code_blocks(self):
        text = "Text\n```\ncode1\n```\nMiddle\n```\ncode2\n```\nEnd"
        segments = _split_fenced_code_blocks(text)
        code_segments = [s for s in segments if s[1] is True]
        assert len(code_segments) == 2

    def test_indented_fence(self):
        text = "List:\n  ```\n  indented code\n  ```\nDone"
        segments = _split_fenced_code_blocks(text)
        code_segments = [s for s in segments if s[1] is True]
        assert len(code_segments) >= 1


class TestStripWikilinksOutsideCode:
    """Tests for _strip_wikilinks_outside_code function."""

    def test_simple_wikilink(self):
        text = "See [[Other Note]] for details."
        result = _strip_wikilinks_outside_code(text)
        assert result == "See Other Note for details."

    def test_wikilink_with_alias(self):
        text = "Check [[Target Note|display text]] here."
        result = _strip_wikilinks_outside_code(text)
        assert result == "Check display text here."

    def test_wikilink_with_heading(self):
        text = "See [[Note#Section]] for info."
        result = _strip_wikilinks_outside_code(text)
        assert result == "See Note#Section for info."

    def test_wikilink_with_heading_and_alias(self):
        text = "Read [[Note#Section|the section]]."
        result = _strip_wikilinks_outside_code(text)
        assert result == "Read the section."

    def test_multiple_wikilinks(self):
        text = "Links: [[A]], [[B|beta]], [[C]]."
        result = _strip_wikilinks_outside_code(text)
        assert result == "Links: A, beta, C."

    def test_wikilink_inside_code_block_preserved(self):
        text = "Before\n```\n[[preserved]]\n```\nAfter [[stripped]]"
        result = _strip_wikilinks_outside_code(text)
        assert "[[preserved]]" in result
        assert "[[stripped]]" not in result
        assert "stripped" in result

    def test_wikilink_in_inline_code_still_stripped(self):
        # Inline code (`...`) is not protected, only fenced blocks
        text = "Inline `[[link]]` here"
        result = _strip_wikilinks_outside_code(text)
        assert result == "Inline `link` here"


class TestObsidianMarkdownLoader:
    """Tests for ObsidianMarkdownLoader class."""

    def test_load_simple_note(self, temp_vault: Path):
        note = write_note(temp_vault, "simple.md", SIMPLE_NOTE_CONTENT)
        loader = ObsidianMarkdownLoader(vault_dir=temp_vault)
        result = loader.load(note)

        assert result is not None
        content, meta = result
        assert "Simple Note" in content
        assert meta["classification"] == "note"

    def test_load_note_with_frontmatter(self, temp_vault: Path):
        note = write_note(temp_vault, "with_fm.md", NOTE_WITH_FRONTMATTER)
        loader = ObsidianMarkdownLoader(vault_dir=temp_vault)
        result = loader.load(note)

        assert result is not None
        _, meta = result
        assert meta["frontmatter"]["title"] == "Test Note"
        assert meta["frontmatter_tags"] == ["python", "test"]

    def test_load_note_with_inline_tags(self, temp_vault: Path):
        note = write_note(temp_vault, "inline_tags.md", NOTE_WITH_INLINE_TAGS)
        loader = ObsidianMarkdownLoader(vault_dir=temp_vault)
        result = loader.load(note)

        assert result is not None
        _, meta = result
        assert "inline" in meta["inline_tags"]
        assert "tags" in meta["inline_tags"]

    def test_load_note_with_wikilinks(self, temp_vault: Path):
        note = write_note(temp_vault, "with_links.md", NOTE_WITH_WIKILINKS)
        loader = ObsidianMarkdownLoader(vault_dir=temp_vault)
        result = loader.load(note)

        assert result is not None
        content, _ = result
        assert "Other Note" in content
        assert "alias" in content
        assert "[[" not in content

    def test_load_moc_note(self, temp_vault: Path):
        note = write_note(temp_vault, "Topic MOC.md", MOC_NOTE_CONTENT)
        loader = ObsidianMarkdownLoader(vault_dir=temp_vault)
        result = loader.load(note)

        assert result is not None
        _, meta = result
        assert meta["classification"] == "moc"

    def test_load_nonexistent_file(self, temp_vault: Path):
        loader = ObsidianMarkdownLoader(vault_dir=temp_vault)
        result = loader.load(temp_vault / "nonexistent.md")
        assert result is None

    def test_embed_expansion_simple(self, temp_vault: Path):
        write_note(temp_vault, "embedded.md", "Embedded content here.")
        main = write_note(temp_vault, "main.md", "Before embed\n![[embedded]]\nAfter embed")

        loader = ObsidianMarkdownLoader(vault_dir=temp_vault, expand_embeds=True)
        result = loader.load(main)

        assert result is not None
        content, _ = result
        assert "Embedded content here" in content
        assert "EMBED: embedded.md" in content

    def test_embed_expansion_disabled(self, temp_vault: Path):
        write_note(temp_vault, "embedded.md", "Embedded content here.")
        main = write_note(temp_vault, "main.md", "Before embed\n![[embedded]]\nAfter embed")

        loader = ObsidianMarkdownLoader(vault_dir=temp_vault, expand_embeds=False)
        result = loader.load(main)

        assert result is not None
        content, _ = result
        assert "Embedded content here" not in content

    def test_embed_expansion_nested(self, temp_vault: Path):
        write_note(temp_vault, "level2.md", "Level 2 content")
        write_note(temp_vault, "level1.md", "Level 1 with ![[level2]]")
        main = write_note(temp_vault, "main.md", "Main with ![[level1]]")

        loader = ObsidianMarkdownLoader(vault_dir=temp_vault, expand_embeds=True)
        result = loader.load(main)

        assert result is not None
        content, _ = result
        assert "Level 1" in content
        assert "Level 2 content" in content

    def test_embed_expansion_cycle_protection(self, temp_vault: Path):
        write_note(temp_vault, "a.md", "A embeds ![[b]]")
        write_note(temp_vault, "b.md", "B embeds ![[a]]")

        loader = ObsidianMarkdownLoader(vault_dir=temp_vault, expand_embeds=True)
        result = loader.load(temp_vault / "a.md")

        # Should not infinite loop; cycle should be broken
        assert result is not None
        content, _ = result
        assert "A embeds" in content

    def test_embed_expansion_max_depth(self, temp_vault: Path):
        # Create chain of 6 notes
        for i in range(6):
            content = f"Note {i} embeds ![[note{i+1}]]" if i < 5 else f"Note {i} final"
            write_note(temp_vault, f"note{i}.md", content)

        loader = ObsidianMarkdownLoader(
            vault_dir=temp_vault, expand_embeds=True, max_embed_depth=4
        )
        result = loader.load(temp_vault / "note0.md")

        assert result is not None
        content, _ = result
        # Should have expanded some but not all due to depth limit
        assert "Note 0" in content

    def test_embed_attachment_placeholder(self, temp_vault: Path):
        # Create a fake image
        img = temp_vault / "image.png"
        img.write_bytes(b"fake png data")
        note = write_note(temp_vault, "with_image.md", "An image: ![[image.png]]")

        loader = ObsidianMarkdownLoader(vault_dir=temp_vault, expand_embeds=True)
        result = loader.load(note)

        assert result is not None
        content, _ = result
        assert "[Embedded attachment: image.png]" in content

    def test_embed_missing_target(self, temp_vault: Path):
        note = write_note(temp_vault, "broken_embed.md", "Embed missing: ![[nonexistent]]")

        loader = ObsidianMarkdownLoader(vault_dir=temp_vault, expand_embeds=True)
        result = loader.load(note)

        assert result is not None
        content, _ = result
        # Missing embed should be removed (empty replacement)
        assert "![[" not in content

    def test_embed_in_subdirectory(self, temp_vault: Path):
        subdir = temp_vault / "notes"
        subdir.mkdir()
        (subdir / "embedded.md").write_text("Nested content")
        main = write_note(temp_vault, "main.md", "Main ![[embedded]]")

        loader = ObsidianMarkdownLoader(vault_dir=temp_vault, expand_embeds=True)
        result = loader.load(main)

        assert result is not None
        content, _ = result
        assert "Nested content" in content

    def test_code_block_content_preserved(self, temp_vault: Path):
        note = write_note(temp_vault, "code.md", CODE_BLOCK_NOTE)
        loader = ObsidianMarkdownLoader(vault_dir=temp_vault)
        result = loader.load(note)

        assert result is not None
        content, _ = result
        assert "[[wikilink]]" in content  # preserved in code block
        assert "[[stripped]]" not in content  # stripped outside
        assert "stripped" in content

    def test_embed_in_code_block_not_expanded(self, temp_vault: Path):
        write_note(temp_vault, "embedded.md", "Should not appear")
        note = write_note(
            temp_vault,
            "code_embed.md",
            "Example:\n```\n![[embedded]]\n```\nEnd.",
        )

        loader = ObsidianMarkdownLoader(vault_dir=temp_vault, expand_embeds=True)
        result = loader.load(note)

        assert result is not None
        content, _ = result
        assert "Should not appear" not in content
        assert "![[embedded]]" in content  # preserved in code block
