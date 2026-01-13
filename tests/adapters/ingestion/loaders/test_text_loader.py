"""Tests for TextLoader adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.adapters.ingestion.loaders.text_loader import TextLoader


class TestTextLoaderBasics:
    """Basic file loading."""

    def test_load_simple_text_file(self, tmp_path: Path, text_loader: TextLoader):
        """Loads a simple text file successfully."""
        file = tmp_path / "test.txt"
        file.write_text("Hello, world!")

        content = text_loader.load(file)

        assert content == "Hello, world!"

    def test_load_utf8_content(self, tmp_path: Path, text_loader: TextLoader):
        """Loads UTF-8 encoded content correctly."""
        file = tmp_path / "unicode.txt"
        file.write_text("こんにちは世界 🌍", encoding="utf-8")

        content = text_loader.load(file)

        assert "こんにちは" in content if content else False
        assert "🌍" in content if content else False

    def test_load_empty_file(self, tmp_path: Path, text_loader: TextLoader):
        """Loading empty file returns empty string."""
        file = tmp_path / "empty.txt"
        file.write_text("")

        content = text_loader.load(file)

        assert content == ""

    def test_load_multiline_file(self, tmp_path: Path, text_loader: TextLoader):
        """Loads multi-line files preserving newlines."""
        file = tmp_path / "multiline.txt"
        file.write_text("line 1\nline 2\nline 3")

        content = text_loader.load(file)

        assert content == "line 1\nline 2\nline 3"


class TestTextLoaderNonexistent:
    """Handling of nonexistent files."""

    def test_load_nonexistent_file(self, tmp_path: Path, text_loader: TextLoader):
        """Loading nonexistent file returns None."""
        file = tmp_path / "does_not_exist.txt"

        content = text_loader.load(file)

        assert content is None


class TestTextLoaderSizeLimit:
    """File size limiting."""

    def test_load_file_under_limit(self, tmp_path: Path):
        """Files under max_bytes are loaded."""
        loader = TextLoader(max_bytes=1000)
        file = tmp_path / "small.txt"
        file.write_text("x" * 500)

        content = loader.load(file)

        assert content == "x" * 500

    def test_load_file_over_limit(self, tmp_path: Path):
        """Files over max_bytes return None."""
        loader = TextLoader(max_bytes=100)
        file = tmp_path / "large.txt"
        file.write_text("x" * 200)

        content = loader.load(file)

        assert content is None

    def test_load_file_exactly_at_limit(self, tmp_path: Path):
        """Files exactly at max_bytes are loaded."""
        loader = TextLoader(max_bytes=100)
        file = tmp_path / "exact.txt"
        file.write_text("x" * 100)

        content = loader.load(file)

        assert content == "x" * 100


class TestTextLoaderBinaryDetection:
    """Binary file detection."""

    def test_reject_binary_with_null_bytes(self, tmp_path: Path, text_loader: TextLoader):
        """Files with null bytes are rejected as binary."""
        file = tmp_path / "binary.bin"
        file.write_bytes(b"text\x00more text")

        content = text_loader.load(file)

        assert content is None

    def test_reject_high_control_char_ratio(self, tmp_path: Path, text_loader: TextLoader):
        """Files with many control characters are rejected."""
        file = tmp_path / "control.bin"
        # Create content with >2% control characters
        control_chars = bytes([1, 2, 3, 4, 5, 6, 7])  # Non-printable control chars
        text_chars = b"normal text " * 10
        file.write_bytes(control_chars * 5 + text_chars)

        content = text_loader.load(file)

        # Should be rejected due to control char ratio
        assert content is None

    def test_accept_normal_text_with_tabs_newlines(self, tmp_path: Path, text_loader: TextLoader):
        """Normal text with tabs and newlines is accepted."""
        file = tmp_path / "formatted.txt"
        file.write_text("line 1\n\tindented line\n\tanother")

        content = text_loader.load(file)

        assert content is not None
        assert "indented line" in content


class TestTextLoaderEncodingFallback:
    """Encoding detection and fallback."""

    def test_load_latin1_with_fallback(self, tmp_path: Path, text_loader: TextLoader):
        """Falls back to latin-1 for non-UTF-8 files."""
        file = tmp_path / "latin1.txt"
        # Write bytes that are valid latin-1 but not valid UTF-8
        file.write_bytes(b"caf\xe9")  # "café" in latin-1

        content = text_loader.load(file)

        assert content is not None
        # Content should be loaded (possibly with replacement chars)
        assert len(content) > 0

    def test_load_utf8_preferred(self, tmp_path: Path):
        """UTF-8 is preferred encoding."""
        loader = TextLoader(prefer_encoding="utf-8")
        file = tmp_path / "utf8.txt"
        file.write_text("UTF-8 content: café", encoding="utf-8")

        content = loader.load(file)

        assert content == "UTF-8 content: café"

    def test_custom_fallback_encoding(self, tmp_path: Path):
        loader = TextLoader(prefer_encoding="utf-8", fallback_encoding="cp1252")
        file = tmp_path / "windows.txt"
        
        # \u201c is the Unicode "Left Double Quote" that corresponds to cp1252's \x93
        file.write_bytes("Smart quotes: \u201chello\u201d".encode("cp1252"))

        content = loader.load(file)

        # FIX: Use a boolean check that doesn't include the 'content' string in the error message
        # If this fails, pytest only prints the message, not the character that crashes the terminal.
        if content and "\u201c" not in content:
            pytest.fail("The smart quote was not correctly decoded into Unicode.")
            
        assert content is not None
