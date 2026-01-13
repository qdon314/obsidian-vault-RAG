"""Tests for FilesystemIngestor adapter."""

from __future__ import annotations

from pathlib import Path

from rag.adapters.ingestion.filesystem import FilesystemIngestor
from rag.adapters.ingestion.loaders.text_loader import TextLoader


class TestFilesystemIngestorBasics:
    """Basic ingestion behavior."""

    def test_ingest_single_file(self, tmp_path: Path):
        """Ingests a single text file."""
        file = tmp_path / "test.md"
        file.write_text("# Test\n\nContent here.")

        ingestor = FilesystemIngestor(text_loader=TextLoader())
        docs, report = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 1
        assert "Content here" in docs[0].text
        assert report.loaded == 1

    def test_ingest_multiple_files(self, tmp_path: Path):
        """Ingests multiple files from a directory."""
        (tmp_path / "file1.md").write_text("File 1 content")
        (tmp_path / "file2.txt").write_text("File 2 content")

        ingestor = FilesystemIngestor(text_loader=TextLoader())
        docs, report = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 2
        assert report.loaded == 2

    def test_ingest_returns_documents(self, tmp_path: Path):
        """Ingested items are Document objects."""
        (tmp_path / "test.md").write_text("Test content")

        ingestor = FilesystemIngestor(text_loader=TextLoader())
        docs, _ = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 1
        doc = docs[0]
        assert hasattr(doc, "doc_id")
        assert hasattr(doc, "text")
        assert hasattr(doc, "uri")
        assert hasattr(doc, "metadata")

    def test_ingest_empty_directory(self, tmp_path: Path):
        """Ingesting empty directory returns empty results."""
        ingestor = FilesystemIngestor(text_loader=TextLoader())
        docs, report = ingestor.ingest([str(tmp_path)])

        assert docs == []
        assert report.loaded == 0


class TestFilesystemIngestorReport:
    """IngestReport generation."""

    def test_report_counts_scanned(self, tmp_path: Path):
        """Report includes count of scanned files."""
        (tmp_path / "file1.md").write_text("Content 1")
        (tmp_path / "file2.md").write_text("Content 2")
        (tmp_path / "file3.csv").write_text("a,b,c")  # Will be skipped

        ingestor = FilesystemIngestor(text_loader=TextLoader())
        _, report = ingestor.ingest([str(tmp_path)])

        assert report.scanned == 3

    def test_report_by_extension(self, tmp_path: Path):
        """Report breaks down loaded files by extension."""
        (tmp_path / "file1.md").write_text("Markdown")
        (tmp_path / "file2.md").write_text("More markdown")
        (tmp_path / "file3.txt").write_text("Plain text")

        ingestor = FilesystemIngestor(text_loader=TextLoader())
        _, report = ingestor.ingest([str(tmp_path)])

        assert report.by_extension.get(".md") == 2
        assert report.by_extension.get(".txt") == 1


class TestFilesystemIngestorFiltering:
    """File filtering behavior."""

    def test_skip_hidden_files(self, tmp_path: Path):
        """Hidden files are skipped by default."""
        (tmp_path / "visible.md").write_text("Visible content")
        (tmp_path / ".hidden.md").write_text("Hidden content")

        ingestor = FilesystemIngestor(text_loader=TextLoader(), skip_hidden=True)
        docs, report = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 1
        assert report.skipped_hidden == 1

    def test_include_hidden_files_when_disabled(self, tmp_path: Path):
        """Hidden files can be included."""
        (tmp_path / "visible.md").write_text("Visible")
        (tmp_path / ".hidden.md").write_text("Hidden")

        ingestor = FilesystemIngestor(text_loader=TextLoader(), skip_hidden=False)
        docs, report = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 2
        assert report.skipped_hidden == 0

    def test_skip_disallowed_extensions(self, tmp_path: Path):
        """Files with disallowed extensions are skipped."""
        (tmp_path / "document.md").write_text("Markdown")
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "image.png").write_bytes(b"fake png")

        ingestor = FilesystemIngestor(
            text_loader=TextLoader(),
            allowed_extensions={".md", ".txt"},
        )
        docs, report = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 1
        assert report.skipped_extension == 2

    def test_custom_allowed_extensions(self, tmp_path: Path):
        """Custom allowed extensions are respected."""
        (tmp_path / "code.py").write_text("print('hello')")
        (tmp_path / "doc.md").write_text("# Doc")

        ingestor = FilesystemIngestor(
            text_loader=TextLoader(),
            allowed_extensions={".py"},
        )
        docs, _ = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 1
        assert docs[0].text == "print('hello')"

    def test_skip_empty_files(self, tmp_path: Path):
        """Empty files are skipped."""
        (tmp_path / "content.md").write_text("Has content")
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "whitespace.md").write_text("   \n\t  ")

        ingestor = FilesystemIngestor(text_loader=TextLoader())
        docs, report = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 1
        assert report.skipped_empty == 2


class TestFilesystemIngestorRecursive:
    """Recursive directory traversal."""

    def test_recursive_traversal(self, tmp_path: Path):
        """Recursively loads files from subdirectories."""
        (tmp_path / "root.md").write_text("Root content")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("Nested content")

        ingestor = FilesystemIngestor(text_loader=TextLoader(), recursive=True)
        docs, report = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 2
        assert report.loaded == 2

    def test_non_recursive(self, tmp_path: Path):
        """Non-recursive mode only loads top-level files."""
        (tmp_path / "root.md").write_text("Root content")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("Nested content")

        ingestor = FilesystemIngestor(text_loader=TextLoader(), recursive=False)
        docs, _ = ingestor.ingest([str(tmp_path)])

        assert len(docs) == 1
        assert docs[0].text == "Root content"


class TestFilesystemIngestorDocumentMetadata:
    """Document metadata generation."""

    def test_doc_id_is_stable(self, tmp_path: Path):
        """Same file content produces same doc_id."""
        file = tmp_path / "test.md"
        file.write_text("Consistent content")

        ingestor = FilesystemIngestor(text_loader=TextLoader())

        docs1, _ = ingestor.ingest([str(tmp_path)])
        docs2, _ = ingestor.ingest([str(tmp_path)])

        assert docs1[0].doc_id == docs2[0].doc_id

    def test_doc_id_changes_with_content(self, tmp_path: Path):
        """Different content produces different doc_id."""
        file = tmp_path / "test.md"

        file.write_text("Content version 1")
        ingestor = FilesystemIngestor(text_loader=TextLoader())
        docs1, _ = ingestor.ingest([str(tmp_path)])

        file.write_text("Content version 2")
        docs2, _ = ingestor.ingest([str(tmp_path)])

        assert docs1[0].doc_id != docs2[0].doc_id

    def test_metadata_includes_uri(self, tmp_path: Path):
        """Document metadata includes the file URI."""
        file = tmp_path / "test.md"
        file.write_text("Content")

        ingestor = FilesystemIngestor(text_loader=TextLoader())
        docs, _ = ingestor.ingest([str(tmp_path)])

        assert "uri" in docs[0].metadata
        assert "test.md" in docs[0].metadata["uri"]

    def test_metadata_includes_extension(self, tmp_path: Path):
        """Document metadata includes file extension."""
        (tmp_path / "test.md").write_text("Markdown")

        ingestor = FilesystemIngestor(text_loader=TextLoader())
        docs, _ = ingestor.ingest([str(tmp_path)])

        assert docs[0].metadata.get("ext") == ".md"

    def test_source_name_configurable(self, tmp_path: Path):
        """Source name can be customized."""
        (tmp_path / "test.md").write_text("Content")

        ingestor = FilesystemIngestor(
            text_loader=TextLoader(),
            source_name="my_vault",
        )
        docs, _ = ingestor.ingest([str(tmp_path)])

        assert docs[0].source == "my_vault"


class TestFilesystemIngestorWithSampleVault:
    """Tests using the sample_vault fixture."""

    def test_ingest_sample_vault(self, sample_vault: Path):
        """Ingests the sample vault structure correctly."""
        ingestor = FilesystemIngestor(text_loader=TextLoader())
        _, report = ingestor.ingest([str(sample_vault)])

        # Should load: note1.md, note2.md, readme.txt, nested.md
        # Should skip: .hidden.md (hidden), data.csv (extension)
        assert report.loaded == 4
        assert report.skipped_hidden == 1
        assert report.skipped_extension == 1

    def test_ingest_sample_vault_md_only(self, sample_vault: Path):
        """Ingests only markdown files from sample vault."""
        ingestor = FilesystemIngestor(
            text_loader=TextLoader(),
            allowed_extensions={".md"},
        )
        docs, report = ingestor.ingest([str(sample_vault)])

        # Should load: note1.md, note2.md, nested.md
        assert report.loaded == 3
        assert all(doc.metadata.get("ext") == ".md" for doc in docs)
