from __future__ import annotations

from pathlib import Path

from rag.adapters.ingestion.loaders.obsidian_markdown_loader import split_obsidian_frontmatter
from rag.adapters.ingestion.regulatory.ecfr_parser import ParsedParagraph, ParsedSection
from rag.adapters.ingestion.regulatory.normalizer import (
    NormalizationConfig,
    normalize_part,
    normalize_section_to_markdown,
)


def _config() -> NormalizationConfig:
    return NormalizationConfig(
        regime="US-NRC",
        instrument="10-CFR",
        instrument_version="2025-01-01",
        source_url="https://example.test/ecfr",
        source_revision="ecfr-2025-01-01",
        effective_date="2025-01-01",
    )


def _section() -> ParsedSection:
    return ParsedSection(
        section_number="50.34",
        title="Contents of applications; technical information",
        part_number="50",
        paragraphs=[
            ParsedParagraph(text="(a) See § 50.36 for details.", level=1, prefix="a"),
            ParsedParagraph(text="(1) More detail.", level=2, prefix="1"),
        ],
    )


def test_normalize_section_to_markdown_contains_expected_fields() -> None:
    markdown = normalize_section_to_markdown(_section(), _config())
    frontmatter, content = split_obsidian_frontmatter(markdown)
    assert frontmatter["citation_key"] == "10 CFR §50.34"
    assert frontmatter["cross_references"] == ["10 CFR §50.36"]
    assert "## (a)" in content
    assert "### (1)" in content


def test_normalize_part_writes_markdown_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "part-50"
    written = normalize_part([_section()], _config(), out_dir)
    assert len(written) == 1
    assert (out_dir / "50.34.md").exists()
