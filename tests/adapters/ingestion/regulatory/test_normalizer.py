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
            ParsedParagraph(
                text="(a) See § 50.36 for details.",
                level=1,
                prefix="a",
                subsection_tokens=("a",),
            ),
            ParsedParagraph(
                text="(1) More detail.",
                level=2,
                prefix="1",
                subsection_tokens=("1",),
            ),
        ],
    )


def test_normalize_section_to_markdown_contains_expected_fields() -> None:
    markdown = normalize_section_to_markdown(_section(), _config())
    frontmatter, content = split_obsidian_frontmatter(markdown)
    assert frontmatter["citation_key"] == "10 CFR §50.34"
    assert frontmatter["cross_references"] == ["10 CFR §50.36"]
    assert "## (a)" in content
    assert "### (a)(1)" in content


def test_normalize_part_writes_markdown_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "part-50"
    written = normalize_part([_section()], _config(), out_dir)
    assert len(written) == 1
    assert (out_dir / "50.34.md").exists()


def test_normalize_section_reconstructs_parent_prefix_for_compound_markers() -> None:
    section = ParsedSection(
        section_number="50.46",
        title="Acceptance criteria",
        part_number="50",
        paragraphs=[
            ParsedParagraph(
                text="(a) Intro paragraph.",
                level=1,
                prefix="a",
                subsection_tokens=("a",),
            ),
            ParsedParagraph(
                text="(3)(i) Compound child marker.",
                level=3,
                prefix="i",
                subsection_tokens=("3", "i"),
            ),
            ParsedParagraph(
                text="(ii) Sibling marker.",
                level=3,
                prefix="ii",
                subsection_tokens=("ii",),
            ),
            ParsedParagraph(
                text="(b)(1) New branch marker.",
                level=2,
                prefix="1",
                subsection_tokens=("b", "1"),
            ),
        ],
    )

    markdown = normalize_section_to_markdown(section, _config())
    _, content = split_obsidian_frontmatter(markdown)

    assert "## (a)" in content
    assert "#### (a)(3)(i)" in content
    assert "#### (a)(3)(ii)" in content
    assert "### (b)(1)" in content
