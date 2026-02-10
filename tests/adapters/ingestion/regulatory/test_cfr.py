from __future__ import annotations

from pathlib import Path

from rag.adapters.ingestion.loaders.obsidian_markdown_loader import split_obsidian_frontmatter
from rag.adapters.ingestion.regulatory.cfr import (
    build_citation_manifest,
    extract_cross_references,
    normalize_cfr_part,
    rewrite_cross_references_to_wikilinks,
)

_SAMPLE_XML = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<ECFR>
  <DIV8 TYPE="SECTION" N="50.34">
    <HEAD>§ 50.34 Contents of applications; technical information.</HEAD>
    <P>(a) Each application shall include the information required by § 50.36.</P>
    <P>(1) The information must be complete and auditable.</P>
    <P>(b) The commission may request supplemental information.</P>
  </DIV8>
  <DIV8 TYPE="SECTION" N="50.36">
    <HEAD>§ 50.36 Technical specifications.</HEAD>
    <P>(a) Technical specifications are derived from safety analyses.</P>
  </DIV8>
</ECFR>
"""


def _write_source_xml(path: Path) -> Path:
    path.write_text(_SAMPLE_XML, encoding="utf-8")
    return path


def test_rewrite_cross_references_to_wikilinks() -> None:
    text = "See § 50.36 and 10 CFR §50.34, but keep [[10 CFR §50.36]] unchanged."

    rewritten = rewrite_cross_references_to_wikilinks(text)

    assert "[[10 CFR §50.36]]" in rewritten
    assert "[[10 CFR §50.34]]" in rewritten
    assert "[[[[" not in rewritten


def test_extract_cross_references_deduplicates_and_normalizes() -> None:
    text = "See § 50.36, 10 CFR §50.36(a), and [[10 CFR §50.34]]."

    refs = extract_cross_references(text)

    assert refs == ["10 CFR §50.34", "10 CFR §50.36"]


def test_normalize_cfr_part_outputs_canonical_markdown(tmp_path: Path) -> None:
    source_xml = _write_source_xml(tmp_path / "part-50.xml")
    out_dir = tmp_path / "corpus" / "us-nrc" / "10-CFR"

    written = normalize_cfr_part(
        raw_source_path=source_xml,
        output_dir=out_dir,
        regime="US-NRC",
        instrument="10-CFR",
        part=50,
        instrument_version="2025-01-01",
        source_url="https://example.test/ecfr/title-10-part-50.xml",
        source_revision="ecfr-2025-01-01",
        effective_date="2025-01-01",
    )

    assert [p.name for p in written] == ["50.34.md", "50.36.md"]

    target = out_dir / "part-50" / "50.34.md"
    frontmatter, content = split_obsidian_frontmatter(target.read_text(encoding="utf-8"))

    assert frontmatter["regime"] == "US-NRC"
    assert frontmatter["instrument"] == "10-CFR"
    assert frontmatter["citation_key"] == "10 CFR §50.34"
    assert frontmatter["corpus"] == "regulatory"
    assert frontmatter["cross_references"] == ["10 CFR §50.36"]

    assert "# 10 CFR §50.34" in content
    assert "## (a)" in content
    assert "### (1)" in content
    assert "[[10 CFR §50.36]]" in content


def test_normalize_cfr_part_is_deterministic(tmp_path: Path) -> None:
    source_xml = _write_source_xml(tmp_path / "part-50.xml")
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"

    normalize_cfr_part(
        raw_source_path=source_xml,
        output_dir=out_a,
        regime="US-NRC",
        instrument="10-CFR",
        part=50,
        instrument_version="2025-01-01",
        source_url="https://example.test/ecfr/title-10-part-50.xml",
        source_revision="ecfr-2025-01-01",
        effective_date="2025-01-01",
    )
    normalize_cfr_part(
        raw_source_path=source_xml,
        output_dir=out_b,
        regime="US-NRC",
        instrument="10-CFR",
        part=50,
        instrument_version="2025-01-01",
        source_url="https://example.test/ecfr/title-10-part-50.xml",
        source_revision="ecfr-2025-01-01",
        effective_date="2025-01-01",
    )

    for section_file in ["50.34.md", "50.36.md"]:
        text_a = (out_a / "part-50" / section_file).read_text(encoding="utf-8")
        text_b = (out_b / "part-50" / section_file).read_text(encoding="utf-8")
        assert text_a == text_b


def test_build_citation_manifest(tmp_path: Path) -> None:
    source_xml = _write_source_xml(tmp_path / "part-50.xml")
    out_dir = tmp_path / "corpus" / "us-nrc" / "10-CFR"

    normalize_cfr_part(
        raw_source_path=source_xml,
        output_dir=out_dir,
        regime="US-NRC",
        instrument="10-CFR",
        part=50,
        instrument_version="2025-01-01",
        source_url="https://example.test/ecfr/title-10-part-50.xml",
        source_revision="ecfr-2025-01-01",
        effective_date="2025-01-01",
    )

    manifest = build_citation_manifest(out_dir)

    assert manifest == {
        "10 CFR §50.34": "part-50/50.34.md",
        "10 CFR §50.36": "part-50/50.36.md",
    }
