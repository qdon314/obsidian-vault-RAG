from __future__ import annotations

from pathlib import Path

import pytest

from rag.adapters.ingestion.regulatory.ecfr_parser import (
    CrossRef,
    ParsedParagraph,
    ParsedSection,
    SectionAmendment,
    parse_ecfr_xml,
)

FIXTURE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<CFRGRANULE>
  <DIV5 N="Part 50" TYPE="PART">
    <DIV8 N="50.34" TYPE="SECTION">
      <HEAD>§ 50.34 Contents of applications; technical information.</HEAD>
      <P>(a) First paragraph.</P>
      <P>(1) Nested paragraph.</P>
    </DIV8>
  </DIV5>
</CFRGRANULE>
"""


def test_parse_ecfr_xml_extracts_sections_and_levels() -> None:
    sections = parse_ecfr_xml(FIXTURE_XML)
    assert len(sections) == 1
    assert sections[0].section_number == "50.34"
    assert sections[0].part_number == "50"
    assert sections[0].paragraphs[0].level == 1
    assert sections[0].paragraphs[1].level == 2
    assert sections[0].paragraphs[0].subsection_tokens == ("a",)
    assert sections[0].paragraphs[1].subsection_tokens == ("1",)


def test_parse_ecfr_xml_extracts_subsection_chains() -> None:
    fixture = """\
<?xml version="1.0" encoding="UTF-8"?>
<CFRGRANULE>
  <DIV5 N="Part 50" TYPE="PART">
    <DIV8 N="50.46" TYPE="SECTION">
      <HEAD>§ 50.46 Acceptance criteria.</HEAD>
      <P>(a)(1)(i) First clause.</P>
      <P>(ii) Sibling clause.</P>
      <P>(b)(1) New branch.</P>
    </DIV8>
  </DIV5>
</CFRGRANULE>
"""
    sections = parse_ecfr_xml(fixture)
    paragraphs = sections[0].paragraphs

    assert paragraphs[0].subsection_tokens == ("a", "1", "i")
    assert paragraphs[0].level == 3
    assert paragraphs[0].prefix == "i"
    assert paragraphs[1].subsection_tokens == ("ii",)
    assert paragraphs[1].level == 3
    assert paragraphs[2].subsection_tokens == ("b", "1")
    assert paragraphs[2].level == 2


def test_cross_ref_dataclass_is_frozen() -> None:
    ref = CrossRef(target_citation="10 CFR §50.55a", kind="cfr")
    assert ref.target_citation == "10 CFR §50.55a"
    assert ref.kind == "cfr"


def test_section_amendment_dataclass_is_frozen() -> None:
    amend = SectionAmendment(
        amendment_id="20241230",
        ref_id="14",
        text="Link to an amendment published at 89 FR 106251, Dec. 30, 2024.",
    )
    assert amend.amendment_id == "20241230"
    assert amend.ref_id == "14"


def test_parsed_paragraph_has_cross_references_field() -> None:
    p = ParsedParagraph(text="See § 50.55a.", level=0, prefix=None)
    assert p.cross_references == ()


def test_parsed_section_has_amendments_field() -> None:
    s = ParsedSection(section_number="50.71", title="Maintenance of records", part_number="50")
    assert s.amendments == ()


XREF_FIXTURE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<CFRGRANULE>
  <DIV5 N="Part 50" TYPE="PART">
    <DIV8 N="50.71" TYPE="SECTION">
      <HEAD>§ 50.71 Maintenance of records, making of reports.</HEAD>
      <XREF ID="20241230" REFID="14" AMDINSN="15">Link to an amendment published at 89 FR 106251, Dec. 30, 2024.</XREF>
      <P>(a) First paragraph.</P>
    </DIV8>
  </DIV5>
</CFRGRANULE>
"""


def test_parse_ecfr_xml_extracts_xref_amendments() -> None:
    sections = parse_ecfr_xml(XREF_FIXTURE_XML)
    assert len(sections) == 1
    assert len(sections[0].amendments) == 1
    amend = sections[0].amendments[0]
    assert amend.amendment_id == "20241230"
    assert amend.ref_id == "14"
    assert "89 FR 106251" in amend.text


def test_parse_ecfr_xml_detects_cfr_cross_references() -> None:
    fixture = """\
<?xml version="1.0" encoding="UTF-8"?>
<CFRGRANULE>
  <DIV5 N="Part 50" TYPE="PART">
    <DIV8 N="50.55a" TYPE="SECTION">
      <HEAD>§ 50.55a Codes and standards.</HEAD>
      <P>(a) Licensees must comply with § 50.46 and 10 CFR 50.34.</P>
    </DIV8>
  </DIV5>
</CFRGRANULE>
"""
    sections = parse_ecfr_xml(fixture)
    para = sections[0].paragraphs[0]
    citations = {ref.target_citation for ref in para.cross_references}
    assert "10 CFR §50.46" in citations
    assert "10 CFR §50.34" in citations
    assert all(ref.kind == "cfr" for ref in para.cross_references)


def test_parse_ecfr_xml_detects_incorporated_standards() -> None:
    fixture = """\
<?xml version="1.0" encoding="UTF-8"?>
<CFRGRANULE>
  <DIV5 N="Part 50" TYPE="PART">
    <DIV8 N="50.55a" TYPE="SECTION">
      <HEAD>§ 50.55a Codes and standards.</HEAD>
      <P>(a) Systems must meet ASME Boiler and Pressure Vessel Code, Section III requirements and IEEE 323-1974 qualification standards.</P>
    </DIV8>
  </DIV5>
</CFRGRANULE>
"""
    sections = parse_ecfr_xml(fixture)
    para = sections[0].paragraphs[0]
    std_refs = [ref for ref in para.cross_references if ref.kind == "incorporated_standard"]
    std_citations = {ref.target_citation for ref in std_refs}
    assert any("ASME" in c for c in std_citations)
    assert any("IEEE 323-1974" in c or "IEEE" in c for c in std_citations)


def test_public_api_exports_new_types() -> None:
    from rag.adapters.ingestion.regulatory import CrossRef, SectionAmendment

    assert CrossRef is not None
    assert SectionAmendment is not None


@pytest.mark.skipif(
    not Path("data/ecfr/title-10-part-50.xml").exists(),
    reason="Real eCFR XML not available",
)
def test_real_xml_cross_references_detected() -> None:
    """Smoke test: real XML produces cross-references on at least some paragraphs."""
    xml_text = Path("data/ecfr/title-10-part-50.xml").read_text(encoding="utf-8")
    sections = parse_ecfr_xml(xml_text)
    assert len(sections) > 50  # part 50 has many sections

    # At least some paragraphs should have CFR cross-references
    cfr_ref_count = sum(
        1
        for section in sections
        for para in section.paragraphs
        if any(ref.kind == "cfr" for ref in para.cross_references)
    )
    assert cfr_ref_count > 0, "Expected CFR cross-references in real XML"

    # At least some paragraphs should have incorporated standards
    std_ref_count = sum(
        1
        for section in sections
        for para in section.paragraphs
        if any(ref.kind == "incorporated_standard" for ref in para.cross_references)
    )
    assert std_ref_count > 0, "Expected incorporated standard references in real XML"

    # § 50.71 should have an amendment (XREF tag)
    sec_50_71 = [s for s in sections if s.section_number == "50.71"]
    if sec_50_71:
        assert len(sec_50_71[0].amendments) >= 1
