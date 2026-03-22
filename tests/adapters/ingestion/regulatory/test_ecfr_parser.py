from __future__ import annotations

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
