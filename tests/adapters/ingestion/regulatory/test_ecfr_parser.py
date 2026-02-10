from __future__ import annotations

from rag.adapters.ingestion.regulatory.ecfr_parser import parse_ecfr_xml

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
