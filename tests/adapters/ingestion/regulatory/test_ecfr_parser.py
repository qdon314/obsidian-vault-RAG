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
