"""Tests for citation extraction."""

from __future__ import annotations

from rag.adapters.ingestion.case.citation_extractor import (
    extract_adams_accessions,
    extract_all_citations,
    extract_cfr_appendices,
    extract_cfr_parts,
    extract_cfr_sections,
    extract_dockets,
    extract_generic_communications,
    extract_nuregs,
)


class TestExtractCfrSections:
    """Strong CFR section references with explicit '10 CFR' anchor."""

    def test_basic_section(self) -> None:
        spans = extract_cfr_sections("See 10 CFR 50.46 for requirements.")
        assert len(spans) == 1
        assert spans[0].key == "cfr:10:50.46"
        assert spans[0].kind == "cfr"
        assert spans[0].raw == "10 CFR 50.46"
        assert spans[0].confidence == 0.95

    def test_section_with_subsections(self) -> None:
        spans = extract_cfr_sections("per 10 CFR 50.46(b)(1)(ii)")
        assert len(spans) == 1
        assert spans[0].key == "cfr:10:50.46(b)(1)(ii)"
        assert spans[0].attrs["subsections"] == ["b", "1", "ii"]

    def test_section_sign_variant(self) -> None:
        spans = extract_cfr_sections("10 CFR §50.46")
        assert len(spans) == 1
        assert spans[0].key == "cfr:10:50.46"

    def test_no_space_variant(self) -> None:
        """10CFR50.46 (no spaces) — rare but exists in ADAMS."""
        spans = extract_cfr_sections("10CFR50.46")
        assert len(spans) == 1
        assert spans[0].key == "cfr:10:50.46"

    def test_multiple_sections(self) -> None:
        text = "10 CFR 50.46 and 10 CFR 50.55a(g)(4) are applicable."
        spans = extract_cfr_sections(text)
        keys = {s.key for s in spans}
        assert "cfr:10:50.46" in keys
        assert "cfr:10:50.55a(g)(4)" in keys

    def test_title_not_10(self) -> None:
        """Handle non-Title-10 CFR refs (e.g., 40 CFR)."""
        spans = extract_cfr_sections("40 CFR 190.10")
        assert len(spans) == 1
        assert spans[0].key == "cfr:40:190.10"

    def test_span_offsets(self) -> None:
        text = "xxx 10 CFR 50.46 yyy"
        spans = extract_cfr_sections(text)
        assert text[spans[0].start : spans[0].end] == "10 CFR 50.46"

    def test_no_false_positive_on_plain_numbers(self) -> None:
        spans = extract_cfr_sections("The value was 50.46 percent.")
        assert len(spans) == 0

    def test_source_field_default(self) -> None:
        spans = extract_cfr_sections("10 CFR 50.46")
        assert spans[0].source_field == "content"

    def test_source_field_override(self) -> None:
        spans = extract_cfr_sections("10 CFR 50.46", source_field="title")
        assert spans[0].source_field == "title"

    def test_letter_suffix_section(self) -> None:
        """Sections like 50.55a — letter suffix on section number."""
        spans = extract_cfr_sections("10 CFR 50.55a")
        assert spans[0].key == "cfr:10:50.55a"

    def test_subsection_with_roman_numerals(self) -> None:
        spans = extract_cfr_sections("10 CFR 50.46(b)(5)(iii)")
        assert spans[0].attrs["subsections"] == ["b", "5", "iii"]


class TestExtractCfrParts:
    def test_basic_part(self) -> None:
        spans = extract_cfr_parts("10 CFR Part 50")
        assert len(spans) == 1
        assert spans[0].key == "cfrpart:10:50"
        assert spans[0].kind == "cfrpart"
        assert spans[0].confidence == 0.90

    def test_part_without_title(self) -> None:
        """'Part 50' without '10 CFR' prefix — lower confidence."""
        spans = extract_cfr_parts("Part 50 requires...")
        assert len(spans) == 1
        assert spans[0].key == "cfrpart:10:50"
        assert spans[0].confidence == 0.70

    def test_multiple_parts(self) -> None:
        spans = extract_cfr_parts("10 CFR Part 50 and 10 CFR Part 21")
        keys = {s.key for s in spans}
        assert keys == {"cfrpart:10:50", "cfrpart:10:21"}

    def test_non_title_10_part(self) -> None:
        spans = extract_cfr_parts("40 CFR Part 190")
        assert spans[0].key == "cfrpart:40:190"


class TestExtractCfrAppendices:
    def test_appendix_b_to_part_50(self) -> None:
        spans = extract_cfr_appendices("10 CFR Part 50, Appendix B")
        assert len(spans) == 1
        assert spans[0].key == "cfrapp:10:50:appendix-b"
        assert spans[0].kind == "cfrapp"

    def test_appendix_a_to_part_100(self) -> None:
        spans = extract_cfr_appendices("Appendix A to 10 CFR Part 100")
        assert len(spans) == 1
        assert spans[0].key == "cfrapp:10:100:appendix-a"

    def test_no_false_positive_on_bare_appendix(self) -> None:
        """'Appendix B' alone without a part reference is not extracted."""
        spans = extract_cfr_appendices("See Appendix B for details.")
        assert len(spans) == 0


class TestExtractDockets:
    def test_docket_no_form(self) -> None:
        spans = extract_dockets("Docket No. 50-247")
        assert len(spans) == 1
        assert spans[0].key == "docket:50-247"
        assert spans[0].kind == "docket"
        assert spans[0].confidence == 0.90

    def test_docket_number_form(self) -> None:
        spans = extract_dockets("Docket Number 50-247")
        assert len(spans) == 1
        assert spans[0].key == "docket:50-247"

    def test_fixed_width_form(self) -> None:
        """ADAMS metadata uses 8-digit form like 05000247."""
        spans = extract_dockets("05000247")
        assert len(spans) == 1
        assert spans[0].key == "docket:50-247"

    def test_fixed_width_70_series(self) -> None:
        spans = extract_dockets("07007002")
        assert len(spans) == 1
        assert spans[0].key == "docket:70-7002"

    def test_hyphenated_form(self) -> None:
        spans = extract_dockets("50-247")
        assert len(spans) == 1
        assert spans[0].key == "docket:50-247"
        assert spans[0].confidence == 0.75  # lower — more ambiguous

    def test_docket_nos_plural(self) -> None:
        spans = extract_dockets("Docket Nos. 50-247 and 50-286")
        keys = {s.key for s in spans}
        assert keys == {"docket:50-247", "docket:50-286"}

    def test_no_false_positive_on_dates(self) -> None:
        """Dates like '95-07' should not match as dockets."""
        spans = extract_dockets("dated 95-07 and filed on 97-10")
        assert len(spans) == 0

    def test_no_false_positive_on_short_numbers(self) -> None:
        spans = extract_dockets("page 247 of the report")
        assert len(spans) == 0


class TestExtractAdamsAccessions:
    def test_modern_accession(self) -> None:
        spans = extract_adams_accessions("See ML021910673 for details.")
        assert len(spans) == 1
        assert spans[0].key == "adams:ML021910673"
        assert spans[0].kind == "adams"
        assert spans[0].confidence == 0.90

    def test_modern_accession_various_prefixes(self) -> None:
        """ML is most common but other prefixes exist."""
        for acc in ["ML021910673", "ML20108D163"]:
            spans = extract_adams_accessions(acc)
            assert len(spans) == 1
            assert spans[0].key == f"adams:{acc}"

    def test_multiple_accessions(self) -> None:
        text = "Documents ML021910673 and ML20108D163 were reviewed."
        spans = extract_adams_accessions(text)
        assert len(spans) == 2

    def test_legacy_numeric_accession(self) -> None:
        """10-digit numeric legacy accession like 8111110271."""
        spans = extract_adams_accessions("document 8111110271 was filed")
        assert len(spans) == 1
        assert spans[0].key == "adamslegacy:8111110271"
        assert spans[0].confidence == 0.60

    def test_no_false_positive_on_phone_numbers(self) -> None:
        """Phone numbers should not match."""
        spans = extract_adams_accessions("call (301) 564-3309")
        assert len(spans) == 0

    def test_no_false_positive_on_dates(self) -> None:
        spans = extract_adams_accessions("on 20060412 the report was filed")
        # This is tricky — 8 digits, not 10
        assert len(spans) == 0

    def test_accession_case_preserved(self) -> None:
        spans = extract_adams_accessions("ml021910673")
        assert len(spans) == 1
        assert spans[0].key == "adams:ML021910673"  # uppercased


class TestExtractNuregs:
    def test_nureg_dash_form(self) -> None:
        spans = extract_nuregs("See NUREG-0800 for guidance.")
        assert len(spans) == 1
        assert spans[0].key == "nureg:0800"
        assert spans[0].kind == "nureg"
        assert spans[0].confidence == 0.90

    def test_nureg_slash_form(self) -> None:
        """Some documents use NUREG/CR-1234 format."""
        spans = extract_nuregs("NUREG/CR-1234")
        assert len(spans) == 1
        assert spans[0].key == "nureg:cr:1234"

    def test_nureg_br_form(self) -> None:
        """NUREG/BR series (brochures)."""
        spans = extract_nuregs("NUREG/BR-1234")
        assert len(spans) == 1
        assert spans[0].key == "nureg:br:1234"

    def test_nureg_cp_form(self) -> None:
        """NUREG/CP series (conference proceedings)."""
        spans = extract_nuregs("NUREG/CP-1234")
        assert len(spans) == 1
        assert spans[0].key == "nureg:cp:1234"

    def test_nureg_with_rev(self) -> None:
        """NUREG with revision like NUREG-0800 Rev. 5."""
        spans = extract_nuregs("NUREG-0800 Rev. 5")
        assert len(spans) == 1
        assert spans[0].key == "nureg:0800:rev5"

    def test_multiple_nuregs(self) -> None:
        spans = extract_nuregs("NUREG-0800 and NUREG-1234")
        keys = {s.key for s in spans}
        assert "nureg:0800" in keys
        assert "nureg:1234" in keys


class TestExtractGenericCommunications:
    def test_ris_format(self) -> None:
        spans = extract_generic_communications("See RIS 2004-03 for details.")
        assert len(spans) == 1
        assert spans[0].key == "ris:2004-03"
        assert spans[0].kind == "ris"
        assert spans[0].confidence == 0.90

    def test_gl_format(self) -> None:
        spans = extract_generic_communications("Refer to GL 2004-01.")
        assert len(spans) == 1
        assert spans[0].key == "gl:2004-01"
        assert spans[0].kind == "gl"

    def test_in_format(self) -> None:
        spans = extract_generic_communications("See IN 2004-05.")
        assert len(spans) == 1
        assert spans[0].key == "in:2004-05"
        assert spans[0].kind == "in"

    def test_full_name_variants(self) -> None:
        """Full names like 'Regulatory Issue Summary 2004-03'."""
        spans = extract_generic_communications("Regulatory Issue Summary 2004-03")
        assert len(spans) == 1
        assert spans[0].key == "ris:2004-03"

    def test_generic_letter_full(self) -> None:
        """Full 'Generic Letter' name."""
        spans = extract_generic_communications("Generic Letter 2004-01")
        assert len(spans) == 1
        assert spans[0].key == "gl:2004-01"

    def test_information_notice_full(self) -> None:
        """Full 'Information Notice' name."""
        spans = extract_generic_communications("Information Notice 2004-05")
        assert len(spans) == 1
        assert spans[0].key == "in:2004-05"

    def test_multiple_gc(self) -> None:
        text = "RIS 2004-03 and GL 2004-01 were issued."
        spans = extract_generic_communications(text)
        keys = {s.key for s in spans}
        assert "ris:2004-03" in keys
        assert "gl:2004-01" in keys


class TestExtractAllCitations:
    def test_extracts_all_kinds(self) -> None:
        text = "10 CFR 50.46, NUREG-0800, RIS 2004-03, and Docket No. 50-247"
        result = extract_all_citations(text)
        keys = {s.key for s in result.spans}
        assert "cfr:10:50.46" in keys
        assert "nureg:0800" in keys
        assert "ris:2004-03" in keys
        assert "docket:50-247" in keys

    def test_returns_unique_keys(self) -> None:
        """Duplicate citations should be deduplicated."""
        text = "10 CFR 50.46 and 10 CFR 50.46"
        result = extract_all_citations(text)
        assert len(result.spans) == 1
        assert result.unique_keys == {"cfr:10:50.46"}

    def test_high_confidence_filter(self) -> None:
        """high_confidence_only should filter to confidence >= 0.85."""
        text = "10 CFR 50.46 and Part 50"  # Part 50 has 0.70 confidence
        result = extract_all_citations(text, high_confidence_only=True)
        keys = {s.key for s in result.spans}
        assert "cfr:10:50.46" in keys
        assert "cfrpart:10:50" not in keys

    def test_by_kind_grouping(self) -> None:
        text = "10 CFR 50.46 and NUREG-0800"
        result = extract_all_citations(text)
        assert result.by_kind["cfr"][0].key == "cfr:10:50.46"
        assert result.by_kind["nureg"][0].key == "nureg:0800"

    def test_empty_text(self) -> None:
        result = extract_all_citations("")
        assert len(result.spans) == 0
        assert result.unique_keys == set()

    def test_no_false_positives(self) -> None:
        """Plain numbers should not be extracted."""
        result = extract_all_citations("The value was 50.46 percent.")
        assert len(result.spans) == 0


class TestCitationExtractorIntegration:
    """Integration tests with realistic document content."""

    def test_typical_inspection_report(self) -> None:
        """Extract citations from a typical inspection report paragraph."""
        text = """
        NRC Inspection Report No. 05000247/2024001
        
        The licensee was in compliance with 10 CFR 50.46(b)(1) requirements.
        The NRC staff reviewed the facility's 10 CFR Part 50, Appendix B program.
        Reference NUREG-0800, Section 4.2 for acceptance criteria.
        See also RIS 2004-03 and Generic Letter 2004-01 for guidance.
        Docket No. 50-247 was reviewed.
        """
        result = extract_all_citations(text)
        keys = {s.key for s in result.spans}

        # CFR sections
        assert "cfr:10:50.46(b)(1)" in keys
        # CFR parts
        assert "cfrpart:10:50" in keys
        # CFR appendices
        assert "cfrapp:10:50:appendix-b" in keys
        # NUREG
        assert "nureg:0800" in keys
        # Generic communications
        assert "ris:2004-03" in keys
        assert "gl:2004-01" in keys
        # Dockets
        assert "docket:50-247" in keys

    def test_ocr_degraded_text(self) -> None:
        """Handle OCR artifacts like split characters."""
        text = "In accordance with 10 C F R 50.46, the licensee..."
        result = extract_all_citations(text)
        keys = {s.key for s in result.spans}
        # The text normalizer should fix "C F R" to "CFR"
        assert "cfr:10:50.46" in keys

    def test_mixed_content_with_noise(self) -> None:
        """Extract citations from text with dates, numbers, and noise."""
        text = """
        On 2024-01-15, the NRC issued a letter regarding 10 CFR 50.55a(g)(4).
        The docket 50-247 was reviewed. Phone: (301) 564-3309.
        Reference ML021910673 for historical context.
        Also see 10 CFR Part 21 and NUREG/CR-1234 Rev. 2.
        """
        result = extract_all_citations(text)
        keys = {s.key for s in result.spans}

        assert "cfr:10:50.55a(g)(4)" in keys
        assert "docket:50-247" in keys
        assert "adams:ML021910673" in keys
        assert "cfrpart:10:21" in keys
        assert "nureg:cr:1234:rev2" in keys
        # Phone number should NOT be extracted
        assert "301-564-3309" not in keys
        # Date should NOT be extracted as docket
        assert "2024-01-15" not in keys

    def test_citation_confidence_ranking(self) -> None:
        """Higher confidence citations are preferred in deduplication."""
        # "10 CFR Part 50" has confidence 0.90, "Part 50" alone has 0.70
        text = "10 CFR Part 50 requirements and Part 50 guidance"
        result = extract_all_citations(text)
        # Should only have one cfrpart:10:50
        cfrpart_spans = [s for s in result.spans if s.key == "cfrpart:10:50"]
        assert len(cfrpart_spans) == 1
        # Should be the higher confidence one
        assert cfrpart_spans[0].confidence == 0.90

    def test_by_kind_grouping_comprehensive(self) -> None:
        """Verify by_kind grouping works with all citation types."""
        text = """
        10 CFR 50.46, 10 CFR Part 50, Appendix A to 10 CFR Part 100,
        Docket No. 50-247, ML021910673, NUREG-0800, RIS 2004-03
        """
        result = extract_all_citations(text)

        assert "cfr" in result.by_kind
        assert "cfrpart" in result.by_kind
        assert "cfrapp" in result.by_kind
        assert "docket" in result.by_kind
        assert "adams" in result.by_kind
        assert "nureg" in result.by_kind
        assert "ris" in result.by_kind
