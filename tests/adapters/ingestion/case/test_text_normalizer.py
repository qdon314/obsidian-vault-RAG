"""Tests for citation-oriented text normalization."""

from __future__ import annotations

from rag.adapters.ingestion.case.text_normalizer import normalize_for_citation_extraction


class TestNormalizeForCitationExtraction:
    def test_collapse_whitespace(self) -> None:
        assert normalize_for_citation_extraction("10  CFR   50.46") == "10 CFR 50.46"

    def test_fix_ocr_split_cfr(self) -> None:
        assert "10 CFR" in normalize_for_citation_extraction("10 C F R 50.46")

    def test_normalize_cfr_variants(self) -> None:
        result = normalize_for_citation_extraction("10 C.F.R. 50.46")
        assert "10 CFR" in result

    def test_unicode_dashes(self) -> None:
        result = normalize_for_citation_extraction("NUREG\u20130800")
        assert "NUREG-0800" in result

    def test_unicode_quotes_and_section_signs(self) -> None:
        result = normalize_for_citation_extraction("\u201c10 CFR 50.46\u201d")
        assert '"10 CFR 50.46"' in result

    def test_hard_line_breaks_collapsed(self) -> None:
        """Single newlines within a paragraph become spaces."""
        result = normalize_for_citation_extraction("10 CFR\n50.46")
        assert "10 CFR 50.46" in result

    def test_paragraph_breaks_preserved(self) -> None:
        """Double newlines (paragraph boundaries) are preserved."""
        result = normalize_for_citation_extraction("paragraph one\n\nparagraph two")
        assert "\n\n" in result

    def test_empty_input(self) -> None:
        assert normalize_for_citation_extraction("") == ""

    def test_no_mutations_on_clean_text(self) -> None:
        clean = "In accordance with 10 CFR 50.46(b)(1), the licensee shall..."
        assert normalize_for_citation_extraction(clean) == clean

    def test_code_of_federal_regulations_expanded(self) -> None:
        result = normalize_for_citation_extraction(
            "Title 10, Code of Federal Regulations, Section 50.46"
        )
        assert "10 CFR" in result
