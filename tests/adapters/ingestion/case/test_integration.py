"""Integration test showing complete adapter workflow."""

import dataclasses
from datetime import datetime

from rag.adapters.ingestion.case.document_adapter import adams_document_to_case_document
from rag.adapters.ingestion.case.normalizer import (
    CaseNormalizationConfig,
    extract_case_metadata,
    normalize_case_document_to_markdown,
)
from rag.domain.case_documents import CaseCategory, CaseSubcategory
from rag.ports.nrc_adams_client import AdamsDocument


class TestIntegrationWorkflow:
    """Test complete workflow from AdamsDocument to markdown."""

    def test_complete_workflow_inspection_report(self):
        """Should convert AdamsDocument through domain model to markdown."""
        # Step 1: AdamsDocument from API (port layer)
        adams_doc = AdamsDocument(
            accession_number="ML20123A456",
            title="Inspection Report 05000289/2020001 - Three Mile Island Unit 1",
            document_type="Inspection Report",
            document_date="2020-04-15",
            author_name="Inspector A. Smith",
            author_affiliation="NRC Region I",
            docket_numbers=["05000289"],
            license_numbers=["DPR-50"],
            keywords=["inspection", "reactor operations"],
            content="INSPECTION REPORT\n\nFacility: Three Mile Island Unit 1\n\n"
            "This inspection reviewed compliance with 10 CFR Part 50 and "
            "specifically §50.46 requirements...",
            url="https://www.nrc.gov/docs/ML2012/ML20123A456.html",
            estimated_pages=25,
        )

        # Step 2: Convert to domain model (CaseDocument)
        case_doc = adams_document_to_case_document(adams_doc)

        # Verify conversion
        assert case_doc.accession_number == "ML20123A456"
        assert case_doc.document_date == datetime(2020, 4, 15)
        assert case_doc.docket_numbers == ("05000289",)  # Converted to tuple
        assert case_doc.license_numbers == ("DPR-50",)
        assert case_doc.keywords == ("inspection", "reactor operations")

        # Step 3: Extract metadata (classification, CFR references, etc.)
        metadata = extract_case_metadata(case_doc)

        assert metadata.case_category == CaseCategory.INSPECTION
        assert metadata.case_subcategory == CaseSubcategory.INSPECTION_ROUTINE_REPORT
        assert metadata.case_category_confidence >= 0.90
        assert "50" in metadata.regulation_parts
        assert "50.46" in metadata.regulation_sections

        # Step 4: Enrich case document with metadata
        case_doc_enriched = dataclasses.replace(case_doc, case_metadata=metadata)

        # Step 5: Generate markdown
        config = CaseNormalizationConfig(
            source_url="https://adams-api.nrc.gov",
            source_revision="2024-01",
        )
        markdown = normalize_case_document_to_markdown(case_doc_enriched, config)

        # Verify markdown structure
        assert "---" in markdown  # YAML frontmatter
        assert 'accession_number: "ML20123A456"' in markdown
        assert 'case_category: "inspection"' in markdown
        assert 'dockets: ["05000289"]' in markdown
        assert 'regulation_parts: ["50"]' in markdown
        assert 'regulation_sections: ["50.46"]' in markdown
        assert "# ML20123A456 —" in markdown  # Main heading
        assert "**Category:** inspection" in markdown
        assert "## Content" in markdown

    def test_complete_workflow_enforcement_action(self):
        """Should handle enforcement document workflow."""
        # API response
        adams_doc = AdamsDocument(
            accession_number="ML19234B567",
            title="Notice of Violation - Facility XYZ",
            document_type="Notice of Violation",
            document_date="2019-08-22T15:30:00Z",
            author_name="Regional Administrator",
            author_affiliation="NRC Region II",
            addressee="Plant Manager",
            addressee_affiliation="Utility Company",
            docket_numbers=["05000123", "05000124"],
            content="NOTICE OF VIOLATION\n\nViolations of 10 CFR 50.59 and §50.73 identified...",
        )

        # Convert and enrich
        case_doc = adams_document_to_case_document(adams_doc)
        metadata = extract_case_metadata(case_doc)

        # Verify classification
        assert metadata.case_category == CaseCategory.ENFORCEMENT
        assert metadata.case_subcategory == CaseSubcategory.ENFORCEMENT_NOTICE_OF_VIOLATION
        # Note: "50.59" format extracts section but not the part
        assert "50.59" in metadata.regulation_sections
        assert "50.73" in metadata.regulation_sections

        # Generate output
        case_doc_enriched = dataclasses.replace(case_doc, case_metadata=metadata)
        config = CaseNormalizationConfig()
        markdown = normalize_case_document_to_markdown(case_doc_enriched, config)

        assert 'case_category: "enforcement"' in markdown
        assert 'case_subcategory: "enforcement_notice_of_violation"' in markdown
        assert 'dockets: ["05000123", "05000124"]' in markdown

    def test_workflow_with_missing_fields(self):
        """Should handle minimal documents gracefully."""
        # Minimal API response
        adams_doc = AdamsDocument(
            accession_number="ML99999Z999",
            title="Miscellaneous Document",
            document_type="Correspondence",
        )

        # Convert
        case_doc = adams_document_to_case_document(adams_doc)

        # Verify defaults
        assert case_doc.document_date is None
        assert case_doc.content is None
        assert case_doc.docket_numbers == ()
        assert case_doc.case_metadata.case_category == CaseCategory.UNKNOWN

        # Should still generate valid markdown
        config = CaseNormalizationConfig()
        markdown = normalize_case_document_to_markdown(case_doc, config)

        assert 'accession_number: "ML99999Z999"' in markdown
        assert 'case_category: "unknown"' in markdown
