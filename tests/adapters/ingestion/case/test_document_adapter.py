"""Tests for AdamsDocument → CaseDocument adapter."""

from datetime import UTC, datetime

import pytest

from rag.adapters.ingestion.case.document_adapter import adams_document_to_case_document
from rag.domain.case_documents import CaseCategory, CaseMetadata, CaseSubcategory
from rag.ports.nrc_adams_client import AdamsDocument


class TestAdamsDocumentToCaseDocument:
    """Test suite for the port-to-domain adapter."""

    def test_minimal_adams_document(self):
        """Should convert a minimal AdamsDocument with required fields only."""
        adams = AdamsDocument(
            accession_number="ML12345A678",
            title="Test Document",
            document_type="Inspection Report",
        )

        case = adams_document_to_case_document(adams)

        assert case.accession_number == "ML12345A678"
        assert case.title == "Test Document"
        assert case.document_type == "Inspection Report"
        assert case.document_date is None
        assert case.content is None
        assert case.author_name is None
        assert case.docket_numbers == ()
        assert case.license_numbers == ()
        assert case.keywords == ()
        assert case.url is None
        assert case.estimated_pages is None
        assert case.case_metadata.case_category == CaseCategory.UNKNOWN

    def test_full_adams_document_with_all_fields(self):
        """Should convert a fully populated AdamsDocument."""
        adams = AdamsDocument(
            accession_number="ML23456B789",
            title="Full Test Document",
            document_type="Notice of Violation",
            document_date="2024-01-15",
            author_name="John Doe",
            author_affiliation="NRC Region I",
            addressee="Jane Smith",
            addressee_affiliation="Facility XYZ",
            docket_numbers=["DOCKET-123", "DOCKET-456"],
            license_numbers=["LICENSE-001", "LICENSE-002"],
            keywords=["enforcement", "violation"],
            content="This is the document content.",
            url="https://www.nrc.gov/docs/ML2345/ML23456B789.html",
            estimated_pages=5,
        )

        case = adams_document_to_case_document(adams)

        assert case.accession_number == "ML23456B789"
        assert case.title == "Full Test Document"
        assert case.document_type == "Notice of Violation"
        assert case.document_date == datetime(2024, 1, 15)
        assert case.author_name == "John Doe"
        assert case.author_affiliation == "NRC Region I"
        assert case.addressee == "Jane Smith"
        assert case.addressee_affiliation == "Facility XYZ"
        assert case.docket_numbers == ("DOCKET-123", "DOCKET-456")
        assert case.license_numbers == ("LICENSE-001", "LICENSE-002")
        assert case.keywords == ("enforcement", "violation")
        assert case.content == "This is the document content."
        assert case.url == "https://www.nrc.gov/docs/ML2345/ML23456B789.html"
        assert case.estimated_pages == 5

    def test_date_parsing_iso_format(self):
        """Should parse ISO 8601 date string to datetime."""
        adams = AdamsDocument(
            accession_number="ML11111A111",
            title="Date Test",
            document_type="Test",
            document_date="2024-03-15",
        )

        case = adams_document_to_case_document(adams)

        assert case.document_date == datetime(2024, 3, 15)

    def test_date_parsing_iso_with_time(self):
        """Should parse ISO 8601 datetime string with time component."""
        adams = AdamsDocument(
            accession_number="ML22222B222",
            title="DateTime Test",
            document_type="Test",
            document_date="2024-03-15T14:30:00",
        )

        case = adams_document_to_case_document(adams)

        assert case.document_date == datetime(2024, 3, 15, 14, 30, 0)

    def test_date_parsing_iso_with_timezone_z(self):
        """Should parse ISO 8601 datetime with 'Z' timezone indicator."""
        adams = AdamsDocument(
            accession_number="ML33333C333",
            title="DateTime UTC Test",
            document_type="Test",
            document_date="2024-03-15T14:30:00Z",
        )

        case = adams_document_to_case_document(adams)

        assert case.document_date == datetime(2024, 3, 15, 14, 30, 0, tzinfo=UTC)

    def test_date_parsing_iso_with_timezone_offset(self):
        """Should parse ISO 8601 datetime with explicit timezone offset."""
        adams = AdamsDocument(
            accession_number="ML44444D444",
            title="DateTime Offset Test",
            document_type="Test",
            document_date="2024-03-15T14:30:00-05:00",
        )

        case = adams_document_to_case_document(adams)

        # datetime.fromisoformat handles the offset
        expected = datetime.fromisoformat("2024-03-15T14:30:00-05:00")
        assert case.document_date == expected

    def test_date_parsing_invalid_date_returns_none(self):
        """Should gracefully handle invalid date strings."""
        adams = AdamsDocument(
            accession_number="ML55555E555",
            title="Invalid Date Test",
            document_type="Test",
            document_date="not-a-date",
        )

        case = adams_document_to_case_document(adams)

        assert case.document_date is None

    def test_date_parsing_empty_string_returns_none(self):
        """Should handle empty date string gracefully."""
        adams = AdamsDocument(
            accession_number="ML66666F666",
            title="Empty Date Test",
            document_type="Test",
            document_date="",
        )

        case = adams_document_to_case_document(adams)

        assert case.document_date is None

    def test_list_to_tuple_conversion_dockets(self):
        """Should convert list of docket numbers to tuple."""
        adams = AdamsDocument(
            accession_number="ML77777G777",
            title="Docket Test",
            document_type="Test",
            docket_numbers=["DOCKET-001", "DOCKET-002", "DOCKET-003"],
        )

        case = adams_document_to_case_document(adams)

        assert case.docket_numbers == ("DOCKET-001", "DOCKET-002", "DOCKET-003")
        assert isinstance(case.docket_numbers, tuple)

    def test_list_to_tuple_conversion_licenses(self):
        """Should convert list of license numbers to tuple."""
        adams = AdamsDocument(
            accession_number="ML88888H888",
            title="License Test",
            document_type="Test",
            license_numbers=["LICENSE-A", "LICENSE-B"],
        )

        case = adams_document_to_case_document(adams)

        assert case.license_numbers == ("LICENSE-A", "LICENSE-B")
        assert isinstance(case.license_numbers, tuple)

    def test_list_to_tuple_conversion_keywords(self):
        """Should convert list of keywords to tuple."""
        adams = AdamsDocument(
            accession_number="ML99999I999",
            title="Keywords Test",
            document_type="Test",
            keywords=["keyword1", "keyword2", "keyword3"],
        )

        case = adams_document_to_case_document(adams)

        assert case.keywords == ("keyword1", "keyword2", "keyword3")
        assert isinstance(case.keywords, tuple)

    def test_empty_lists_become_empty_tuples(self):
        """Should convert empty lists to empty tuples."""
        adams = AdamsDocument(
            accession_number="ML00000J000",
            title="Empty Lists Test",
            document_type="Test",
            docket_numbers=[],
            license_numbers=[],
            keywords=[],
        )

        case = adams_document_to_case_document(adams)

        assert case.docket_numbers == ()
        assert case.license_numbers == ()
        assert case.keywords == ()

    def test_with_custom_case_metadata(self):
        """Should accept and use provided CaseMetadata."""
        custom_metadata = CaseMetadata(
            case_category=CaseCategory.ENFORCEMENT,
            case_subcategory=CaseSubcategory.ENFORCEMENT_NOTICE_OF_VIOLATION,
            case_category_confidence=0.95,
            case_category_method="rules",
            case_category_reasons=("keyword=NOV",),
        )

        adams = AdamsDocument(
            accession_number="ML12121K121",
            title="Metadata Test",
            document_type="Notice of Violation",
        )

        case = adams_document_to_case_document(adams, case_metadata=custom_metadata)

        assert case.case_metadata.case_category == CaseCategory.ENFORCEMENT
        assert (
            case.case_metadata.case_subcategory == CaseSubcategory.ENFORCEMENT_NOTICE_OF_VIOLATION
        )
        assert case.case_metadata.case_category_confidence == 0.95
        assert case.case_metadata.case_category_method == "rules"
        assert case.case_metadata.case_category_reasons == ("keyword=NOV",)

    def test_without_custom_metadata_uses_default(self):
        """Should use default empty CaseMetadata when none provided."""
        adams = AdamsDocument(
            accession_number="ML13131L131",
            title="Default Metadata Test",
            document_type="Test",
        )

        case = adams_document_to_case_document(adams)

        assert case.case_metadata.case_category == CaseCategory.UNKNOWN
        assert case.case_metadata.case_subcategory == CaseSubcategory.UNKNOWN
        assert case.case_metadata.case_category_confidence == 0.0
        assert case.case_metadata.case_category_method == "rules"
        assert case.case_metadata.case_category_reasons == ()

    def test_case_document_is_frozen(self):
        """Should produce a frozen CaseDocument."""
        adams = AdamsDocument(
            accession_number="ML14141M141",
            title="Frozen Test",
            document_type="Test",
        )

        case = adams_document_to_case_document(adams)

        with pytest.raises(AttributeError):
            case.title = "Modified Title"  # type: ignore

    def test_preserves_none_values(self):
        """Should preserve None values for optional fields."""
        adams = AdamsDocument(
            accession_number="ML15151N151",
            title="None Values Test",
            document_type="Test",
            document_date=None,
            author_name=None,
            content=None,
            url=None,
        )

        case = adams_document_to_case_document(adams)

        assert case.document_date is None
        assert case.author_name is None
        assert case.content is None
        assert case.url is None
        assert case.estimated_pages is None


class TestRealWorldScenarios:
    """Test realistic scenarios from NRC ADAMS."""

    def test_inspection_report_conversion(self):
        """Should convert a typical inspection report."""
        adams = AdamsDocument(
            accession_number="ML20123A456",
            title="Inspection Report 05000289/2020001",
            document_type="Inspection Report",
            document_date="2020-04-15",
            author_name="Inspector A. Smith",
            author_affiliation="NRC Region I",
            docket_numbers=["05000289"],
            license_numbers=["DPR-50"],
            keywords=["inspection", "reactor operations"],
            content="INSPECTION REPORT\n\nFacility: Three Mile Island Unit 1...",
            url="https://www.nrc.gov/docs/ML2012/ML20123A456.html",
            estimated_pages=25,
        )

        case = adams_document_to_case_document(adams)

        assert case.accession_number == "ML20123A456"
        assert case.document_type == "Inspection Report"
        assert case.document_date == datetime(2020, 4, 15)
        assert case.docket_numbers == ("05000289",)
        assert case.license_numbers == ("DPR-50",)
        assert "INSPECTION REPORT" in (case.content or "")

    def test_enforcement_action_conversion(self):
        """Should convert a typical enforcement action."""
        adams = AdamsDocument(
            accession_number="ML19234B567",
            title="Notice of Violation - Facility XYZ",
            document_type="Notice of Violation",
            document_date="2019-08-22T15:30:00Z",
            author_name="Regional Administrator",
            author_affiliation="NRC Region II",
            addressee="Plant Manager",
            addressee_affiliation="Utility Company",
            docket_numbers=["05000123", "05000124"],
            content="NOTICE OF VIOLATION\n\nViolation A: Failure to...",
            url="https://www.nrc.gov/docs/ML1923/ML19234B567.html",
        )

        case = adams_document_to_case_document(adams)

        assert case.document_type == "Notice of Violation"
        assert case.document_date == datetime(2019, 8, 22, 15, 30, 0, tzinfo=UTC)
        assert case.addressee == "Plant Manager"
        assert case.docket_numbers == ("05000123", "05000124")
        assert "NOTICE OF VIOLATION" in (case.content or "")

    def test_ler_conversion(self):
        """Should convert a Licensee Event Report."""
        adams = AdamsDocument(
            accession_number="ML21045C678",
            title="LER 2021-001, Event Date 2021-02-10",
            document_type="Licensee Event Report",
            document_date="2021-03-12",
            author_name="Nuclear Safety Officer",
            author_affiliation="Utility Company",
            docket_numbers=["05000456"],
            license_numbers=["NPF-12"],
            keywords=["LER", "reactor trip", "automatic scram"],
            content="LICENSEE EVENT REPORT\n\nEvent Description: Automatic reactor trip...",
            url="https://www.nrc.gov/docs/ML2104/ML21045C678.html",
            estimated_pages=12,
        )

        case = adams_document_to_case_document(adams)

        assert "LER" in case.title
        assert case.document_type == "Licensee Event Report"
        assert case.keywords == ("LER", "reactor trip", "automatic scram")
