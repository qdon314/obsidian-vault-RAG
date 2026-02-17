"""Tests for HttpNrcAdamsClient — uses unittest.mock for httpx."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.adapters.ingestion.case.http_client import (
    HttpNrcAdamsClient,
    RetryConfig,
    _parse_document,
    _parse_search_result,
)
from rag.domain.errors import (
    AdamsApiError,
    AdamsAuthError,
    AdamsRateLimitError,
)


def _make_response(status_code: int = 200, json_data: object = None, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = ""
    resp.headers = headers or {}
    return resp


def _make_search_page(results: list[dict]) -> dict:
    return {"results": results}


def _make_raw_result(accession: str = "ML24001A001", title: str = "Test Report") -> dict:
    return {
        "AccessionNumber": accession,
        "DocumentTitle": title,
        "DocumentType": "Inspection Report",
        "DocumentDate": "2024-01-15",
        "AuthorName": "John Doe",
        "AuthorAffiliation": "NRC",
        "DocketNumber": "05000001",
        "Url": f"https://adams.nrc.gov/doc/{accession}",
        "EstimatedPages": 10,
    }


def _make_raw_document(accession: str = "ML24001A001") -> dict:
    return {
        **_make_raw_result(accession),
        "AddresseeName": "Jane Smith",
        "AddresseeAffiliation": "Licensee Corp",
        "LicenseNumber": "NPF-1; NPF-2",
        "Keyword": "safety; inspection",
        "content": "Full text of the document.",
    }


NO_RETRY = RetryConfig(max_retries=0)
FAST_RETRY = RetryConfig(max_retries=2, base_delay=0.0, jitter=False)


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def adapter(mock_client: MagicMock) -> HttpNrcAdamsClient:
    return HttpNrcAdamsClient._for_test(client=mock_client, retry=NO_RETRY)


@pytest.fixture
def adapter_with_retry(mock_client: MagicMock) -> HttpNrcAdamsClient:
    return HttpNrcAdamsClient._for_test(client=mock_client, retry=FAST_RETRY)


class TestSearchDocuments:
    def test_yields_results_from_single_page(
        self, adapter: HttpNrcAdamsClient, mock_client: MagicMock
    ) -> None:
        page = _make_search_page([_make_raw_result(f"ML0000{i}") for i in range(3)])
        mock_client.request.return_value = _make_response(json_data=page)

        results = list(adapter.search_documents("safety valve"))
        assert len(results) == 3
        assert results[0].accession_number == "ML00000"

    def test_paginates_until_partial_page(
        self, mock_client: MagicMock
    ) -> None:
        client = HttpNrcAdamsClient._for_test(client=mock_client, page_size=2, retry=NO_RETRY)

        full_page = _make_search_page([_make_raw_result(f"ML0000{i}") for i in range(2)])
        partial_page = _make_search_page([_make_raw_result("ML00010")])
        mock_client.request.side_effect = [
            _make_response(json_data=full_page),
            _make_response(json_data=partial_page),
        ]

        results = list(client.search_documents())
        assert len(results) == 3
        assert mock_client.request.call_count == 2

    def test_stops_on_empty_page(
        self, mock_client: MagicMock
    ) -> None:
        client = HttpNrcAdamsClient._for_test(client=mock_client, page_size=2, retry=NO_RETRY)

        full_page = _make_search_page([_make_raw_result(f"ML0000{i}") for i in range(2)])
        empty_page = _make_search_page([])
        mock_client.request.side_effect = [
            _make_response(json_data=full_page),
            _make_response(json_data=empty_page),
        ]

        results = list(client.search_documents())
        assert len(results) == 2

    def test_passes_filters_in_body(
        self, adapter: HttpNrcAdamsClient, mock_client: MagicMock
    ) -> None:
        mock_client.request.return_value = _make_response(json_data={"results": []})
        filters = [{"field": "DocumentType", "value": "Inspection Report"}]

        list(adapter.search_documents(query="test", filters=filters))

        call_kwargs = mock_client.request.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["filters"] == filters
        assert body["q"] == "test"

    def test_raises_auth_error_on_401(
        self, adapter: HttpNrcAdamsClient, mock_client: MagicMock
    ) -> None:
        mock_client.request.return_value = _make_response(status_code=401)

        with pytest.raises(AdamsAuthError):
            list(adapter.search_documents())

    def test_raises_auth_error_on_403(
        self, adapter: HttpNrcAdamsClient, mock_client: MagicMock
    ) -> None:
        mock_client.request.return_value = _make_response(status_code=403)

        with pytest.raises(AdamsAuthError):
            list(adapter.search_documents())


class TestSearchDocumentsRetry:
    @patch("rag.adapters.ingestion.case.http_client.time.sleep")
    def test_retries_on_500_then_succeeds(
        self,
        mock_sleep: MagicMock,
        adapter_with_retry: HttpNrcAdamsClient,
        mock_client: MagicMock,
    ) -> None:
        page = _make_search_page([_make_raw_result()])
        mock_client.request.side_effect = [
            _make_response(status_code=500),
            _make_response(json_data=page),
        ]

        results = list(adapter_with_retry.search_documents())
        assert len(results) == 1
        mock_sleep.assert_called_once()

    @patch("rag.adapters.ingestion.case.http_client.time.sleep")
    def test_raises_after_max_retries_exhausted(
        self,
        mock_sleep: MagicMock,
        adapter_with_retry: HttpNrcAdamsClient,
        mock_client: MagicMock,
    ) -> None:
        mock_client.request.return_value = _make_response(status_code=500)

        with pytest.raises(AdamsApiError):
            list(adapter_with_retry.search_documents())

        # 3 attempts total: initial + 2 retries
        assert mock_client.request.call_count == 3

    @patch("rag.adapters.ingestion.case.http_client.time.sleep")
    def test_raises_rate_limit_on_429_exhausted(
        self,
        mock_sleep: MagicMock,
        adapter_with_retry: HttpNrcAdamsClient,
        mock_client: MagicMock,
    ) -> None:
        mock_client.request.return_value = _make_response(status_code=429)

        with pytest.raises(AdamsRateLimitError):
            list(adapter_with_retry.search_documents())

    @patch("rag.adapters.ingestion.case.http_client.time.sleep")
    def test_respects_retry_after_header(
        self,
        mock_sleep: MagicMock,
        adapter_with_retry: HttpNrcAdamsClient,
        mock_client: MagicMock,
    ) -> None:
        page = _make_search_page([_make_raw_result()])
        mock_client.request.side_effect = [
            _make_response(status_code=429, headers={"Retry-After": "7"}),
            _make_response(json_data=page),
        ]

        results = list(adapter_with_retry.search_documents())
        assert len(results) == 1
        mock_sleep.assert_any_call(7.0)


class TestGetDocument:
    def test_returns_document_on_success(
        self, adapter: HttpNrcAdamsClient, mock_client: MagicMock
    ) -> None:
        raw = _make_raw_document("ML24001A001")
        mock_client.request.return_value = _make_response(json_data=raw)

        doc = adapter.get_document("ML24001A001")
        assert doc is not None
        assert doc.accession_number == "ML24001A001"
        assert doc.content == "Full text of the document."
        assert doc.license_numbers == ["NPF-1", "NPF-2"]
        assert doc.keywords == ["safety", "inspection"]

    def test_returns_none_on_404(
        self, adapter: HttpNrcAdamsClient, mock_client: MagicMock
    ) -> None:
        mock_client.request.return_value = _make_response(status_code=404)

        doc = adapter.get_document("ML00000XXXX")
        assert doc is None

    def test_raises_auth_error_on_401(
        self, adapter: HttpNrcAdamsClient, mock_client: MagicMock
    ) -> None:
        mock_client.request.return_value = _make_response(status_code=401)

        with pytest.raises(AdamsAuthError):
            adapter.get_document("ML24001A001")


class TestRetryBackoff:
    def test_backoff_increases_exponentially(self) -> None:
        cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        client = HttpNrcAdamsClient._for_test(client=MagicMock(), retry=cfg)

        delays = [client._backoff_delay(i) for i in range(4)]
        assert delays == [1.0, 2.0, 4.0, 8.0]

    def test_backoff_capped_at_max(self) -> None:
        cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=5.0, jitter=False)
        client = HttpNrcAdamsClient._for_test(client=MagicMock(), retry=cfg)

        assert client._backoff_delay(10) == 5.0

    def test_jitter_produces_variation(self) -> None:
        cfg = RetryConfig(base_delay=10.0, jitter=True)
        client = HttpNrcAdamsClient._for_test(client=MagicMock(), retry=cfg)

        delays = {client._backoff_delay(0) for _ in range(20)}
        # With jitter, we should get multiple distinct values
        assert len(delays) > 1


class TestParseHelpers:
    def test_parse_search_result_extracts_fields(self) -> None:
        raw = _make_raw_result("ML24001A001", "Safety Valve Test")
        result = _parse_search_result(raw)

        assert result.accession_number == "ML24001A001"
        assert result.title == "Safety Valve Test"
        assert result.document_type == "Inspection Report"
        assert result.document_date == "2024-01-15"
        assert result.estimated_pages == 10

    def test_parse_search_result_handles_missing_fields(self) -> None:
        raw = {"AccessionNumber": "ML00001", "DocumentTitle": "Minimal"}
        result = _parse_search_result(raw)

        assert result.accession_number == "ML00001"
        assert result.document_date is None
        assert result.url is None

    def test_parse_document_extracts_content(self) -> None:
        raw = _make_raw_document()
        doc = _parse_document(raw)

        assert doc.content == "Full text of the document."
        assert doc.addressee == "Jane Smith"

    def test_parse_document_handles_missing_optional_fields(self) -> None:
        raw = {
            "AccessionNumber": "ML00001",
            "DocumentTitle": "Minimal Doc",
            "DocumentType": "Letter",
        }
        doc = _parse_document(raw)

        assert doc.accession_number == "ML00001"
        assert doc.content is None
        assert doc.docket_numbers == []
        assert doc.keywords == []

    def test_parse_document_splits_semicolon_lists(self) -> None:
        raw = {
            "AccessionNumber": "ML00001",
            "DocumentTitle": "Multi-Docket",
            "DocumentType": "Letter",
            "DocketNumber": "05000001; 05000002; 05000003",
        }
        doc = _parse_document(raw)

        assert doc.docket_numbers == ["05000001", "05000002", "05000003"]
