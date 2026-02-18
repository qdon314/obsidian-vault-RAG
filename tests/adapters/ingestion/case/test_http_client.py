"""Tests for HTTP NRC ADAMS client adapter."""

from unittest.mock import MagicMock, Mock

import pytest

from rag.adapters.ingestion.case.http_client import HttpNrcAdamsClient, RetryConfig
from rag.domain.errors import (
    AdamsApiError,
    AdamsAuthError,
    AdamsNotFoundError,
    AdamsRateLimitError,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_httpx():
    """Mock httpx module and Client."""
    import sys
    from unittest.mock import MagicMock

    # Create mock httpx module
    mock_httpx_module = MagicMock()
    mock_httpx_module.Client = MagicMock
    mock_httpx_module.TimeoutException = type("TimeoutException", (Exception,), {})
    mock_httpx_module.NetworkError = type("NetworkError", (Exception,), {})

    # Inject into sys.modules so imports work
    sys.modules["httpx"] = mock_httpx_module

    yield mock_httpx_module

    # Cleanup
    if "httpx" in sys.modules:
        del sys.modules["httpx"]


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing."""
    return MagicMock()


# ============================================================
# Client Creation Tests
# ============================================================


class TestHttpNrcAdamsClientCreation:
    """Test client initialization and configuration."""

    def test_for_test_factory(self, mock_http_client):
        """Should create client with injected httpx client for testing."""
        client = HttpNrcAdamsClient._for_test(
            subscription_key="test-key",
            base_url="https://test.example.com",
            timeout=10.0,
            page_size=50,
            client=mock_http_client,
        )

        assert client.subscription_key == "test-key"
        assert client.base_url == "https://test.example.com"
        assert client.timeout == 10.0
        assert client.page_size == 50
        assert client._client is mock_http_client

    def test_default_retry_config(self, mock_http_client):
        """Should use default retry configuration."""
        client = HttpNrcAdamsClient._for_test(client=mock_http_client)

        assert isinstance(client.retry, RetryConfig)
        assert client.retry.max_retries == 3
        assert client.retry.base_delay == 1.0


# ============================================================
# Search Tests
# ============================================================


class TestSearchDocuments:
    """Test document search endpoint."""

    def test_search_single_page(self, mock_http_client):
        """Should return results from single page search."""
        mock_http_client.request.return_value = Mock(
            status_code=200,
            json=lambda: {
                "results": [
                    {
                        "AccessionNumber": "ML12345A678",
                        "DocumentTitle": "Test Document 1",
                        "DocumentType": "Inspection Report",
                        "DocumentDate": "2024-01-15",
                    },
                    {
                        "AccessionNumber": "ML23456B789",
                        "DocumentTitle": "Test Document 2",
                        "DocumentType": "Notice of Violation",
                        "DocumentDate": "2024-02-20",
                    },
                ]
            },
        )

        client = HttpNrcAdamsClient._for_test(
            subscription_key="test-key",
            page_size=100,
            client=mock_http_client,
        )

        results = list(client.search_documents("test query"))

        assert len(results) == 2
        assert results[0].accession_number == "ML12345A678"
        assert results[0].title == "Test Document 1"
        assert results[1].accession_number == "ML23456B789"
        assert results[1].title == "Test Document 2"

    def test_search_multiple_pages(self, mock_http_client):
        """Should paginate automatically through multiple result pages."""
        # First page: full page (2 results with page_size=2)
        # Second page: partial page (1 result)
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Mock(
                    status_code=200,
                    json=lambda: {
                        "results": [
                            {
                                "AccessionNumber": "ML11111A111",
                                "DocumentTitle": "Doc 1",
                                "DocumentType": "Test",
                            },
                            {
                                "AccessionNumber": "ML22222B222",
                                "DocumentTitle": "Doc 2",
                                "DocumentType": "Test",
                            },
                        ]
                    },
                )
            else:  # call_count == 2
                return Mock(
                    status_code=200,
                    json=lambda: {
                        "results": [
                            {
                                "AccessionNumber": "ML33333C333",
                                "DocumentTitle": "Doc 3",
                                "DocumentType": "Test",
                            }
                        ]
                    },
                )

        mock_http_client.request.side_effect = side_effect

        client = HttpNrcAdamsClient._for_test(
            page_size=2,  # Force pagination
            client=mock_http_client,
        )

        results = list(client.search_documents("test"))

        assert len(results) == 3
        assert results[0].accession_number == "ML11111A111"
        assert results[1].accession_number == "ML22222B222"
        assert results[2].accession_number == "ML33333C333"
        assert call_count == 2  # Two API calls

    def test_search_with_filters(self, mock_http_client):
        """Should pass filters to API endpoint."""
        mock_http_client.request.return_value = Mock(
            status_code=200,
            json=lambda: {
                "results": [
                    {
                        "document": {
                            "AccessionNumber": "TEST-001",
                            "DocumentTitle": "Test Document",
                        }
                    }
                ]
            },
        )

        client = HttpNrcAdamsClient._for_test(client=mock_http_client)

        filters: list[dict[str, object]] = [{"field": "DocumentType", "value": "Inspection Report"}]
        any_filters: list[dict[str, object]] = [{"field": "DocketNumber", "value": "DOCKET-123"}]

        list(
            client.search_documents(
                "test",
                filters=filters,
                any_filters=any_filters,
                sort_by="DocumentDate",
                sort_desc=False,
            )
        )

        # Verify the request was made with correct body
        call_args = mock_http_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "json" in call_args[1]
        body = call_args[1]["json"]
        assert body["q"] == "test"
        assert body["filters"] == filters
        assert body["anyFilters"] == any_filters
        assert body["sort"] == "DocumentDate"
        assert body["sortDirection"] == 1  # not desc

    def test_search_empty_results(self, mock_http_client):
        """Should handle empty search results."""
        mock_http_client.request.return_value = Mock(
            status_code=200,
            json=lambda: {"results": []},
        )

        client = HttpNrcAdamsClient._for_test(client=mock_http_client)

        results = list(client.search_documents("nonexistent"))

        assert len(results) == 0

    def test_search_malformed_results(self, mock_http_client):
        """Should handle malformed API response gracefully."""
        mock_http_client.request.return_value = Mock(
            status_code=200,
            json=lambda: {},  # Missing 'results' key
        )

        client = HttpNrcAdamsClient._for_test(client=mock_http_client)

        results = list(client.search_documents("test"))

        assert len(results) == 0  # Should not crash


# ============================================================
# Get Document Tests
# ============================================================


class TestGetDocument:
    """Test single document fetch endpoint."""

    def test_get_document_success(self, mock_http_client):
        """Should fetch and parse a single document."""
        mock_http_client.request.return_value = Mock(
            status_code=200,
            json=lambda: {
                "AccessionNumber": "ML12345A678",
                "DocumentTitle": "Test Document",
                "DocumentType": "Inspection Report",
                "DocumentDate": "2024-01-15",
                "AuthorName": "John Doe",
                "AuthorAffiliation": "NRC",
                "AddresseeName": "Jane Smith",
                "AddresseeAffiliation": "Utility",
                "DocketNumber": "DOCKET-123; DOCKET-456",
                "LicenseNumber": "LICENSE-001",
                "Keyword": "inspection; reactor",
                "content": "Document content here...",
                "Url": "https://www.nrc.gov/docs/ML1234/ML12345A678.html",
                "EstimatedPages": 10,
            },
        )

        client = HttpNrcAdamsClient._for_test(client=mock_http_client)

        doc = client.get_document("ML12345A678")

        assert doc is not None
        assert doc.accession_number == "ML12345A678"
        assert doc.title == "Test Document"
        assert doc.document_type == "Inspection Report"
        assert doc.document_date == "2024-01-15"
        assert doc.author_name == "John Doe"
        assert doc.addressee == "Jane Smith"
        assert doc.docket_numbers == ["DOCKET-123", "DOCKET-456"]
        assert doc.license_numbers == ["LICENSE-001"]
        assert doc.keywords == ["inspection", "reactor"]
        assert doc.content == "Document content here..."
        assert doc.url == "https://www.nrc.gov/docs/ML1234/ML12345A678.html"
        assert doc.estimated_pages == 10

    def test_get_document_not_found(self, mock_http_client):
        """Should return None when document not found."""

        mock_http_client.request.side_effect = AdamsNotFoundError("Not found")

        client = HttpNrcAdamsClient._for_test(client=mock_http_client)

        doc = client.get_document("ML99999Z999")

        assert doc is None

    def test_get_document_minimal_fields(self, mock_http_client):
        """Should handle documents with minimal fields."""
        mock_http_client.request.return_value = Mock(
            status_code=200,
            json=lambda: {
                "AccessionNumber": "ML12345A678",
                "DocumentTitle": "Minimal Doc",
                "DocumentType": "Test",
            },
        )

        client = HttpNrcAdamsClient._for_test(client=mock_http_client)

        doc = client.get_document("ML12345A678")

        assert doc is not None
        assert doc.accession_number == "ML12345A678"
        assert doc.title == "Minimal Doc"
        assert doc.document_type == "Test"
        assert doc.document_date is None
        assert doc.docket_numbers == []
        assert doc.license_numbers == []
        assert doc.keywords == []

    def test_get_document_splits_semicolon_lists(self, mock_http_client):
        """Should split semicolon-delimited lists from API."""
        mock_http_client.request.return_value = Mock(
            status_code=200,
            json=lambda: {
                "AccessionNumber": "ML12345A678",
                "DocumentTitle": "Test",
                "DocumentType": "Test",
                "DocketNumber": "D1; D2; D3",
                "LicenseNumber": "L1; L2",
                "Keyword": "k1; k2; k3; k4",
            },
        )

        client = HttpNrcAdamsClient._for_test(client=mock_http_client)

        doc = client.get_document("ML12345A678")

        assert doc is not None
        assert doc.docket_numbers == ["D1", "D2", "D3"]
        assert doc.license_numbers == ["L1", "L2"]
        assert doc.keywords == ["k1", "k2", "k3", "k4"]


# ============================================================
# Error Handling Tests
# ============================================================


class TestErrorHandling:
    """Test error handling and retries."""

    def test_auth_error_401(self, mock_http_client):
        """Should raise AdamsAuthError on 401."""
        mock_http_client.request.return_value = Mock(
            status_code=401,
            text="Unauthorized",
        )

        client = HttpNrcAdamsClient._for_test(
            retry=RetryConfig(max_retries=0),
            client=mock_http_client,
        )

        with pytest.raises(AdamsAuthError, match="401"):
            client.get_document("ML12345A678")

    def test_auth_error_403(self, mock_http_client):
        """Should raise AdamsAuthError on 403."""
        mock_http_client.request.return_value = Mock(
            status_code=403,
            text="Forbidden",
        )

        client = HttpNrcAdamsClient._for_test(
            retry=RetryConfig(max_retries=0),
            client=mock_http_client,
        )

        with pytest.raises(AdamsAuthError, match="403"):
            client.get_document("ML12345A678")

    def test_not_found_404(self, mock_http_client):
        """Should raise AdamsNotFoundError on 404."""
        mock_http_client.request.return_value = Mock(
            status_code=404,
            text="Not Found",
        )

        client = HttpNrcAdamsClient._for_test(
            retry=RetryConfig(max_retries=0),
            client=mock_http_client,
        )

        with pytest.raises(AdamsNotFoundError, match="404"):
            list(client.search_documents("test"))

    def test_rate_limit_429_no_retry(self, mock_http_client):
        """Should raise AdamsRateLimitError when retries exhausted."""
        mock_http_client.request.return_value = Mock(
            status_code=429,
            text="Rate limit exceeded",
            headers={},
        )

        client = HttpNrcAdamsClient._for_test(
            retry=RetryConfig(max_retries=0),
            client=mock_http_client,
        )

        with pytest.raises(AdamsRateLimitError, match="429"):
            list(client.search_documents("test"))

    def test_rate_limit_429_with_retry(self, mock_http_client):
        """Should retry on 429 and succeed eventually."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return Mock(status_code=429, text="Rate limit", headers={})
            else:
                return Mock(status_code=200, json=lambda: {"results": []})

        mock_http_client.request.side_effect = side_effect

        client = HttpNrcAdamsClient._for_test(
            retry=RetryConfig(max_retries=3, base_delay=0.01),
            client=mock_http_client,
        )

        results = list(client.search_documents("test"))

        assert call_count == 3
        assert results == []

    def test_server_error_500_with_retry(self, mock_http_client):
        """Should retry on 500 server error."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Mock(status_code=500, text="Internal Server Error")
            else:
                return Mock(status_code=200, json=lambda: {"results": []})

        mock_http_client.request.side_effect = side_effect

        client = HttpNrcAdamsClient._for_test(
            retry=RetryConfig(max_retries=2, base_delay=0.01),
            client=mock_http_client,
        )

        results = list(client.search_documents("test"))

        assert call_count == 2
        assert results == []

    def test_timeout_with_retry(self, mock_http_client):
        """Should retry on timeout exception."""
        import httpx

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("Request timeout")
            else:
                return Mock(status_code=200, json=lambda: {"results": []})

        mock_http_client.request.side_effect = side_effect

        client = HttpNrcAdamsClient._for_test(
            retry=RetryConfig(max_retries=2, base_delay=0.01),
            client=mock_http_client,
        )

        results = list(client.search_documents("test"))

        assert call_count == 2
        assert results == []

    def test_network_error_with_retry(self, mock_http_client):
        """Should retry on network error."""
        import httpx

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.NetworkError("Connection failed")
            else:
                return Mock(
                    status_code=200,
                    json=lambda: {
                        "results": [
                            {
                                "document": {
                                    "AccessionNumber": "TEST-001",
                                    "DocumentTitle": "Test Document",
                                }
                            }
                        ]
                    },
                )

        mock_http_client.request.side_effect = side_effect

        client = HttpNrcAdamsClient._for_test(
            retry=RetryConfig(max_retries=2, base_delay=0.01),
            client=mock_http_client,
        )

        results = list(client.search_documents("test"))

        assert results
        assert call_count == 2

    def test_non_retryable_error(self, mock_http_client):
        """Should not retry non-retryable status codes."""
        mock_http_client.request.return_value = Mock(
            status_code=400,
            text="Bad Request",
        )

        client = HttpNrcAdamsClient._for_test(
            retry=RetryConfig(max_retries=3),
            client=mock_http_client,
        )

        with pytest.raises(AdamsApiError, match="400"):
            list(client.search_documents("test"))

        # Should only call once (no retries)
        assert mock_http_client.request.call_count == 1


# ============================================================
# Retry Configuration Tests
# ============================================================


class TestRetryBehavior:
    """Test retry backoff and configuration."""

    def test_exponential_backoff_delay(self):
        """Should calculate exponential backoff delays."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False,
        )

        client = HttpNrcAdamsClient._for_test(
            retry=config,
            client=MagicMock(),
        )

        assert client._backoff_delay(0) == 1.0  # 1 * 2^0
        assert client._backoff_delay(1) == 2.0  # 1 * 2^1
        assert client._backoff_delay(2) == 4.0  # 1 * 2^2
        assert client._backoff_delay(3) == 8.0  # 1 * 2^3
        assert client._backoff_delay(4) == 10.0  # capped at max_delay

    def test_backoff_with_jitter(self):
        """Should add jitter to backoff delays."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=True,
        )

        client = HttpNrcAdamsClient._for_test(
            retry=config,
            client=MagicMock(),
        )

        # With jitter, delays should vary but be in expected range
        delays = [client._backoff_delay(1) for _ in range(10)]

        # All should be in range [1.0, 3.0] (2.0 * [0.5, 1.5])
        assert all(1.0 <= d <= 3.0 for d in delays)

        # Should not all be the same (very unlikely with jitter)
        assert len(set(delays)) > 1
