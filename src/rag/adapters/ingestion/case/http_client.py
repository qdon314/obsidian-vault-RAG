"""HTTP implementation of the NrcAdamsClient port."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from rag.domain.errors import (
    AdamsApiError,
    AdamsAuthError,
    AdamsNotFoundError,
    AdamsRateLimitError,
)
from rag.ports.nrc_adams_client import AdamsDocument, AdamsSearchResult

logger = logging.getLogger(__name__)


# ------------------------------------
# Configuration
# ------------------------------------


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for HTTP retry behaviour."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )


# ------------------------------------
# Adapter
# ------------------------------------


@dataclass(frozen=True, slots=True)
class HttpNrcAdamsClient:
    """HTTP adapter for the NRC ADAMS Public Search API.

    Parameters
    ----------
    subscription_key:
        ADAMS API subscription key (``Ocp-Apim-Subscription-Key`` header).
    base_url:
        API base URL.
    timeout:
        Request timeout in seconds.
    page_size:
        Maximum results per search page (API limit is 100).
    retry:
        Retry configuration for transient failures.
    """

    subscription_key: str
    base_url: str = "https://adams-api.nrc.gov/aps/api"
    timeout: float = 30.0
    page_size: int = 100
    retry: RetryConfig = field(default_factory=RetryConfig)
    _client: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        import httpx  # type: ignore[import-untyped]

        object.__setattr__(
            self,
            "_client",
            httpx.Client(
                base_url=self.base_url,
                headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
                timeout=self.timeout,
            ),
        )

    @classmethod
    def _for_test(
        cls,
        *,
        subscription_key: str = "test-key",
        base_url: str = "https://adams-api.nrc.gov/aps/api",
        timeout: float = 5.0,
        page_size: int = 100,
        retry: RetryConfig | None = None,
        client: Any,
    ) -> HttpNrcAdamsClient:
        """Create an instance with an injected HTTP client (for testing)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "subscription_key", subscription_key)
        object.__setattr__(obj, "base_url", base_url)
        object.__setattr__(obj, "timeout", timeout)
        object.__setattr__(obj, "page_size", page_size)
        object.__setattr__(obj, "retry", retry or RetryConfig())
        object.__setattr__(obj, "_client", client)
        return obj

    # ---- public methods ----

    def search_documents(
        self,
        query: str = "",
        *,
        filters: list[dict[str, object]] | None = None,
        any_filters: list[dict[str, object]] | None = None,
        sort_by: str = "DocumentDate",
        sort_desc: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[AdamsSearchResult]:
        """Search for documents with automatic pagination."""
        skip = 0
        while True:
            body: dict[str, object] = {
                "q": query,
                "filters": filters or [],
                "anyFilters": any_filters or [],
                "legacyLibFilter": True,
                "mainLibFilter": True,
                "sort": sort_by,
                "sortDirection": 0 if sort_desc else 1,
                "skip": skip,
            }
            data = self._post("/search", body)
            results = data.get("results", [])
            if not isinstance(results, list):
                results = []

            for raw in results:
                yield _parse_search_result(raw)

            if len(results) < self.page_size:
                break
            skip += len(results)

    def get_document(
        self,
        accession_number: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> AdamsDocument | None:
        """Fetch a single document by accession number."""
        try:
            data = self._get(f"/search/{accession_number}")
        except AdamsNotFoundError:
            return None
        return _parse_document(data)

    # ---- private HTTP helpers ----

    def _post(self, endpoint: str, body: dict[str, object]) -> dict[str, Any]:
        return self._request("POST", endpoint, json=body)

    def _get(self, endpoint: str) -> dict[str, Any]:
        return self._request("GET", endpoint)

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        import httpx

        last_exc: Exception | None = None
        attempts = self.retry.max_retries + 1

        for attempt in range(attempts):
            if attempt > 0:
                delay = self._backoff_delay(attempt - 1)
                logger.info(
                    "Retrying %s %s (attempt %d/%d, delay=%.1fs)",
                    method,
                    endpoint,
                    attempt + 1,
                    attempts,
                    delay,
                )
                time.sleep(delay)

            try:
                response = self._client.request(method, endpoint, **kwargs)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < self.retry.max_retries:
                    continue
                raise AdamsApiError(
                    f"{method} {endpoint} failed after {attempts} attempts: {exc}"
                ) from exc

            status = response.status_code

            if status == 200:
                return response.json()  # type: ignore[no-any-return]

            if status in (401, 403):
                raise AdamsAuthError(
                    f"{method} {endpoint} returned {status}: {response.text}"
                )

            if status == 404:
                raise AdamsNotFoundError(
                    f"{method} {endpoint} returned 404: {response.text}"
                )

            if status == 429:
                last_exc = AdamsRateLimitError(
                    f"{method} {endpoint} returned 429: {response.text}"
                )
                if attempt < self.retry.max_retries:
                    # Respect Retry-After header if present, otherwise backoff
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            time.sleep(float(retry_after))
                        except ValueError:
                            time.sleep(self._backoff_delay(attempt))
                    else:
                        time.sleep(self._backoff_delay(attempt))
                    continue
                raise last_exc

            if status in self.retry.retryable_statuses:
                last_exc = AdamsApiError(
                    f"{method} {endpoint} returned {status}: {response.text}"
                )
                if attempt < self.retry.max_retries:
                    continue
                raise last_exc

            # Non-retryable error
            raise AdamsApiError(
                f"{method} {endpoint} returned {status}: {response.text}"
            )

        # Should not reach here, but satisfy type checker
        raise AdamsApiError(  # pragma: no cover
            f"{method} {endpoint} failed after {attempts} attempts"
        )

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(
            self.retry.base_delay * (self.retry.exponential_base**attempt),
            self.retry.max_delay,
        )
        if self.retry.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay


# ------------------------------------
# Parse helpers (pure functions)
# ------------------------------------


def _parse_search_result(raw: dict[str, Any]) -> AdamsSearchResult:
    return AdamsSearchResult(
        accession_number=raw.get("AccessionNumber", ""),
        title=raw.get("DocumentTitle", ""),
        document_type=raw.get("DocumentType", ""),
        document_date=raw.get("DocumentDate"),
        author_name=raw.get("AuthorName"),
        author_affiliation=raw.get("AuthorAffiliation"),
        docket_number=raw.get("DocketNumber"),
        url=raw.get("Url"),
        estimated_pages=raw.get("EstimatedPages"),
    )


def _parse_document(raw: dict[str, Any]) -> AdamsDocument:
    return AdamsDocument(
        accession_number=raw.get("AccessionNumber", ""),
        title=raw.get("DocumentTitle", ""),
        document_type=raw.get("DocumentType", ""),
        document_date=raw.get("DocumentDate"),
        author_name=raw.get("AuthorName"),
        author_affiliation=raw.get("AuthorAffiliation"),
        addressee=raw.get("AddresseeName"),
        addressee_affiliation=raw.get("AddresseeAffiliation"),
        docket_numbers=raw.get("DocketNumber", "").split("; ") if raw.get("DocketNumber") else [],
        license_numbers=raw.get("LicenseNumber", "").split("; ") if raw.get("LicenseNumber") else [],
        keywords=raw.get("Keyword", "").split("; ") if raw.get("Keyword") else [],
        content=raw.get("content"),
        url=raw.get("Url"),
        estimated_pages=raw.get("EstimatedPages"),
    )
