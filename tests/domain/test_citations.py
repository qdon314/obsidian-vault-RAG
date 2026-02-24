"""Tests for the CitationSpan domain model."""

from __future__ import annotations

import pytest

from rag.domain.citations import CitationSpan, normalize_citation_key


class TestCitationSpan:
    def test_frozen(self) -> None:
        span = CitationSpan(
            kind="cfr",
            raw="10 CFR 50.46",
            key="cfr:10:50.46",
            start=0,
            end=12,
            confidence=0.95,
            source_field="content",
        )
        assert span.kind == "cfr"
        assert span.key == "cfr:10:50.46"
        # Frozen — assignment raises
        try:
            span.kind = "other"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass

    def test_defaults(self) -> None:
        span = CitationSpan(
            kind="docket",
            raw="50-247",
            key="docket:50-247",
            start=10,
            end=16,
            confidence=0.85,
            source_field="content",
        )
        assert span.context is None
        assert span.attrs == {}

    def test_with_attrs(self) -> None:
        span = CitationSpan(
            kind="cfr",
            raw="10 CFR 50.46(b)(1)",
            key="cfr:10:50.46(b)(1)",
            start=0,
            end=18,
            confidence=0.95,
            source_field="content",
            attrs={"title": 10, "part": 50, "section": "46", "subsections": ["b", "1"]},
        )
        assert span.attrs["part"] == 50
        assert span.attrs["subsections"] == ["b", "1"]

    def test_equality_by_value(self) -> None:
        a = CitationSpan(
            kind="cfr",
            raw="10 CFR 50.46",
            key="cfr:10:50.46",
            start=0,
            end=12,
            confidence=0.95,
            source_field="content",
        )
        b = CitationSpan(
            kind="cfr",
            raw="10 CFR 50.46",
            key="cfr:10:50.46",
            start=0,
            end=12,
            confidence=0.95,
            source_field="content",
        )
        assert a == b

    def test_context_window(self) -> None:
        span = CitationSpan(
            kind="cfr",
            raw="10 CFR 50.46",
            key="cfr:10:50.46",
            start=100,
            end=112,
            confidence=0.95,
            source_field="content",
            context="...in accordance with 10 CFR 50.46 requirements...",
        )
        assert "in accordance with" in span.context  # type: ignore[operator]


class TestNormalizeCitationKey:
    """Tests for normalize_citation_key covering the design normalization table."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Already canonical — no-op
            ("10 CFR §50.36", "10 CFR §50.36"),
            # Missing § — inserted
            ("10 CFR 50.36", "10 CFR §50.36"),
            # Preserve subsection markers
            ("10 CFR §50.36(c)(2)", "10 CFR §50.36(c)(2)"),
            ("10 CFR 50.36(c)(2)", "10 CFR §50.36(c)(2)"),
            # Double § collapsed
            ("10 CFR §§50.36", "10 CFR §50.36"),
            # Title-number corruption
            ("0 CFR §50.55a", "10 CFR §50.55a"),
            ("0 CFR 50.55a", "10 CFR §50.55a"),
            # Whitespace normalization
            ("  10  CFR   §50.36  ", "10 CFR §50.36"),
            ("10  CFR  50.36", "10 CFR §50.36"),
            # Part references — passthrough (not section-level)
            ("10 CFR Part 50", "10 CFR Part 50"),
            # Appendix references — passthrough
            ("10 CFR 50 Appendix A", "10 CFR 50 Appendix A"),
            # Non-CFR strings — passthrough
            ("docket:50-247", "docket:50-247"),
            ("adams:ML021910673", "adams:ML021910673"),
            # Alphanumeric section suffix
            ("10 CFR 50.55a", "10 CFR §50.55a"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_citation_key(raw) == expected
