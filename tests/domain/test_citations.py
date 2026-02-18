"""Tests for the CitationSpan domain model."""

from __future__ import annotations

from rag.domain.citations import CitationSpan


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
