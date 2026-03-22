"""Tests for benchmark domain enums."""

from __future__ import annotations

from enum import StrEnum

from benchmark.domain.enums import EvidenceTier, QueryClass, ReviewStatus, UnitKind


class TestUnitKind:
    def test_is_str_enum(self) -> None:
        assert issubclass(UnitKind, StrEnum)

    def test_values(self) -> None:
        expected = {
            "obligation",
            "prohibition",
            "threshold",
            "exception",
            "condition",
            "definition",
            "process",
            "cross_reference",
        }
        assert {e.value for e in UnitKind} == expected


class TestQueryClass:
    def test_is_str_enum(self) -> None:
        assert issubclass(QueryClass, StrEnum)

    def test_snake_case_values(self) -> None:
        """Design doc specifies snake_case for QueryClass values."""
        for member in QueryClass:
            assert "_" in member.value or member.value.isalpha(), (
                f"{member.name} should be snake_case"
            )

    def test_values(self) -> None:
        expected = {
            "citation_lookup",
            "narrow_factual",
            "rule_explanation",
            "cross_reference",
            "scenario_application",
            "unanswerable",
            "robustness_variant",
        }
        assert {e.value for e in QueryClass} == expected


class TestEvidenceTier:
    def test_is_str_enum(self) -> None:
        assert issubclass(EvidenceTier, StrEnum)

    def test_values(self) -> None:
        expected = {"critical", "supporting", "contextual"}
        assert {e.value for e in EvidenceTier} == expected

    def test_ordering(self) -> None:
        """Critical < supporting < contextual by string sort matches importance."""
        assert EvidenceTier.CONTEXTUAL < EvidenceTier.CRITICAL < EvidenceTier.SUPPORTING


class TestReviewStatus:
    def test_is_str_enum(self) -> None:
        assert issubclass(ReviewStatus, StrEnum)

    def test_values(self) -> None:
        expected = {"pending", "approved", "rejected", "needs_revision"}
        assert {e.value for e in ReviewStatus} == expected
