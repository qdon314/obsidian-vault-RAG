"""Tests for M5 domain model additions.

Covers:
- GoldAnswer dataclass construction and immutability
- BenchmarkRecord M5 field defaults and round-trip through dataclasses.asdict()
- Backward compat: existing BenchmarkRecord construction without M5 fields
"""

from __future__ import annotations

import dataclasses

from benchmark.domain.enums import EvidenceTier, QueryClass
from benchmark.domain.models import (
    BenchmarkRecord,
    EvidenceEntry,
    GoldAnswer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_record() -> BenchmarkRecord:
    """BenchmarkRecord with all required fields; M5 fields left at defaults."""
    entry = EvidenceEntry(
        span_id="span_1",
        citation="10 CFR 50.46(b)(1)",
        text="The peak cladding temperature shall not exceed 2200°F.",
        char_start=0,
        char_end=55,
        chunk_ids=("chunk_1",),
        tier=EvidenceTier.CRITICAL,
    )
    return BenchmarkRecord(
        qid="reg_fact_000001",
        query="What is the peak cladding temperature limit?",
        query_class=QueryClass.CITATION_LOOKUP,
        difficulty="easy",
        source_unit_ids=("50.46_b_1",),
        source_citations=("10 CFR 50.46(b)(1)",),
        critical_evidence=(entry,),
        supporting_evidence=(),
        contextual_evidence=(),
        hard_negative_chunk_ids=("chunk_99",),
        hard_negatives_retriever_config={"model": "text-embedding-3-small", "top_k": 20},
        corpus_snapshot_id="abc123",
        valid_as_of="2026-03-24",
    )


# ---------------------------------------------------------------------------
# GoldAnswer tests
# ---------------------------------------------------------------------------


class TestGoldAnswer:
    def test_required_field_only(self) -> None:
        ga = GoldAnswer(gold_answer="The limit is 2200°F.")
        assert ga.gold_answer == "The limit is 2200°F."
        assert ga.acceptable_answer_variants == ()
        assert ga.required_points == ()
        assert ga.forbidden_errors == ()

    def test_all_fields(self) -> None:
        ga = GoldAnswer(
            gold_answer="The limit is 2200°F.",
            acceptable_answer_variants=("2200 degrees Fahrenheit",),
            required_points=("Contains 2200°F threshold",),
            forbidden_errors=("Wrong threshold value",),
        )
        assert ga.acceptable_answer_variants == ("2200 degrees Fahrenheit",)
        assert ga.required_points == ("Contains 2200°F threshold",)
        assert ga.forbidden_errors == ("Wrong threshold value",)

    def test_frozen(self) -> None:
        ga = GoldAnswer(gold_answer="The limit is 2200°F.")
        try:
            ga.gold_answer = "changed"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except dataclasses.FrozenInstanceError:
            pass

    def test_replace(self) -> None:
        ga = GoldAnswer(gold_answer="original")
        updated = dataclasses.replace(ga, gold_answer="updated")
        assert updated.gold_answer == "updated"
        assert ga.gold_answer == "original"

    def test_asdict_round_trip(self) -> None:
        ga = GoldAnswer(
            gold_answer="The limit is 2200°F.",
            acceptable_answer_variants=("variant",),
            required_points=("point",),
            forbidden_errors=("error",),
        )
        d = dataclasses.asdict(ga)
        assert d["gold_answer"] == "The limit is 2200°F."
        assert d["acceptable_answer_variants"] == ("variant",)
        assert d["required_points"] == ("point",)
        assert d["forbidden_errors"] == ("error",)


# ---------------------------------------------------------------------------
# BenchmarkRecord M5 field tests
# ---------------------------------------------------------------------------


class TestBenchmarkRecordM5Fields:
    def test_defaults_are_none_and_empty(self) -> None:
        record = _minimal_record()
        assert record.gold_answer is None
        assert record.acceptable_answer_variants == ()
        assert record.required_points == ()
        assert record.forbidden_errors == ()
        assert record.contamination_probes == {}

    def test_gold_answer_populated(self) -> None:
        record = dataclasses.replace(
            _minimal_record(),
            gold_answer="The limit is 2200°F.",
            acceptable_answer_variants=("2200 degrees Fahrenheit",),
            required_points=("Contains threshold",),
            forbidden_errors=("Wrong value",),
        )
        assert record.gold_answer == "The limit is 2200°F."
        assert record.acceptable_answer_variants == ("2200 degrees Fahrenheit",)
        assert record.required_points == ("Contains threshold",)
        assert record.forbidden_errors == ("Wrong value",)

    def test_contamination_probes_populated(self) -> None:
        record = dataclasses.replace(
            _minimal_record(),
            gold_answer="The limit is 2200°F.",
            contamination_probes={"gpt-4o-2025-01-01": False, "claude-sonnet-4": True},
        )
        assert record.contamination_probes["gpt-4o-2025-01-01"] is False
        assert record.contamination_probes["claude-sonnet-4"] is True

    def test_contamination_probes_idempotent_update(self) -> None:
        """Adding a new model probe preserves existing ones."""
        record = dataclasses.replace(
            _minimal_record(),
            contamination_probes={"model-a": False},
        )
        updated = dataclasses.replace(
            record,
            contamination_probes={**record.contamination_probes, "model-b": True},
        )
        assert updated.contamination_probes == {"model-a": False, "model-b": True}

    def test_asdict_round_trip(self) -> None:
        record = dataclasses.replace(
            _minimal_record(),
            gold_answer="The limit is 2200°F.",
            acceptable_answer_variants=("variant",),
            required_points=("point",),
            forbidden_errors=("error",),
            contamination_probes={"gpt-4o": False},
        )
        d = dataclasses.asdict(record)
        assert d["gold_answer"] == "The limit is 2200°F."
        assert d["acceptable_answer_variants"] == ("variant",)
        assert d["required_points"] == ("point",)
        assert d["forbidden_errors"] == ("error",)
        assert d["contamination_probes"] == {"gpt-4o": False}

    def test_backward_compat_no_m5_kwargs(self) -> None:
        """Records constructed without M5 kwargs work unchanged (backward compat)."""
        record = _minimal_record()
        assert record.qid == "reg_fact_000001"
        assert record.gold_answer is None
        assert record.contamination_probes == {}

    def test_frozen(self) -> None:
        record = _minimal_record()
        try:
            record.gold_answer = "changed"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except dataclasses.FrozenInstanceError:
            pass
