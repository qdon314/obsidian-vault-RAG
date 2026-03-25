"""Tests for Stage 5a deterministic query validation."""

from __future__ import annotations

from benchmark.adapters.validation.deterministic_validator import (
    DeterministicValidator,
)
from benchmark.domain.enums import QueryClass
from benchmark.domain.models import (
    EvidenceSet,
    QueryCandidate,
    ValidatedQuery,
)
from benchmark.ports.query_validator import QueryValidator

# ── Helpers ──────────────────────────────────────────────────────────


def _make_candidate(
    *,
    query: str = "What does 10 CFR 50.46 require regarding peak cladding temperature?",
    query_class: QueryClass = QueryClass.CITATION_LOOKUP,
    evidence_span_ids: tuple[str, ...] = ("u1_0",),
    corpus_snapshot_id: str = "snap1",
    candidate_id: str = "qc_1",
) -> QueryCandidate:
    return QueryCandidate(
        candidate_id=candidate_id,
        unit_id="50.46_b_1",
        query=query,
        query_class=query_class,
        source_citations=("10 CFR 50.46(b)(1)",),
        evidence_span_ids=evidence_span_ids,
        corpus_snapshot_id=corpus_snapshot_id,
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestProtocol:
    def test_satisfies_query_validator_protocol(self) -> None:
        val = DeterministicValidator()
        assert isinstance(val, QueryValidator)


class TestValidCandidate:
    def test_valid_candidate_passes(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate())
        assert result.is_valid is True
        assert result.flags == ()
        assert result.candidate_id == "qc_1"


class TestCitationFormat:
    def test_missing_cfr_citation(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(query="What is the temperature limit?"))
        assert not result.is_valid
        assert "no_cfr_citation" in result.flags

    def test_valid_cfr_citation_passes(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(query="What does 10 CFR 50.46 require?"))
        assert "no_cfr_citation" not in result.flags


class TestDuplicateDetection:
    def test_duplicate_query_flagged(self) -> None:
        val = DeterministicValidator(
            known_queries=["What does 10 CFR 50.46 require regarding peak cladding temperature?"]
        )
        result = val.validate(_make_candidate())
        assert not result.is_valid
        assert "duplicate_query" in result.flags

    def test_case_insensitive_dedup(self) -> None:
        val = DeterministicValidator(
            known_queries=["WHAT DOES 10 CFR 50.46 REQUIRE REGARDING PEAK CLADDING TEMPERATURE?"]
        )
        result = val.validate(_make_candidate())
        assert "duplicate_query" in result.flags

    def test_whitespace_normalized_dedup(self) -> None:
        val = DeterministicValidator(
            known_queries=[
                "What  does  10  CFR  50.46  require  regarding  peak  cladding  temperature?"
            ]
        )
        result = val.validate(_make_candidate())
        assert "duplicate_query" in result.flags

    def test_tracks_validated_queries_for_future_dedup(self) -> None:
        val = DeterministicValidator()
        val.validate(_make_candidate(candidate_id="qc_1"))
        result = val.validate(_make_candidate(candidate_id="qc_2"))
        assert "duplicate_query" in result.flags

    def test_unique_query_passes(self) -> None:
        val = DeterministicValidator(known_queries=["Some other question?"])
        result = val.validate(_make_candidate())
        assert "duplicate_query" not in result.flags


class TestBannedPrefix:
    def test_summarize_rejected(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(query="Summarize the requirements in 10 CFR 50.46"))
        assert "banned_prefix:summarize" in result.flags

    def test_list_all_rejected(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(query="List all requirements in 10 CFR 50.46"))
        assert "banned_prefix:list all" in result.flags

    def test_describe_everything_rejected(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(query="Describe everything about 10 CFR 50.46"))
        assert "banned_prefix:describe everything" in result.flags

    def test_what_are_all_the_rules_rejected(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(query="What are all the rules in 10 CFR 50.46?"))
        assert "banned_prefix:what are all the rules" in result.flags

    def test_case_insensitive_ban(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(query="SUMMARIZE the requirements in 10 CFR 50.46"))
        assert "banned_prefix:summarize" in result.flags


class TestLength:
    def test_too_short(self) -> None:
        val = DeterministicValidator(min_query_length=10)
        result = val.validate(_make_candidate(query="10 CFR?"))
        assert "too_short" in result.flags

    def test_too_long(self) -> None:
        val = DeterministicValidator(max_query_length=20)
        result = val.validate(
            _make_candidate(query="What does 10 CFR 50.46 require regarding the limit?")
        )
        assert "too_long" in result.flags

    def test_within_bounds_passes(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate())
        assert "too_short" not in result.flags
        assert "too_long" not in result.flags


class TestEvidence:
    def test_no_evidence_flagged(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(evidence_span_ids=()))
        assert "no_evidence" in result.flags

    def test_too_many_evidence_spans_for_narrow_class(self) -> None:
        val = DeterministicValidator(max_evidence_spans=2)
        result = val.validate(_make_candidate(evidence_span_ids=("s1", "s2", "s3")))
        assert "too_many_evidence_spans" in result.flags

    def test_evidence_size_not_checked_for_broad_classes(self) -> None:
        val = DeterministicValidator(max_evidence_spans=2)
        result = val.validate(
            _make_candidate(
                query="Explain the rule in 10 CFR 50.46?",
                query_class=QueryClass.RULE_EXPLANATION,
                evidence_span_ids=("s1", "s2", "s3", "s4"),
            )
        )
        assert "too_many_evidence_spans" not in result.flags


class TestSnapshotId:
    def test_missing_snapshot_flagged(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(corpus_snapshot_id=""))
        assert "missing_snapshot_id" in result.flags

    def test_present_snapshot_passes(self) -> None:
        val = DeterministicValidator()
        result = val.validate(_make_candidate(corpus_snapshot_id="snap1"))
        assert "missing_snapshot_id" not in result.flags


class TestMultipleFlags:
    def test_multiple_failing_checks_produce_multiple_flags(self) -> None:
        val = DeterministicValidator()
        result = val.validate(
            _make_candidate(
                query="Short",
                evidence_span_ids=(),
                corpus_snapshot_id="",
            )
        )
        assert not result.is_valid
        assert "no_cfr_citation" in result.flags
        assert "too_short" in result.flags
        assert "no_evidence" in result.flags
        assert "missing_snapshot_id" in result.flags
        assert len(result.flags) == 4


class TestRefineEvidence:
    def test_returns_evidence_unchanged(self) -> None:
        val = DeterministicValidator()
        evidence = EvidenceSet(unit_id="u1")
        vq = ValidatedQuery(
            candidate_id="qc_1",
            unit_id="u1",
            query="Test?",
            query_class=QueryClass.CITATION_LOOKUP,
            source_citations=(),
            evidence_span_ids=(),
            difficulty="easy",
            corpus_snapshot_id="snap1",
        )
        result = val.refine_evidence(vq, evidence)
        assert result is evidence  # identity — not a copy
