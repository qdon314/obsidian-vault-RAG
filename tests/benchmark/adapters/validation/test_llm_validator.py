"""Tests for the M4 LLM validator adapter."""

from __future__ import annotations

import json

from benchmark.adapters.validation.deterministic_validator import DeterministicValidator
from benchmark.adapters.validation.llm_validator import LLMValidator
from benchmark.domain.enums import EvidenceTier, QueryClass
from benchmark.domain.models import (
    EvidenceEntry,
    EvidenceSet,
    QueryCandidate,
    StageConfig,
    ValidatedQuery,
)
from benchmark.ports.llm_client import LLMResponse
from benchmark.ports.query_validator import QueryValidator

# -- Helpers ------------------------------------------------------------------


def _make_candidate(
    query: str = "What is the peak cladding temperature limit under 10 CFR 50.46(b)(1)?",
    evidence_ids: tuple[str, ...] = ("span_0",),
    query_class: QueryClass = QueryClass.CITATION_LOOKUP,
) -> QueryCandidate:
    return QueryCandidate(
        candidate_id="qc_test_0",
        unit_id="50.46_b_1",
        query=query,
        query_class=query_class,
        source_citations=("10 CFR 50.46(b)(1)",),
        evidence_span_ids=evidence_ids,
        corpus_snapshot_id="snap1",
    )


def _make_validated_query(
    query: str = "What is the peak cladding temperature limit under 10 CFR 50.46(b)(1)?",
    query_class: QueryClass = QueryClass.CITATION_LOOKUP,
) -> ValidatedQuery:
    return ValidatedQuery(
        candidate_id="qc_test_0",
        unit_id="50.46_b_1",
        query=query,
        query_class=query_class,
        source_citations=("10 CFR 50.46(b)(1)",),
        evidence_span_ids=("span_0",),
        difficulty="easy",
        corpus_snapshot_id="snap1",
    )


def _make_evidence(unit_id: str = "50.46_b_1") -> EvidenceSet:
    return EvidenceSet(
        unit_id=unit_id,
        critical=(
            EvidenceEntry(
                span_id="span_0",
                citation="10 CFR 50.46(b)(1)",
                text="Peak cladding temperature shall not exceed 2200F.",
                char_start=0,
                char_end=50,
                chunk_ids=("chunk_1",),
                tier=EvidenceTier.CRITICAL,
            ),
        ),
        supporting=(
            EvidenceEntry(
                span_id="span_1",
                citation="10 CFR 50.46(b)(2)",
                text="Local oxidation limit.",
                char_start=51,
                char_end=80,
                chunk_ids=("chunk_2",),
                tier=EvidenceTier.SUPPORTING,
            ),
        ),
    )


class _MockLLMClient:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[str] = []

    def complete(self, prompt: str, config: StageConfig) -> LLMResponse:
        self.calls.append(prompt)
        return LLMResponse(
            text=self._response_text,
            model=config.model,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


def _good_scores() -> str:
    return json.dumps({
        "plausibility": 0.9,
        "boundedness": 0.8,
        "ambiguity": 0.9,
        "specificity": 0.85,
        "leakage": 0.8,
    })


def _make_validator(
    response_text: str = "",
    thresholds: dict[str, float] | None = None,
) -> tuple[LLMValidator, _MockLLMClient]:
    client = _MockLLMClient(response_text or _good_scores())
    det = DeterministicValidator()
    config = StageConfig(model="gpt-4o")
    validator = LLMValidator(client, config, deterministic=det, score_thresholds=thresholds)
    return validator, client


# -- Protocol conformance -----------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_query_validator_protocol(self) -> None:
        validator, _ = _make_validator()
        assert isinstance(validator, QueryValidator)


# -- Deterministic gate -------------------------------------------------------


class TestDeterministicGate:
    def test_no_llm_call_when_deterministic_fails(self) -> None:
        validator, client = _make_validator()
        # Missing CFR citation and no evidence → deterministic fails
        candidate = _make_candidate(
            query="short",
            evidence_ids=(),
        )
        result = validator.validate(candidate)

        assert not result.is_valid
        assert len(client.calls) == 0  # No LLM call made

    def test_llm_called_when_deterministic_passes(self) -> None:
        validator, client = _make_validator()
        candidate = _make_candidate()

        validator.validate(candidate)

        assert len(client.calls) == 1

    def test_deterministic_failure_has_empty_scores(self) -> None:
        validator, _ = _make_validator()
        candidate = _make_candidate(query="short", evidence_ids=())
        result = validator.validate(candidate)

        assert result.scores == {}


# -- LLM scoring --------------------------------------------------------------


class TestLLMScoring:
    def test_all_five_dimensions_in_scores(self) -> None:
        validator, _ = _make_validator()
        result = validator.validate(_make_candidate())

        for dim in ("plausibility", "boundedness", "ambiguity", "specificity", "leakage"):
            assert dim in result.scores

    def test_passing_scores_yield_valid_result(self) -> None:
        validator, _ = _make_validator()
        result = validator.validate(_make_candidate())

        assert result.is_valid

    def test_low_score_adds_flag(self) -> None:
        scores = {
            "plausibility": 0.3,  # below 0.6 threshold
            "boundedness": 0.8,
            "ambiguity": 0.9,
            "specificity": 0.8,
            "leakage": 0.9,
        }
        validator, _ = _make_validator(json.dumps(scores))
        result = validator.validate(_make_candidate())

        assert any("low_plausibility" in f for f in result.flags)
        assert not result.is_valid

    def test_multiple_low_scores_add_multiple_flags(self) -> None:
        scores = {
            "plausibility": 0.3,
            "boundedness": 0.2,
            "ambiguity": 0.9,
            "specificity": 0.9,
            "leakage": 0.9,
        }
        validator, _ = _make_validator(json.dumps(scores))
        result = validator.validate(_make_candidate())

        low_flags = [f for f in result.flags if f.startswith("low_")]
        assert len(low_flags) == 2

    def test_flag_includes_score_value(self) -> None:
        scores = {
            "plausibility": 0.3,
            "boundedness": 0.8,
            "ambiguity": 0.9,
            "specificity": 0.8,
            "leakage": 0.9,
        }
        validator, _ = _make_validator(json.dumps(scores))
        result = validator.validate(_make_candidate())

        assert any("0.30" in f for f in result.flags)

    def test_custom_threshold_respected(self) -> None:
        # All scores 0.7 — should fail with default 0.8 threshold but pass with 0.5
        scores = dict.fromkeys(("plausibility", "boundedness", "ambiguity", "specificity", "leakage"), 0.7)
        validator_strict, _ = _make_validator(
            json.dumps(scores),
            thresholds={"plausibility": 0.8},
        )
        result = validator_strict.validate(_make_candidate())
        assert any("low_plausibility" in f for f in result.flags)

        validator_lenient, _ = _make_validator(
            json.dumps(scores),
            thresholds={"plausibility": 0.5},
        )
        result_lenient = validator_lenient.validate(_make_candidate())
        assert not any("low_plausibility" in f for f in result_lenient.flags)


# -- Malformed LLM response ---------------------------------------------------


class TestMalformedResponse:
    def test_parse_error_adds_flag(self) -> None:
        validator, _ = _make_validator("this is not json")
        result = validator.validate(_make_candidate())

        assert "llm_parse_error" in result.flags
        assert not result.is_valid

    def test_parse_error_scores_empty(self) -> None:
        validator, _ = _make_validator("not json")
        result = validator.validate(_make_candidate())

        assert result.scores == {}

    def test_non_dict_response_adds_flag(self) -> None:
        validator, _ = _make_validator(json.dumps([1, 2, 3]))
        result = validator.validate(_make_candidate())

        assert "llm_parse_error" in result.flags

    def test_missing_dimension_adds_flag(self) -> None:
        # Omit one required dimension
        partial = {"plausibility": 0.8, "boundedness": 0.8, "ambiguity": 0.8}
        validator, _ = _make_validator(json.dumps(partial))
        result = validator.validate(_make_candidate())

        assert "llm_parse_error" in result.flags

    def test_code_fence_stripped_before_parse(self) -> None:
        inner = _good_scores()
        fenced = f"```json\n{inner}\n```"
        validator, _ = _make_validator(fenced)
        result = validator.validate(_make_candidate())

        assert result.is_valid
        assert "llm_parse_error" not in result.flags


# -- refine_evidence ----------------------------------------------------------


class TestRefineEvidence:
    def _refine_response(
        self,
        critical: list[str],
        supporting: list[str] | None = None,
        contextual: list[str] | None = None,
    ) -> str:
        return json.dumps({
            "critical": critical,
            "supporting": supporting or [],
            "contextual": contextual or [],
        })

    def test_returns_new_evidence_set(self) -> None:
        validator, _ = _make_validator(self._refine_response(["span_0"]))
        query = _make_validated_query()
        evidence = _make_evidence()

        refined = validator.refine_evidence(query, evidence)

        assert refined is not evidence

    def test_span_ids_preserved_correctly(self) -> None:
        validator, _ = _make_validator(self._refine_response(["span_0"]))
        query = _make_validated_query()
        evidence = _make_evidence()

        refined = validator.refine_evidence(query, evidence)

        assert len(refined.critical) == 1
        assert refined.critical[0].span_id == "span_0"

    def test_fallback_on_malformed_json(self) -> None:
        validator, _ = _make_validator("not json at all")
        query = _make_validated_query()
        evidence = _make_evidence()

        refined = validator.refine_evidence(query, evidence)

        # Falls back to original evidence unchanged
        assert refined is evidence

    def test_hallucinated_span_ids_filtered(self) -> None:
        # LLM returns a span_id not in the original evidence
        validator, _ = _make_validator(
            self._refine_response(["span_0", "span_FAKE_99"])
        )
        query = _make_validated_query()
        evidence = _make_evidence()

        refined = validator.refine_evidence(query, evidence)

        assert all(e.span_id != "span_FAKE_99" for e in refined.critical)

    def test_unit_id_preserved(self) -> None:
        validator, _ = _make_validator(self._refine_response(["span_0"]))
        query = _make_validated_query()
        evidence = _make_evidence("50.46_b_1")

        refined = validator.refine_evidence(query, evidence)

        assert refined.unit_id == "50.46_b_1"

    def test_tier_reassignment(self) -> None:
        # LLM promotes span_1 (originally supporting) to critical
        validator, _ = _make_validator(
            self._refine_response(critical=["span_0", "span_1"])
        )
        query = _make_validated_query()
        evidence = _make_evidence()

        refined = validator.refine_evidence(query, evidence)

        critical_ids = {e.span_id for e in refined.critical}
        assert "span_1" in critical_ids
