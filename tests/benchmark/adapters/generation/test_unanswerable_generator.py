"""Tests for Stage 3 unanswerable query generation."""

from __future__ import annotations

import json

import pytest

from benchmark.adapters.generation.unanswerable_generator import (
    STRATEGY_DOMAIN_BOUNDARY,
    STRATEGY_FABRICATED_CITATION,
    STRATEGY_NEAR_MISS,
    UnanswerableGenerator,
)
from benchmark.domain.enums import EvidenceTier, QueryClass, UnitKind
from benchmark.domain.models import (
    BenchmarkSourceSpan,
    EvidenceEntry,
    EvidenceSet,
    RegulatoryUnit,
    StageConfig,
)
from benchmark.ports.llm_client import LLMResponse
from benchmark.ports.query_generator import QueryGenerator

# ── Helpers ──────────────────────────────────────────────────────────


def _make_span(text: str = "Peak cladding temperature.") -> BenchmarkSourceSpan:
    return BenchmarkSourceSpan(
        source_doc_id="doc_1",
        citation="10 CFR 50.46(b)(1)",
        citation_key="10_cfr_50.46_b_1",
        section_title="ECCS criteria",
        text=text,
        char_start=0,
        char_end=len(text),
        chunk_ids_overlapping_span=("c1",),
        parent_section_id="50.46",
        effective_date="2026-01-01",
        corpus_snapshot_id="snap1",
    )


def _make_unit(unit_id: str = "50.46_b_1") -> RegulatoryUnit:
    return RegulatoryUnit(
        unit_id=unit_id,
        kind=UnitKind.THRESHOLD,
        spans=(_make_span(),),
        citation="10 CFR 50.46(b)(1)",
        subsection_chain=("b", "1"),
        parent_section_id="50.46",
        corpus_snapshot_id="snap1",
        canonical_statement="peak cladding temperature limit",
    )


def _make_evidence(unit_id: str = "50.46_b_1") -> EvidenceSet:
    return EvidenceSet(
        unit_id=unit_id,
        critical=(
            EvidenceEntry(
                span_id=f"{unit_id}_0",
                citation="10 CFR 50.46(b)(1)",
                text="Peak cladding temperature.",
                char_start=0,
                char_end=26,
                chunk_ids=("c1",),
                tier=EvidenceTier.CRITICAL,
            ),
        ),
    )


class _MockLLMClient:
    """Mock LLM client that returns pre-configured responses."""

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


def _llm_response(query: str, reason: str) -> str:
    """Build a valid JSON response string."""
    return json.dumps({"query": query, "reason": reason})


# ── Protocol conformance ────────────────────────────────────────────


class TestProtocolConformance:
    def test_satisfies_query_generator_protocol(self) -> None:
        client = _MockLLMClient("{}")
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))
        assert isinstance(gen, QueryGenerator)


# ── Core generation behavior ────────────────────────────────────────


class TestGenerate:
    def test_raises_for_non_unanswerable_class(self) -> None:
        client = _MockLLMClient("{}")
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        with pytest.raises(ValueError, match="only supports UNANSWERABLE"):
            gen.generate(_make_unit(), _make_evidence(), QueryClass.CITATION_LOOKUP)

    def test_produces_candidate_with_correct_class(self) -> None:
        resp = _llm_response("Is there a 10 CFR 50.46(b)(6)?", "does not exist")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        assert len(result) == 1
        assert result[0].query_class == QueryClass.UNANSWERABLE

    def test_evidence_span_ids_empty(self) -> None:
        resp = _llm_response("Question?", "not in corpus")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        assert result[0].evidence_span_ids == ()

    def test_source_citations_empty(self) -> None:
        resp = _llm_response("Question?", "not in corpus")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        assert result[0].source_citations == ()

    def test_metadata_contains_unanswerable_fields(self) -> None:
        resp = _llm_response("Question?", "subsection does not exist")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        meta = result[0].metadata
        assert meta["is_unanswerable"] is True
        assert meta["unanswerable_reason"] == "subsection does not exist"
        assert meta["unanswerable_strategy"] in (
            STRATEGY_NEAR_MISS,
            STRATEGY_DOMAIN_BOUNDARY,
            STRATEGY_FABRICATED_CITATION,
        )

    def test_candidate_id_format(self) -> None:
        resp = _llm_response("Question?", "reason")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        cid = result[0].candidate_id
        assert cid.startswith("qc_50.46_b_1_unanswerable_")
        assert cid.endswith("_0")

    def test_corpus_snapshot_id_from_unit(self) -> None:
        resp = _llm_response("Question?", "reason")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        assert result[0].corpus_snapshot_id == "snap1"


# ── Strategy selection ──────────────────────────────────────────────


class TestStrategySelection:
    def test_deterministic_for_same_unit_id(self) -> None:
        resp = _llm_response("Q?", "reason")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        r1 = gen.generate(_make_unit("test_a"), _make_evidence("test_a"), QueryClass.UNANSWERABLE)
        r2 = gen.generate(_make_unit("test_a"), _make_evidence("test_a"), QueryClass.UNANSWERABLE)

        assert r1[0].metadata["unanswerable_strategy"] == r2[0].metadata["unanswerable_strategy"]

    def test_varies_across_different_unit_ids(self) -> None:
        """At least two different strategies across several unit IDs."""
        resp = _llm_response("Q?", "reason")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        strategies = set()
        for i in range(20):
            uid = f"unit_{i}"
            result = gen.generate(_make_unit(uid), _make_evidence(uid), QueryClass.UNANSWERABLE)
            strategies.add(result[0].metadata["unanswerable_strategy"])

        assert len(strategies) >= 2


# ── Per-strategy prompt content ─────────────────────────────────────


class TestNearMissStrategy:
    def test_prompt_references_section(self) -> None:
        resp = _llm_response("Q about 50.46?", "subsection missing")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        # Force near_miss strategy by finding a unit_id that hashes to it
        for i in range(100):
            uid = f"near_miss_test_{i}"
            gen_strategy = gen._select_strategy(uid)
            if gen_strategy == STRATEGY_NEAR_MISS:
                gen.generate(_make_unit(uid), _make_evidence(uid), QueryClass.UNANSWERABLE)
                assert len(client.calls) > 0
                assert "50.46" in client.calls[-1]
                return

        pytest.skip("Could not find unit_id that hashes to near_miss")


class TestDomainBoundaryStrategy:
    def test_prompt_references_adjacent_domain(self) -> None:
        resp = _llm_response("What does OSHA require?", "outside 10 CFR")
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        for i in range(100):
            uid = f"domain_test_{i}"
            gen_strategy = gen._select_strategy(uid)
            if gen_strategy == STRATEGY_DOMAIN_BOUNDARY:
                gen.generate(_make_unit(uid), _make_evidence(uid), QueryClass.UNANSWERABLE)
                prompt = client.calls[-1]
                assert any(domain in prompt for domain in UnanswerableGenerator.ADJACENT_DOMAINS)
                return

        pytest.skip("Could not find unit_id that hashes to domain_boundary")


class TestFabricatedCitationStrategy:
    def test_prompt_mentions_non_existent(self) -> None:
        resp = json.dumps(
            {
                "query": "What does 10 CFR 50.46(b)(6) require?",
                "fabricated_citation": "10 CFR 50.46(b)(6)",
                "reason": "only (b)(1)-(5) exist",
            }
        )
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        for i in range(100):
            uid = f"fab_test_{i}"
            gen_strategy = gen._select_strategy(uid)
            if gen_strategy == STRATEGY_FABRICATED_CITATION:
                gen.generate(_make_unit(uid), _make_evidence(uid), QueryClass.UNANSWERABLE)
                prompt = client.calls[-1]
                assert "NON-EXISTENT" in prompt
                return

        pytest.skip("Could not find unit_id that hashes to fabricated_citation")


# ── Fallback behavior ───────────────────────────────────────────────


class TestFallback:
    def test_fallback_on_malformed_json(self) -> None:
        client = _MockLLMClient("this is not json")
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        assert len(result) == 1
        assert result[0].metadata["is_unanswerable"] is True
        assert result[0].metadata.get("is_fallback") is True
        assert result[0].evidence_span_ids == ()

    def test_fallback_on_empty_query_field(self) -> None:
        resp = json.dumps({"query": "", "reason": "test"})
        client = _MockLLMClient(resp)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        assert len(result) == 1
        assert result[0].metadata.get("is_fallback") is True

    def test_fallback_on_non_dict_response(self) -> None:
        client = _MockLLMClient(json.dumps(["not", "a", "dict"]))
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        assert len(result) == 1
        assert result[0].metadata.get("is_fallback") is True

    def test_fallback_strips_code_fences(self) -> None:
        inner = json.dumps({"query": "Q?", "reason": "R"})
        fenced = f"```json\n{inner}\n```"
        client = _MockLLMClient(fenced)
        gen = UnanswerableGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.UNANSWERABLE)

        # Should parse successfully — NOT a fallback
        assert len(result) == 1
        assert result[0].query == "Q?"
        assert result[0].metadata.get("is_fallback") is not True
