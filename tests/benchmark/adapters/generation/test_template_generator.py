"""Tests for Stage 3 template-based query generation."""

from __future__ import annotations

import json

from benchmark.adapters.generation.template_generator import TemplateQueryGenerator
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


def _make_unit(
    *,
    canonical_statement: str | None = "peak cladding temperature limit",
    entities: tuple[str, ...] = ("licensee",),
) -> RegulatoryUnit:
    return RegulatoryUnit(
        unit_id="50.46_b_1",
        kind=UnitKind.THRESHOLD,
        spans=(_make_span(),),
        citation="10 CFR 50.46(b)(1)",
        subsection_chain=("b", "1"),
        parent_section_id="50.46",
        corpus_snapshot_id="snap1",
        canonical_statement=canonical_statement,
        entities=entities,
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


# ── Tests ────────────────────────────────────────────────────────────


class TestTemplateQueryGeneratorProtocol:
    def test_satisfies_query_generator_protocol(self) -> None:
        client = _MockLLMClient("[]")
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))
        assert isinstance(gen, QueryGenerator)


class TestGenerate:
    def test_produces_candidates_from_paraphrases(self) -> None:
        paraphrases = json.dumps(
            [
                "What is the peak cladding temperature limit per 10 CFR 50.46(b)(1)?",
                "Under 10 CFR 50.46(b)(1), what temperature threshold applies?",
            ]
        )
        client = _MockLLMClient(paraphrases)
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))
        unit = _make_unit()
        evidence = _make_evidence()

        result = gen.generate(unit, evidence, QueryClass.CITATION_LOOKUP)

        assert len(result) == 2
        assert all(c.query_class == QueryClass.CITATION_LOOKUP for c in result)
        assert all(c.unit_id == "50.46_b_1" for c in result)
        assert all(c.corpus_snapshot_id == "snap1" for c in result)

    def test_candidate_id_is_unique_and_deterministic(self) -> None:
        paraphrases = json.dumps(["Q1?", "Q2?", "Q3?"])
        client = _MockLLMClient(paraphrases)
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.CITATION_LOOKUP)

        ids = [c.candidate_id for c in result]
        assert len(ids) == len(set(ids))  # all unique
        assert ids[0] == "qc_50.46_b_1_citation_lookup_0"
        assert ids[1] == "qc_50.46_b_1_citation_lookup_1"
        assert ids[2] == "qc_50.46_b_1_citation_lookup_2"

    def test_evidence_span_ids_from_critical_tier(self) -> None:
        paraphrases = json.dumps(["Question?"])
        client = _MockLLMClient(paraphrases)
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.CITATION_LOOKUP)

        assert result[0].evidence_span_ids == ("50.46_b_1_0",)

    def test_source_citations_from_unit(self) -> None:
        paraphrases = json.dumps(["Question?"])
        client = _MockLLMClient(paraphrases)
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.CITATION_LOOKUP)

        assert result[0].source_citations == ("10 CFR 50.46(b)(1)",)

    def test_fallback_to_template_on_malformed_json(self) -> None:
        client = _MockLLMClient("this is not json")
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.CITATION_LOOKUP)

        assert len(result) == 1
        assert "10 CFR 50.46(b)(1)" in result[0].query
        assert "peak cladding temperature limit" in result[0].query

    def test_fallback_to_template_on_empty_array(self) -> None:
        client = _MockLLMClient("[]")
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.CITATION_LOOKUP)

        assert len(result) == 1
        assert "10 CFR 50.46(b)(1)" in result[0].query

    def test_fallback_to_template_on_non_string_items(self) -> None:
        client = _MockLLMClient(json.dumps([123, None, ""]))
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.CITATION_LOOKUP)

        assert len(result) == 1  # fallback

    def test_strips_code_fences_from_response(self) -> None:
        fenced = '```json\n["What about 10 CFR 50.46(b)(1)?"]\n```'
        client = _MockLLMClient(fenced)
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.CITATION_LOOKUP)

        assert len(result) == 1
        assert result[0].query == "What about 10 CFR 50.46(b)(1)?"

    def test_raises_for_non_citation_lookup_class(self) -> None:
        client = _MockLLMClient("[]")
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))

        import pytest

        with pytest.raises(ValueError, match="Only CITATION_LOOKUP"):
            gen.generate(
                _make_unit(),
                _make_evidence(),
                QueryClass.NARROW_FACTUAL,
            )

    def test_unit_without_canonical_statement(self) -> None:
        paraphrases = json.dumps(["What does 10 CFR 50.46(b)(1) require?"])
        client = _MockLLMClient(paraphrases)
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))
        unit = _make_unit(canonical_statement=None, entities=())

        result = gen.generate(unit, _make_evidence(), QueryClass.CITATION_LOOKUP)

        assert len(result) == 1

    def test_difficulty_defaults_to_easy(self) -> None:
        paraphrases = json.dumps(["Question?"])
        client = _MockLLMClient(paraphrases)
        gen = TemplateQueryGenerator(client, StageConfig(model="gpt-4o"))

        result = gen.generate(_make_unit(), _make_evidence(), QueryClass.CITATION_LOOKUP)

        assert result[0].difficulty == "easy"
