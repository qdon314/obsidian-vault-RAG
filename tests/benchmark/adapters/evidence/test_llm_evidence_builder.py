"""Tests for Stage 2 LLM-based evidence tier assignment."""

from __future__ import annotations

import json

from benchmark.adapters.evidence.llm_evidence_builder import LLMEvidenceBuilder
from benchmark.domain.enums import EvidenceTier, UnitKind
from benchmark.domain.models import BenchmarkSourceSpan, RegulatoryUnit, StageConfig
from benchmark.ports.evidence_builder import EvidenceBuilder
from benchmark.ports.llm_client import LLMResponse

# ── Helpers ──────────────────────────────────────────────────────────


def _make_span(
    *,
    citation_key: str = "10_cfr_50.46_b_1",
    citation: str = "10 CFR 50.46(b)(1)",
    text: str = "Peak cladding temperature shall not exceed 2200°F.",
    char_start: int = 0,
    char_end: int = 51,
    parent_section_id: str = "50.46",
    subsection_tokens: tuple[str, ...] = ("b", "1"),
) -> BenchmarkSourceSpan:
    return BenchmarkSourceSpan(
        source_doc_id="doc_1",
        citation=citation,
        citation_key=citation_key,
        section_title="ECCS criteria",
        text=text,
        char_start=char_start,
        char_end=char_end,
        chunk_ids_overlapping_span=("c1",),
        parent_section_id=parent_section_id,
        effective_date="2026-01-01",
        corpus_snapshot_id="snap1",
        metadata={"subsection_tokens": subsection_tokens},
    )


def _make_unit(
    spans: tuple[BenchmarkSourceSpan, ...] | None = None,
) -> RegulatoryUnit:
    if spans is None:
        spans = (_make_span(),)
    return RegulatoryUnit(
        unit_id="50.46_b_1",
        kind=UnitKind.THRESHOLD,
        spans=spans,
        citation="10 CFR 50.46(b)(1)",
        subsection_chain=("b", "1"),
        parent_section_id="50.46",
        corpus_snapshot_id="snap1",
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


_CONFIG = StageConfig(model="test-model")


# ── Tests ────────────────────────────────────────────────────────────


class TestLLMEvidenceBuilder:
    def test_satisfies_evidence_builder_protocol(self) -> None:
        client = _MockLLMClient("{}")
        builder = LLMEvidenceBuilder(client, _CONFIG, [])
        assert isinstance(builder, EvidenceBuilder)

    def test_single_span_classified_critical(self) -> None:
        response = json.dumps({"unit_spans": {"0": "critical"}})
        client = _MockLLMClient(response)
        builder = LLMEvidenceBuilder(client, _CONFIG, [_make_span()])

        es = builder.build(_make_unit())
        assert es.unit_id == "50.46_b_1"
        assert len(es.critical) == 1
        assert len(es.supporting) == 0
        assert len(es.contextual) == 0
        assert es.critical[0].tier == EvidenceTier.CRITICAL
        assert es.critical[0].citation == "10 CFR 50.46(b)(1)"

    def test_multi_span_tier_assignment(self) -> None:
        span_a = _make_span(
            citation_key="a",
            text="Span A.",
            char_start=0,
            char_end=7,
        )
        span_b = _make_span(
            citation_key="b",
            text="Span B.",
            char_start=7,
            char_end=14,
        )
        response = json.dumps({"unit_spans": {"0": "critical", "1": "supporting"}})
        client = _MockLLMClient(response)
        builder = LLMEvidenceBuilder(client, _CONFIG, [span_a, span_b])

        es = builder.build(_make_unit(spans=(span_a, span_b)))
        assert len(es.critical) == 1
        assert len(es.supporting) == 1
        assert es.critical[0].text == "Span A."
        assert es.supporting[0].text == "Span B."

    def test_neighbor_spans_as_contextual(self) -> None:
        unit_span = _make_span(
            citation_key="unit",
            text="Unit text.",
            char_start=0,
            char_end=10,
        )
        neighbor = _make_span(
            citation_key="neighbor",
            text="Neighbor text.",
            char_start=10,
            char_end=24,
        )
        response = json.dumps(
            {
                "unit_spans": {"0": "critical"},
                "contextual_neighbors": [0],
            }
        )
        client = _MockLLMClient(response)
        builder = LLMEvidenceBuilder(client, _CONFIG, [unit_span, neighbor])

        es = builder.build(_make_unit(spans=(unit_span,)))
        assert len(es.critical) == 1
        assert len(es.contextual) == 1
        assert es.contextual[0].text == "Neighbor text."
        assert es.contextual[0].span_id == "50.46_b_1_n0"

    def test_malformed_response_falls_back_to_all_critical(self) -> None:
        client = _MockLLMClient("not json at all")
        span = _make_span()
        builder = LLMEvidenceBuilder(client, _CONFIG, [span])

        es = builder.build(_make_unit(spans=(span,)))
        assert len(es.critical) == 1
        assert len(es.supporting) == 0
        assert len(es.contextual) == 0

    def test_missing_span_index_defaults_to_critical(self) -> None:
        """If LLM omits a span index, it should default to critical."""
        response = json.dumps({"unit_spans": {}})  # no assignments
        client = _MockLLMClient(response)
        span = _make_span()
        builder = LLMEvidenceBuilder(client, _CONFIG, [span])

        es = builder.build(_make_unit(spans=(span,)))
        assert len(es.critical) == 1

    def test_invalid_tier_defaults_to_critical(self) -> None:
        response = json.dumps({"unit_spans": {"0": "bogus_tier"}})
        client = _MockLLMClient(response)
        span = _make_span()
        builder = LLMEvidenceBuilder(client, _CONFIG, [span])

        es = builder.build(_make_unit(spans=(span,)))
        assert len(es.critical) == 1

    def test_no_spans_from_other_sections_as_neighbors(self) -> None:
        """Neighbors must share the same parent_section_id."""
        unit_span = _make_span(parent_section_id="50.46")
        other_section_span = _make_span(
            citation_key="other",
            parent_section_id="50.47",
            text="Different section.",
            char_start=0,
            char_end=18,
        )
        response = json.dumps(
            {
                "unit_spans": {"0": "critical"},
                "contextual_neighbors": [0],
            }
        )
        client = _MockLLMClient(response)
        builder = LLMEvidenceBuilder(client, _CONFIG, [unit_span, other_section_span])

        es = builder.build(_make_unit(spans=(unit_span,)))
        # No contextual — the other-section span is not a valid neighbor.
        assert len(es.contextual) == 0

    def test_span_id_format(self) -> None:
        response = json.dumps({"unit_spans": {"0": "critical", "1": "supporting"}})
        span_a = _make_span(citation_key="a", char_start=0, char_end=5)
        span_b = _make_span(citation_key="b", char_start=5, char_end=10)
        client = _MockLLMClient(response)
        builder = LLMEvidenceBuilder(client, _CONFIG, [span_a, span_b])

        es = builder.build(_make_unit(spans=(span_a, span_b)))
        assert es.critical[0].span_id == "50.46_b_1_0"
        assert es.supporting[0].span_id == "50.46_b_1_1"

    def test_chunk_ids_carried_from_span(self) -> None:
        response = json.dumps({"unit_spans": {"0": "critical"}})
        client = _MockLLMClient(response)
        span = _make_span()
        builder = LLMEvidenceBuilder(client, _CONFIG, [span])

        es = builder.build(_make_unit(spans=(span,)))
        assert es.critical[0].chunk_ids == ("c1",)

    def test_prompt_includes_tier_definitions(self) -> None:
        response = json.dumps({"unit_spans": {"0": "critical"}})
        client = _MockLLMClient(response)
        builder = LLMEvidenceBuilder(client, _CONFIG, [_make_span()])
        builder.build(_make_unit())

        assert "critical" in client.calls[0]
        assert "supporting" in client.calls[0]
        assert "contextual" in client.calls[0]
        assert "incomprehensible" in client.calls[0]
