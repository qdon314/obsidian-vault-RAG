"""Tests for Stage 1b LLM-based semantic classification."""

from __future__ import annotations

import json

from benchmark.domain.enums import UnitKind
from benchmark.domain.models import BenchmarkSourceSpan, RegulatoryUnit, StageConfig
from benchmark.ports.llm_client import LLMClient, LLMResponse

from benchmark.adapters.extraction.llm_extractor import LLMExtractor

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
    kind: UnitKind = UnitKind.OBLIGATION,
    cross_refs: tuple[str, ...] = (),
) -> RegulatoryUnit:
    return RegulatoryUnit(
        unit_id="50.46_b_1",
        kind=kind,
        spans=(_make_span(),),
        citation="10 CFR 50.46(b)(1)",
        subsection_chain=("b", "1"),
        parent_section_id="50.46",
        corpus_snapshot_id="snap1",
        cross_references=cross_refs,
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


_VALID_RESPONSE = json.dumps(
    {
        "kind": "threshold",
        "canonical_statement": "Peak cladding temperature must not exceed 2200\u00b0F.",
        "entities": ["licensee"],
        "value": "2200\u00b0F",
        "conditions": ["during LOCA"],
    }
)

_CONFIG = StageConfig(model="test-model")


# ── Tests ────────────────────────────────────────────────────────────


class TestLLMExtractor:
    def test_classify_enriches_semantic_fields(self) -> None:
        client = _MockLLMClient(_VALID_RESPONSE)
        extractor = LLMExtractor(client, _CONFIG)
        units = extractor.classify([_make_unit()])

        assert len(units) == 1
        unit = units[0]
        assert unit.kind == UnitKind.THRESHOLD
        assert unit.canonical_statement == "Peak cladding temperature must not exceed 2200\u00b0F."
        assert unit.entities == ("licensee",)
        assert unit.value == "2200\u00b0F"
        assert unit.conditions == ("during LOCA",)

    def test_unit_id_never_changes(self) -> None:
        client = _MockLLMClient(_VALID_RESPONSE)
        extractor = LLMExtractor(client, _CONFIG)
        original = _make_unit()
        enriched = extractor.classify([original])[0]

        assert enriched.unit_id == original.unit_id
        assert enriched.spans == original.spans
        assert enriched.citation == original.citation
        assert enriched.subsection_chain == original.subsection_chain
        assert enriched.parent_section_id == original.parent_section_id
        assert enriched.corpus_snapshot_id == original.corpus_snapshot_id
        assert enriched.cross_references == original.cross_references

    def test_malformed_json_flags_low_confidence(self) -> None:
        client = _MockLLMClient("this is not JSON at all")
        extractor = LLMExtractor(client, _CONFIG)
        units = extractor.classify([_make_unit()])

        assert len(units) == 1
        assert units[0].metadata.get("low_confidence") is True
        # Semantic fields remain at defaults.
        assert units[0].canonical_statement is None
        assert units[0].kind == UnitKind.OBLIGATION  # unchanged

    def test_invalid_kind_keeps_existing(self) -> None:
        bad_kind = json.dumps(
            {
                "kind": "not_a_real_kind",
                "canonical_statement": "Some statement.",
                "entities": [],
                "value": None,
                "conditions": [],
            }
        )
        client = _MockLLMClient(bad_kind)
        extractor = LLMExtractor(client, _CONFIG)
        units = extractor.classify([_make_unit(kind=UnitKind.PROHIBITION)])

        assert units[0].kind == UnitKind.PROHIBITION
        # Other fields still populated.
        assert units[0].canonical_statement == "Some statement."

    def test_markdown_code_fence_stripped(self) -> None:
        fenced = f"```json\n{_VALID_RESPONSE}\n```"
        client = _MockLLMClient(fenced)
        extractor = LLMExtractor(client, _CONFIG)
        units = extractor.classify([_make_unit()])

        assert units[0].kind == UnitKind.THRESHOLD

    def test_multiple_units(self) -> None:
        client = _MockLLMClient(_VALID_RESPONSE)
        extractor = LLMExtractor(client, _CONFIG)
        units = extractor.classify([_make_unit(), _make_unit()])

        assert len(units) == 2
        assert len(client.calls) == 2

    def test_empty_input(self) -> None:
        client = _MockLLMClient(_VALID_RESPONSE)
        extractor = LLMExtractor(client, _CONFIG)
        assert extractor.classify([]) == []
        assert len(client.calls) == 0

    def test_prompt_includes_span_text(self) -> None:
        client = _MockLLMClient(_VALID_RESPONSE)
        extractor = LLMExtractor(client, _CONFIG)
        extractor.classify([_make_unit()])

        assert "Peak cladding temperature." in client.calls[0]

    def test_prompt_includes_subsection_chain(self) -> None:
        client = _MockLLMClient(_VALID_RESPONSE)
        extractor = LLMExtractor(client, _CONFIG)
        extractor.classify([_make_unit()])

        assert "b > 1" in client.calls[0]

    def test_non_dict_json_flags_low_confidence(self) -> None:
        client = _MockLLMClient('"just a string"')
        extractor = LLMExtractor(client, _CONFIG)
        units = extractor.classify([_make_unit()])

        assert units[0].metadata.get("low_confidence") is True
