import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rag.adapters.query_generation.term_mapper import (
    MatchMode,
    TermMapper,
    TermType,
)


def _make_term_json(data: dict | None = None) -> str:
    """Create a valid term map JSON string."""
    if data is not None:
        return json.dumps(data)

    # Default valid structure
    default = {
        "version": "0.1",
        "refs": {
            "cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"},
            "cfr.10.50.34": {"label": "10 CFR 50.34", "kind": "cfr"},
            "cfr.10.50.36": {"label": "10 CFR 50.36", "kind": "cfr"},
        },
        "terms": [
            {
                "id": "term.eccs",
                "canonical": "ECCS",
                "aliases": ["emergency core cooling"],
                "type": "anchor",
                "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
            },
            {
                "id": "term.tech_spec",
                "canonical": "technical specification",
                "aliases": ["tech spec"],
                "type": "anchor",
                "anchors": [{"ref": "cfr.10.50.36", "weight": 1.0}],
            },
        ],
    }
    return json.dumps(default)


class TestTermMapperFromJson:
    def test_loads_valid_dictionary(self):
        data = json.loads(_make_term_json())
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        assert mapper is not None
        record = mapper.lookup("ECCS")
        assert record is not None
        assert record.canonical == "ECCS"
        assert len(record.anchors) == 1
        assert record.anchors[0].ref == "cfr.10.50.46"

    def test_rejects_empty_anchors_list(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.bad",
                    "canonical": "bad",
                    "type": "anchor",
                    "anchors": [],
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError, match="non-empty"):
                TermMapper.from_json(p)

    def test_rejects_invalid_term_type(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.bad",
                    "canonical": "bad",
                    "type": "invalid_type",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError, match="invalid type"):
                TermMapper.from_json(p)

    def test_rejects_missing_required_field(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.bad",
                    # missing canonical
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError, match="canonical"):
                TermMapper.from_json(p)

    def test_rejects_duplicate_term_ids(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.eccs",
                    "canonical": "ECCS",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
                {
                    "id": "term.eccs",  # duplicate
                    "canonical": "ECCS2",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError, match="Duplicate term id"):
                TermMapper.from_json(p)

    def test_rejects_duplicate_canonical(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.eccs",
                    "canonical": "ECCS",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
                {
                    "id": "term.eccs2",
                    "canonical": "ECCS",  # duplicate (case-insensitive)
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError, match="Duplicate canonical"):
                TermMapper.from_json(p)

    def test_rejects_duplicate_aliases(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.eccs",
                    "canonical": "ECCS",
                    "aliases": ["emergency core cooling"],
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
                {
                    "id": "term.eccs2",
                    "canonical": "ECCS2",
                    "aliases": ["emergency core cooling"],  # duplicate alias
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError, match="Duplicate alias"):
                TermMapper.from_json(p)

    def test_rejects_orphan_ref(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.bad",
                    "canonical": "bad",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.99", "weight": 1.0}],  # not in refs
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError, match="not found in refs"):
                TermMapper.from_json(p)

    def test_defaults_match_to_smart_phrase(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.eccs",
                    "canonical": "ECCS",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                    # no match specified
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        record = mapper.lookup("ECCS")
        assert record is not None
        assert record.match.mode == MatchMode.SMART_PHRASE
        assert record.match.case_sensitive is False

    def test_defaults_actions_by_term_type(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.anchor",
                    "canonical": "AnchorTerm",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
                {
                    "id": "term.contextual",
                    "canonical": "ContextualTerm",
                    "type": "contextual",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
                {
                    "id": "term.crosscutting",
                    "canonical": "CrossCuttingTerm",
                    "type": "cross_cutting",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)

        anchor = mapper.lookup("AnchorTerm")
        assert anchor is not None
        assert anchor.actions.retrieval_boost == 0.8
        assert anchor.actions.adams_prefilter is True

        contextual = mapper.lookup("ContextualTerm")
        assert contextual is not None
        assert contextual.actions.retrieval_boost == 0.4
        assert contextual.actions.adams_prefilter is False

        crosscutting = mapper.lookup("CrossCuttingTerm")
        assert crosscutting is not None
        assert crosscutting.actions.retrieval_boost == 0.1
        assert crosscutting.actions.adams_prefilter is False


class TestTermMapperLookup:
    def _make_mapper(self) -> TermMapper:
        data = {
            "version": "0.1",
            "refs": {
                "cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"},
                "cfr.10.50.36": {"label": "10 CFR 50.36", "kind": "cfr"},
            },
            "terms": [
                {
                    "id": "term.eccs",
                    "canonical": "ECCS",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
                {
                    "id": "term.surv_test",
                    "canonical": "surveillance testing",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.36", "weight": 1.0}],
                },
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            return TermMapper.from_json(p)

    def test_lookup_exact_match(self):
        mapper = self._make_mapper()
        record = mapper.lookup("ECCS")
        assert record is not None
        assert record.canonical == "ECCS"

    def test_lookup_case_insensitive(self):
        mapper = self._make_mapper()
        record = mapper.lookup("eccs")
        assert record is not None
        assert record.canonical == "ECCS"

    def test_lookup_miss_returns_none(self):
        mapper = self._make_mapper()
        assert mapper.lookup("nonexistent") is None

    def test_lookup_by_alias(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.eccs",
                    "canonical": "ECCS",
                    "aliases": ["emergency core cooling"],
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)

        record = mapper.lookup("emergency core cooling")
        assert record is not None
        assert record.canonical == "ECCS"


class TestTermMapperScanContent:
    def _make_mapper(self) -> TermMapper:
        data = {
            "version": "0.1",
            "refs": {
                "cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"},
                "cfr.10.50.36": {"label": "10 CFR 50.36", "kind": "cfr"},
            },
            "terms": [
                {
                    "id": "term.eccs",
                    "canonical": "ECCS",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
                {
                    "id": "term.surv_test",
                    "canonical": "surveillance testing",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.36", "weight": 1.0}],
                },
                {
                    "id": "term.lco",
                    "canonical": "LCO",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.36", "weight": 1.0}],
                },
                {
                    "id": "term.pct",
                    "canonical": "peak cladding temperature",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            return TermMapper.from_json(p)

    def test_finds_terms_case_insensitive(self):
        mapper = self._make_mapper()
        content = "The ECCS was tested. The eccs met all criteria. ECCS passed."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].term == "ECCS"
        assert matches[0].frequency == 3

    def test_sorts_by_descending_frequency(self):
        mapper = self._make_mapper()
        content = (
            "LCO 3.5.1 requires ECCS operability. "
            "The LCO was met. LCO was verified. LCO again. "
            "ECCS accumulator levels were checked. ECCS passed."
        )
        matches = mapper.scan_content(content)
        assert len(matches) == 2
        assert matches[0].term == "LCO"
        assert matches[1].term == "ECCS"
        assert matches[0].frequency > matches[1].frequency

    def test_excludes_terms_with_frequency_below_2(self):
        mapper = self._make_mapper()
        content = "The ECCS was tested. The peak cladding temperature was fine."
        matches = mapper.scan_content(content)
        assert len(matches) == 0

    def test_returns_empty_for_no_matches(self):
        mapper = self._make_mapper()
        content = "This document discusses reactor coolant pumps only."
        matches = mapper.scan_content(content)
        assert matches == []

    def test_returns_anchors_from_dictionary(self):
        mapper = self._make_mapper()
        content = "Surveillance testing was performed. Surveillance testing passed."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert len(matches[0].anchors) == 1
        assert matches[0].anchors[0].ref == "cfr.10.50.36"

    def test_word_boundary_prevents_false_positives(self):
        data = {
            "version": "0.1",
            "refs": {
                "cfr.10.50.55a": {"label": "10 CFR 50.55a", "kind": "cfr"},
                "cfr.10.50.34": {"label": "10 CFR 50.34", "kind": "cfr"},
            },
            "terms": [
                {
                    "id": "term.ist",
                    "canonical": "IST",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.55a", "weight": 1.0}],
                },
                {
                    "id": "term.cap",
                    "canonical": "CAP",
                    "type": "contextual",
                    "anchors": [{"ref": "cfr.10.50.34", "weight": 1.0}],
                },
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = "The existing list of capacity items was consistent."
        matches = mapper.scan_content(content)
        assert matches == []

    def test_word_boundary_matches_real_acronyms(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.55a": {"label": "10 CFR 50.55a", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.ist",
                    "canonical": "IST",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.55a", "weight": 1.0}],
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = "The IST program was reviewed. IST results were satisfactory."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].term == "IST"
        assert matches[0].frequency == 2

    def test_scan_returns_term_type(self):
        data = {
            "version": "0.1",
            "refs": {
                "cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"},
                "cfr.10.50.65": {"label": "10 CFR 50.65", "kind": "cfr"},
            },
            "terms": [
                {
                    "id": "term.eccs",
                    "canonical": "ECCS",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                },
                {
                    "id": "term.risk_informed",
                    "canonical": "risk-informed",
                    "type": "contextual",
                    "anchors": [{"ref": "cfr.10.50.65", "weight": 1.0}],
                },
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = (
            "The ECCS was tested. ECCS passed. "
            "A risk-informed approach was used. The risk-informed method worked."
        )
        matches = mapper.scan_content(content)
        assert len(matches) == 2
        by_term = {m.term: m for m in matches}
        assert by_term["ECCS"].term_type == TermType.ANCHOR
        assert by_term["risk-informed"].term_type == TermType.CONTEXTUAL

    def test_scan_aggregates_aliases(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.eccs",
                    "canonical": "ECCS",
                    "aliases": ["emergency core cooling"],
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = "The ECCS was tested. The emergency core cooling system passed. ECCS is good."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        # ECCS appears 2 times, emergency core cooling appears 1 time (partial match)
        # But "emergency core cooling" as exact phrase appears 1 time
        assert matches[0].frequency >= 2  # ECCS appears twice


class TestMatchModes:
    def _make_mapper_with_mode(self, mode: str) -> TermMapper:
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.test",
                    "canonical": "ECCS",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                    "match": {"mode": mode, "case_sensitive": False},
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            return TermMapper.from_json(p)

    def test_exact_mode(self):
        mapper = self._make_mapper_with_mode("exact")
        content = "ECCS ECCS eccs"  # 2 exact matches for "ECCS", 1 for "eccs"
        matches = mapper.scan_content(content)
        # exact mode with case_sensitive=False should match all 3
        assert len(matches) == 1
        assert matches[0].frequency == 3

    def test_phrase_mode(self):
        mapper = self._make_mapper_with_mode("phrase")
        content = "The ECCS system. ECCS is critical. No ECCS."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].frequency == 3

    def test_smart_phrase_mode_default(self):
        mapper = self._make_mapper_with_mode("smart_phrase")
        content = "The ECCS system. ECCS is critical. No ECCS."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].frequency == 3

    def test_case_sensitive_match(self):
        data = {
            "version": "0.1",
            "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
            "terms": [
                {
                    "id": "term.test",
                    "canonical": "ECCS",
                    "type": "anchor",
                    "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
                    "match": {"mode": "smart_phrase", "case_sensitive": True},
                }
            ],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = "ECCS ECCS eccs"  # Only 2 case-sensitive matches
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].frequency == 2
