import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rag.adapters.query_generation.term_mapper import TermMapper


class TestTermMapperFromJson:
    def test_loads_valid_dictionary(self):
        data = {
            "ECCS": ["10 CFR 50.46", "10 CFR 50.34"],
            "technical specification": ["10 CFR 50.36"],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        assert mapper.lookup("ECCS") == ["10 CFR 50.46", "10 CFR 50.34"]

    def test_rejects_empty_citation_list(self):
        data = {"bad_term": []}
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError):
                TermMapper.from_json(p)

    def test_rejects_non_string_citation(self):
        data = {"bad_term": [123]}
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError):
                TermMapper.from_json(p)


class TestTermMapperLookup:
    def _make_mapper(self) -> TermMapper:
        data = {
            "ECCS": ["10 CFR 50.46"],
            "surveillance testing": ["10 CFR 50.36"],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            return TermMapper.from_json(p)

    def test_lookup_exact_match(self):
        mapper = self._make_mapper()
        assert mapper.lookup("ECCS") == ["10 CFR 50.46"]

    def test_lookup_case_insensitive(self):
        mapper = self._make_mapper()
        assert mapper.lookup("eccs") == ["10 CFR 50.46"]

    def test_lookup_miss_returns_empty(self):
        mapper = self._make_mapper()
        assert mapper.lookup("nonexistent") == []


class TestTermMapperScanContent:
    def _make_mapper(self) -> TermMapper:
        data = {
            "ECCS": ["10 CFR 50.46"],
            "surveillance testing": ["10 CFR 50.36"],
            "LCO": ["10 CFR 50.36"],
            "peak cladding temperature": ["10 CFR 50.46"],
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

    def test_returns_citations_from_dictionary(self):
        mapper = self._make_mapper()
        content = "Surveillance testing was performed. Surveillance testing passed."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].citations == ["10 CFR 50.36"]

    def test_word_boundary_prevents_false_positives(self):
        data = {
            "IST": ["10 CFR 50.55a"],
            "CAP": ["10 CFR 50.34"],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = "The existing list of capacity items was consistent."
        matches = mapper.scan_content(content)
        assert matches == []

    def test_word_boundary_matches_real_acronyms(self):
        data = {"IST": ["10 CFR 50.55a"]}
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = "The IST program was reviewed. IST results were satisfactory."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].term == "IST"
        assert matches[0].frequency == 2
