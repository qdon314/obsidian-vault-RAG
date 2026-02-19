import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rag.adapters.query_generation.case_query_generator import CaseQueryGenerator
from rag.adapters.query_generation.term_mapper import TermMapper

FIXTURE_CASE_MD = """\
---
corpus: "case"
regime: "us-nrc"
corpus_label: "nrc-cases"
accession_number: "ML99999A001"
title: "Test Inspection Report"
document_type: "Inspection Report"
document_date: "2024-06-15T00:00:00"
case_category: "inspection"
case_subcategory: "inspection_routine_report"
case_category_confidence: 0.95
facility_name: "Test Nuclear Station"
reactor_type: "PWR"
dockets: ["05000999"]
regulation_parts: []
regulation_sections: ["50.46", "50.36"]
citation_keys: ["cfr:10:50.46", "cfr:10:50.36"]
cross_references: ["10 CFR §50.46", "10 CFR §50.36"]
source_url: "https://example.com"
---

# ML99999A001 — Test Inspection Report

**Facility:** Test Nuclear Station

## Content

The licensee failed to perform ECCS accumulator surveillance testing as
required. The ECCS accumulator level was found below the LCO limit.
Surveillance testing was not completed within the required interval.
The ECCS system was declared inoperable. ECCS operability was restored
after corrective maintenance. Surveillance testing confirmed ECCS
accumulator parameters met technical specification requirements.
The LCO action statement was entered and exited appropriately.
LCO 3.5.1 requires accumulator operability when reactor coolant
system pressure is above 1000 psig.
"""


def _make_term_json(terms: list[dict[str, Any]] | None = None) -> str:
    """Create a valid term map JSON string."""
    if terms is None:
        terms = [
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
                "id": "term.accumulator",
                "canonical": "accumulator",
                "type": "anchor",
                "anchors": [
                    {"ref": "cfr.10.50.46", "weight": 0.6},
                    {"ref": "cfr.10.50.36", "weight": 0.4},
                ],
            },
        ]

    data = {
        "version": "0.1",
        "refs": {
            "cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"},
            "cfr.10.50.36": {"label": "10 CFR 50.36", "kind": "cfr"},
        },
        "terms": terms,
    }
    return json.dumps(data)


def _make_term_mapper(terms: list[dict[str, Any]] | None = None) -> TermMapper:
    with TemporaryDirectory() as td:
        p = Path(td) / "terms.json"
        p.write_text(_make_term_json(terms))
        return TermMapper.from_json(p)


def _make_case_file(tmp_dir: str, content: str = FIXTURE_CASE_MD) -> Path:
    p = Path(tmp_dir) / "ML99999A001.md"
    p.write_text(content)
    return p


class TestStrategy1DirectCitation:
    def test_generates_one_query_per_cross_reference(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s1 = [q for q in queries if "citation-direct" in q["tags"]]
        assert len(s1) == 2

    def test_query_references_cited_regulation(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s1 = [q for q in queries if "citation-direct" in q["tags"]]
        citations_in_queries = {q["relevant_citations"][0] for q in s1}
        assert "10 CFR §50.46" in citations_in_queries
        assert "10 CFR §50.36" in citations_in_queries

    def test_query_has_required_fields(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        required = {
            "qid",
            "query",
            "difficulty",
            "query_type",
            "requires_synthesis",
            "is_unanswerable",
            "expected_answer",
            "unanswerable_reason",
            "relevant_citations",
            "tags",
            "source_case",
            "metadata",
        }
        for q in queries:
            missing = required - set(q.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_strategy1_field_values(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s1 = [q for q in queries if "citation-direct" in q["tags"]]
        for q in s1:
            assert q["difficulty"] == "easy"
            assert q["query_type"] == "factual"
            assert q["requires_synthesis"] is False
            assert q["is_unanswerable"] is False
            assert q["expected_answer"] is None
            assert q["source_case"] == "ML99999A001"
            assert q["case_document_type"] == "Inspection Report"
            assert q["metadata"] == {
                "filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}
            }

    def test_uses_varied_templates(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s1 = [q for q in queries if "citation-direct" in q["tags"]]
        texts = [q["query"] for q in s1]
        assert len(set(texts)) == 2
        # The two queries should use different templates (i=0, i=1)
        assert texts[0] != texts[1]


class TestStrategy2TermMapping:
    def test_generates_queries_for_matched_terms(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        assert len(s2) >= 1

    def test_includes_technical_term_field(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        for q in s2:
            assert "technical_term" in q
            assert isinstance(q["technical_term"], str)

    def test_strategy2_field_values(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        for q in s2:
            assert q["difficulty"] == "medium"
            assert q["query_type"] == "interpretive"
            assert q["requires_synthesis"] is True
            assert q["source_case"] == "ML99999A001"
            assert q["metadata"] == {
                "filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}
            }

    def test_overlap_tagging_when_citation_in_cross_references(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        # ECCS maps to "10 CFR 50.46" which overlaps with cross_ref "10 CFR §50.46"
        eccs_queries = [q for q in s2 if q["technical_term"] == "ECCS"]
        assert len(eccs_queries) >= 1
        eccs_q = eccs_queries[0]
        assert eccs_q["overlaps_direct_citation"] is True
        assert "overlaps-citation" in eccs_q["tags"]

    def test_max_5_term_mapping_queries(self):
        # Create 10 terms that all appear >= 2 times
        many_terms = [
            {
                "id": f"term.term{i}",
                "canonical": f"term{i}",
                "type": "anchor",
                "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
            }
            for i in range(10)
        ]
        content_parts = []
        for i in range(10):
            content_parts.append(f"term{i} term{i} term{i}")
        body = "\n".join(content_parts)
        md = (
            "---\n"
            'accession_number: "ML99999A001"\n'
            'document_type: "Report"\n'
            "cross_references: []\n"
            "---\n\n" + body
        )
        mapper = _make_term_mapper(many_terms)
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td, content=md)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        assert len(s2) == 5

    def test_anchor_term_tagged(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        eccs_queries = [q for q in s2 if q["technical_term"] == "ECCS"]
        assert len(eccs_queries) >= 1
        assert "anchor-term" in eccs_queries[0]["tags"]
        assert eccs_queries[0]["term_type"] == "anchor"

    def test_contextual_term_tagged(self):
        terms = [
            {
                "id": "term.risk_informed",
                "canonical": "risk-informed",
                "type": "contextual",
                "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
            }
        ]
        body_lines = ["A risk-informed approach was used. " * 3]
        md = (
            "---\n"
            'accession_number: "ML99999A001"\n'
            'document_type: "Report"\n'
            "cross_references: []\n"
            "---\n\n" + "\n".join(body_lines)
        )
        mapper = _make_term_mapper(terms)
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td, content=md)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        assert len(s2) >= 1
        assert "contextual-term" in s2[0]["tags"]
        assert "anchor-term" not in s2[0]["tags"]
        assert s2[0]["term_type"] == "contextual"

    def test_includes_anchor_refs_field(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        for q in s2:
            assert "anchor_refs" in q
            assert isinstance(q["anchor_refs"], list)


class TestQidUniqueness:
    def test_qids_are_unique_within_case(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        qids = [q["qid"] for q in queries]
        assert len(qids) == len(set(qids))

    def test_qids_are_unique_across_cases(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        all_qids: list[str] = []
        for suffix in ("A001", "A002"):
            md = FIXTURE_CASE_MD.replace("ML99999A001", f"ML99999{suffix}")
            with TemporaryDirectory() as td:
                p = Path(td) / f"ML99999{suffix}.md"
                p.write_text(md)
                queries = gen.generate(p)
            all_qids.extend(q["qid"] for q in queries)
        assert len(all_qids) == len(set(all_qids))


class TestMaxQueriesPerCase:
    def test_truncates_to_max(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper, max_queries_per_case=3)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        assert len(queries) <= 3
