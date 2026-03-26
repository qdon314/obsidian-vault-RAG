import json
from pathlib import Path

import pytest

from benchmark_review.engine.loader import load_run
from benchmark_review.engine.models import ReviewStatus


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    candidates = [
        {
            "candidate_id": "qc_50.1_citation_lookup_0",
            "unit_id": "50.1",
            "query": "What does 10 CFR 50.1 say?",
            "query_class": "citation_lookup",
            "source_citations": ["10 CFR 50.1"],
            "evidence_span_ids": ["50.1_0"],
            "difficulty": "easy",
            "corpus_snapshot_id": "",
            "metadata": {},
        }
    ]
    validation = [
        {
            "candidate_id": "qc_50.1_citation_lookup_0",
            "is_valid": False,
            "flags": ["missing_snapshot_id"],
            "scores": {},
        }
    ]
    evidence = [
        {
            "unit_id": "50.1",
            "critical": [
                {
                    "span_id": "50.1_0",
                    "citation": "10 CFR 50.1",
                    "text": "The regulations in this part...",
                    "char_start": 0,
                    "char_end": 100,
                    "chunk_ids": [],
                    "tier": "critical",
                }
            ],
            "supporting": [],
            "contextual": [],
        }
    ]
    (tmp_path / "candidate_generation.jsonl").write_text(
        "\n".join(json.dumps(c) for c in candidates)
    )
    (tmp_path / "query_validation_results.jsonl").write_text(
        "\n".join(json.dumps(v) for v in validation)
    )
    (tmp_path / "evidence_tiers.jsonl").write_text("\n".join(json.dumps(e) for e in evidence))
    return tmp_path


def test_load_run_returns_one_record(run_dir: Path):
    records = load_run(run_dir)
    assert len(records) == 1


def test_load_run_joins_evidence(run_dir: Path):
    records = load_run(run_dir)
    rec = records[0]
    assert len(rec.critical_evidence) == 1
    assert rec.critical_evidence[0].span_id == "50.1_0"
    assert rec.critical_evidence[0].text == "The regulations in this part..."


def test_load_run_joins_validation(run_dir: Path):
    records = load_run(run_dir)
    rec = records[0]
    assert rec.is_valid is False
    assert "missing_snapshot_id" in rec.validation_flags


def test_load_run_defaults_review_status_to_pending(run_dir: Path):
    records = load_run(run_dir)
    assert records[0].review_status == ReviewStatus.PENDING


def test_load_run_merges_sidecar(run_dir: Path):
    sidecar = [
        {
            "candidate_id": "qc_50.1_citation_lookup_0",
            "review_status": "approved",
            "reviewed_by": "jsmith",
            "reviewed_at": "2026-03-25T10:00:00Z",
            "revision_notes": None,
            "rejection_note": None,
        }
    ]
    (run_dir / "review_state.jsonl").write_text("\n".join(json.dumps(s) for s in sidecar))
    records = load_run(run_dir)
    assert records[0].review_status == ReviewStatus.APPROVED
    assert records[0].reviewed_by == "jsmith"
