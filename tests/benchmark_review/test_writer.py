import json
from pathlib import Path

from benchmark_review.engine.models import ReviewStatus
from benchmark_review.engine.writer import save_decision


def test_save_decision_creates_sidecar(tmp_path: Path):
    save_decision(
        run_dir=tmp_path,
        candidate_id="qc_50.1_cit_0",
        status=ReviewStatus.APPROVED,
        reviewed_by="jsmith",
        revision_notes=None,
        rejection_note=None,
    )
    sidecar = tmp_path / "review_state.jsonl"
    assert sidecar.exists()
    record = json.loads(sidecar.read_text().strip())
    assert record["review_status"] == "approved"
    assert record["reviewed_by"] == "jsmith"
    assert "reviewed_at" in record


def test_save_decision_overwrites_existing_entry(tmp_path: Path):
    for status in [ReviewStatus.NEEDS_REVISION, ReviewStatus.APPROVED]:
        save_decision(
            run_dir=tmp_path,
            candidate_id="qc_50.1_cit_0",
            status=status,
            reviewed_by="jsmith",
            revision_notes="fix the citation" if status == ReviewStatus.NEEDS_REVISION else None,
            rejection_note=None,
        )
    lines = (tmp_path / "review_state.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1  # deduped, not appended
    assert json.loads(lines[0])["review_status"] == "approved"


def test_save_decision_preserves_other_entries(tmp_path: Path):
    save_decision(tmp_path, "qc_a", ReviewStatus.APPROVED, "jsmith", None, None)
    save_decision(tmp_path, "qc_b", ReviewStatus.REJECTED, "jsmith", None, "duplicate")
    lines = (tmp_path / "review_state.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
