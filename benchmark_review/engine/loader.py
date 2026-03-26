from __future__ import annotations

import json
from pathlib import Path

from benchmark_review.engine.models import EvidenceSpan, ReviewRecord, ReviewStatus


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _parse_evidence_spans(spans: list[dict], tier: str) -> tuple[EvidenceSpan, ...]:
    return tuple(
        EvidenceSpan(
            span_id=s["span_id"],
            citation=s["citation"],
            text=s["text"],
            char_start=s["char_start"],
            char_end=s["char_end"],
            tier=tier,
        )
        for s in spans
    )


def load_run(run_dir: Path) -> list[ReviewRecord]:
    candidates = _read_jsonl(run_dir / "candidate_generation.jsonl")
    validations = {
        v["candidate_id"]: v for v in _read_jsonl(run_dir / "query_validation_results.jsonl")
    }
    evidence_by_unit = {e["unit_id"]: e for e in _read_jsonl(run_dir / "evidence_tiers.jsonl")}
    sidecar = {s["candidate_id"]: s for s in _read_jsonl(run_dir / "review_state.jsonl")}

    records: list[ReviewRecord] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        uid = cand["unit_id"]
        val = validations.get(cid, {})
        ev = evidence_by_unit.get(uid, {})
        side = sidecar.get(cid, {})

        rec = ReviewRecord(
            candidate_id=cid,
            unit_id=uid,
            query=cand["query"],
            query_class=cand["query_class"],
            difficulty=cand.get("difficulty", "easy"),
            source_citations=tuple(cand.get("source_citations", [])),
            evidence_span_ids=tuple(cand.get("evidence_span_ids", [])),
            is_valid=val.get("is_valid", False),
            validation_flags=tuple(val.get("flags", [])),
            critical_evidence=_parse_evidence_spans(ev.get("critical", []), "critical"),
            supporting_evidence=_parse_evidence_spans(ev.get("supporting", []), "supporting"),
            contextual_evidence=_parse_evidence_spans(ev.get("contextual", []), "contextual"),
            is_unanswerable=cand.get("metadata", {}).get("is_unanswerable", False),
            unanswerable_reason=cand.get("metadata", {}).get("unanswerable_reason"),
            review_status=ReviewStatus(side["review_status"])
            if "review_status" in side
            else ReviewStatus.PENDING,
            reviewed_by=side.get("reviewed_by"),
            reviewed_at=side.get("reviewed_at"),
            revision_notes=side.get("revision_notes"),
            rejection_note=side.get("rejection_note"),
        )
        records.append(rec)

    return records
