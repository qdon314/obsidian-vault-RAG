from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_review.engine.models import ReviewStatus


def save_decision(
    run_dir: Path,
    candidate_id: str,
    status: ReviewStatus,
    reviewed_by: str,
    revision_notes: str | None,
    rejection_note: str | None,
) -> None:
    sidecar = run_dir / "review_state.jsonl"

    # Load existing entries (keyed by candidate_id for dedup)
    existing: dict[str, dict] = {}
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                existing[entry["candidate_id"]] = entry

    existing[candidate_id] = {
        "candidate_id": candidate_id,
        "review_status": status.value,
        "reviewed_by": reviewed_by,
        "reviewed_at": datetime.now(UTC).isoformat(),
        "revision_notes": revision_notes,
        "rejection_note": rejection_note,
    }

    sidecar.write_text(
        "\n".join(json.dumps(e) for e in existing.values()) + "\n"
    )
