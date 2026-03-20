# eval/app_v2/engine/loaders/traces.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.app_v2.engine.domain.models import QueryTrace
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.base import LoadedArtifact

logger = logging.getLogger(__name__)


def _extract_reranked_ids(row: dict) -> tuple[str, ...] | None:
    # Real schema: reranked = [{chunk: {chunk_id: ...}, score: ...}, ...]
    candidates = row.get("reranked")
    if candidates is None:
        return None
    ids = []
    for c in candidates:
        if isinstance(c, dict):
            chunk = c.get("chunk", {})
            chunk_id = chunk.get("chunk_id") if isinstance(chunk, dict) else None
            if chunk_id:
                ids.append(chunk_id)
    return tuple(ids) if ids else None


def _extract_packed_ids(row: dict) -> tuple[str, ...] | None:
    # Real schema: packed_chunk_ids = ["id1", "id2", ...]
    packed = row.get("packed_chunk_ids")
    if packed is None:
        return None
    return tuple(packed)


class TracesLoader:
    artifact_name = "traces.jsonl"

    def can_load(self, run_dir: Path) -> bool:
        return (run_dir / "traces.jsonl").exists()

    def load(self, run_dir: Path) -> LoadedArtifact:
        warnings: list[BundleWarning] = []
        traces: dict[str, QueryTrace] = {}
        path = run_dir / "traces.jsonl"
        for i, line in enumerate(path.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                tid = row.get("trace_id") or row.get("id")
                if not tid:
                    warnings.append(
                        BundleWarning(
                            code=BundleWarningCode.ORPHAN_TRACE,
                            message=f"Row {i} has no trace_id",
                            artifact_name=self.artifact_name,
                        )
                    )
                    continue
                traces[tid] = QueryTrace(
                    trace_id=tid,
                    reranked_chunk_ids=_extract_reranked_ids(row),
                    packed_chunk_ids=_extract_packed_ids(row),
                    raw_data=row,
                )
            except Exception as exc:
                warnings.append(
                    BundleWarning(
                        code=BundleWarningCode.PARTIAL_TRACE_PARSE,
                        message=f"Row {i} parse error: {exc}",
                        artifact_name=self.artifact_name,
                    )
                )
        return LoadedArtifact(
            artifact_name=self.artifact_name,
            payload=traces,
            warnings=tuple(warnings),
        )
