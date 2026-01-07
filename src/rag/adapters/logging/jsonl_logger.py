from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.domain.models import QueryTrace
from rag.ports import QueryLogger


def _json_default(o: Any):
    if is_dataclass(o):
        return asdict(o) # type: ignore
    return str(o)

class JsonlQueryLogger(QueryLogger):
    def __init__(self, path: str | Path, *, redact_text: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redact_text = redact_text

    def new_trace_id(self) -> str:
        return uuid.uuid4().hex

    def now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def log(self, trace: QueryTrace) -> None:
        payload = asdict(trace)

        if self.redact_text:
            payload = _redact(payload)

        line = json.dumps(payload, ensure_ascii=False, default=_json_default)

        # atomic append on POSIX (good enough for single-process; for multi-proc add a file lock)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {"text", "page_content", "chunk_text", "context_text", "answer"}:
                out[k] = "[REDACTED]"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj
