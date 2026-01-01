from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

def _looks_binary(data: bytes) -> bool:
    """
    Heuristic: if there are NUL bytes, or too many non-text bytes, treat as binary.
    """
    if not data:
        return False
    if b"\x00" in data:
        return True

    sample = data[:4096]
    # Count bytes that are "text-ish": tab/newline/cr + printable ASCII + common UTF-8 lead bytes are allowed,
    # but for a quick heuristic, just check for a high ratio of control bytes.
    control = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return (control / max(1, len(sample))) > 0.02  # Treating 2% control chars as suspicious


@dataclass(frozen=True, slots=True)
class TextLoader:
    """
    Loads a text file from disk.

    - Tries utf-8 first, falls back to latin-1 if needed.
    - Skips files larger than max_bytes (best-effort guardrail).
    """
    max_bytes: int = 2_000_000  # 2MB
    prefer_encoding: str = "utf-8"
    fallback_encoding: str = "latin-1"

    def load(self, path: Path) -> Optional[str]:
        try:
            size = path.stat().st_size
            if size > self.max_bytes:
                return None
        except OSError:
            return None

        try:
            data = path.read_bytes()
        except OSError:
            return None

        if _looks_binary(data):
            return None

        try:
            return data.decode(self.prefer_encoding, errors="strict")
        except UnicodeDecodeError:
            # Keep content visible but safe; "replace" is better than "ignore"
            try:
                return data.decode(self.fallback_encoding, errors="replace")
            except Exception:
                return None
