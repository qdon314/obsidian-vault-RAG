from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping


def json_sanitize(x: Any) -> Any:
    """
    Convert common non-JSON types into JSON-safe types.
    - datetime/date -> ISO string
    - Path -> str
    - set/tuple -> list
    - mappings/sequences -> recursively sanitized
    - unknown objects -> str(x)
    """
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, (datetime, date)):
        return x.isoformat()

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, set):
        return [json_sanitize(v) for v in sorted(x, key=lambda v: str(v))]

    if isinstance(x, tuple):
        return [json_sanitize(v) for v in x]

    if isinstance(x, list):
        return [json_sanitize(v) for v in x]

    if isinstance(x, Mapping):
        return {str(k): json_sanitize(v) for k, v in x.items()}

    # Fallback: stringify unknown objects (YAML tags, custom classes, etc.)
    return str(x)
