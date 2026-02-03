"""Environment variable override layer for settings.

Convention: RAG_<SECTION>__<KEY>=<value>
  - Prefix: RAG_
  - Double underscore separates section from key
  - Key is lowercased to match settings.toml field names
  - Values are coerced to match the existing type in the TOML dict

Known limitation: When the existing default is None, the override is kept
as a string. This may cause type mismatches downstream if the consuming
code expects a specific type. Long-term fix: use the Settings dataclass
field types as the source of truth for coercion.

Examples:
  RAG_VECTORSTORE__BACKEND=qdrant
  RAG_VECTORSTORE__QDRANT_URL=http://qdrant:6333
  RAG_RETRIEVAL__TOP_K=12
  RAG_RERANK__ENABLED=false
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_PREFIX = "RAG_"
_SEP = "__"


def _coerce(value: str, existing: Any, *, env_key: str = "") -> Any:
    """Coerce a string env value to match the type of the existing TOML value."""
    if existing is None:
        if env_key:
            log.warning(
                "Env override %s: existing default is None, keeping as string '%s'. "
                "Type coercion skipped.",
                env_key,
                value,
            )
        return value
    if isinstance(existing, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    return value


def apply_env_overrides(
    raw: dict[str, Any],
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply RAG_* environment variables as overrides to a parsed TOML dict.

    Parameters
    ----------
    raw:
        The parsed TOML dictionary (mutated in place and returned).
    environ:
        Environment dict to scan. Defaults to os.environ.
    """
    env = environ if environ is not None else dict(os.environ)

    for key, value in env.items():
        if not key.startswith(_PREFIX):
            continue

        remainder = key[len(_PREFIX):]
        if _SEP not in remainder:
            continue

        section, field = remainder.split(_SEP, 1)
        section = section.lower()
        field = field.lower()

        if section not in raw:
            raw[section] = {}

        existing = raw[section].get(field)
        raw[section][field] = _coerce(value, existing, env_key=key)

    return raw
