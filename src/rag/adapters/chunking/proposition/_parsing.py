from __future__ import annotations

import json
import re

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _parse_json_list_loose(text: str) -> list[str]:
    """
    Best-effort parse of a JSON list from a model response.

    We accept:
      - a perfect JSON array
      - JSON array embedded in other text
    """
    text = text.strip()

    # 1) direct
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass

    # 2) find an array substring
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass

    return []


def _extract_json_list(text: str) -> list[str]:
    """
    Parse the propositionizer output into a list of proposition strings.

    The seq2seq model is expected to produce a JSON list of strings, e.g.
    ``["Proposition one.", "Proposition two."]``.  However, the model output
    can contain stray text around the JSON array.  This function applies
    two parsing strategies:

    1. Try ``json.loads`` on the full output.
    2. If that fails, locate the outermost ``[...]`` substring and parse that.

    Returns an empty list if neither strategy succeeds.
    """
    text = text.strip()
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass

    left = text.find("[")
    right = text.rfind("]")
    if 0 <= left < right:
        try:
            val = json.loads(text[left : right + 1])
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass

    return []
