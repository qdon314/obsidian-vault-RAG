"""Pre-processing for citation extraction: OCR cleanup and CFR normalization.

Improves regex recall by standardizing common surface variations of
regulatory references before span extraction runs.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_for_citation_extraction(text: str) -> str:
    """Normalize *text* for citation extraction.

    Transformations (in order):
    1. Unicode normalization (NFC)
    2. Replace smart quotes, em-dashes, en-dashes with ASCII equivalents
    3. Normalize "C.F.R." / "C F R" → "CFR"
    4. Normalize "Code of Federal Regulations" → "CFR"
    5. Collapse single newlines to spaces (preserve paragraph breaks)
    6. Collapse runs of spaces to single space
    """
    if not text:
        return ""

    s = unicodedata.normalize("NFC", text)

    # Smart quotes → ASCII
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")

    # Em-dash / en-dash → hyphen
    s = s.replace("\u2013", "-").replace("\u2014", "-")

    # "Code of Federal Regulations" → CFR  (with optional Title N prefix)
    s = re.sub(
        r"Title\s+(\d+)\s*,?\s*Code\s+of\s+Federal\s+Regulations\s*,?\s*(?:Section\s+)?",
        r"\1 CFR ",
        s,
        flags=re.IGNORECASE,
    )

    # "C.F.R." → "CFR"
    s = re.sub(r"C\s*\.\s*F\s*\.\s*R\s*\.?", "CFR", s)

    # "C F R" (OCR split) → "CFR"
    s = re.sub(r"\bC\s+F\s+R\b", "CFR", s)

    # Collapse single newlines to spaces (preserve double-newline paragraph breaks)
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)

    # Collapse runs of whitespace (except newlines) to single space
    s = re.sub(r"[^\S\n]+", " ", s)

    return s.strip()
