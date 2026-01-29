def _estimate_tokens(text: str) -> int:
    """
    Rough heuristic: ~4 chars/token for English-ish text.
    This is not exact; it's good enough to enforce a budget.
    If you later add tiktoken, swap this function.
    """
    return max(1, len(text) // 4)


def _normalize_for_dedupe(text: str) -> str:
    return " ".join(text.lower().split())