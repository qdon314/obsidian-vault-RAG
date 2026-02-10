from __future__ import annotations

from rag.adapters.ingestion.regulatory.cross_references import (
    extract_cross_references,
    rewrite_cross_references_to_wikilinks,
)


def test_extract_cross_references_handles_subsections() -> None:
    refs = extract_cross_references("See § 50.36(a) and 10 CFR §50.34.")
    assert refs == ["10 CFR §50.34", "10 CFR §50.36"]


def test_rewrite_cross_references_preserves_existing_wikilinks() -> None:
    text = "See § 50.36 and [[10 CFR §50.34]]."
    out = rewrite_cross_references_to_wikilinks(text)
    assert out.count("[[10 CFR §50.34]]") == 1
    assert "[[10 CFR §50.36]]" in out
