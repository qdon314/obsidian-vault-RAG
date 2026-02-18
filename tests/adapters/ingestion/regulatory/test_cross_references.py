from __future__ import annotations

from rag.adapters.ingestion.regulatory.cross_references import (
    extract_cross_references,
    rewrite_cross_references_to_wikilinks,
)

# --- extract_cross_references ------------------------------------------------


def test_extract_cross_references_handles_subsections() -> None:
    refs = extract_cross_references("See § 50.36(a) and 10 CFR §50.34.")
    assert refs == ["10 CFR §50.34", "10 CFR §50.36"]


def test_extract_handles_title_cfr_without_section_symbol() -> None:
    """``10 CFR 50.46`` (no §) should be extracted — common in case documents."""
    refs = extract_cross_references("Requirements of 10 CFR 50.46 apply.")
    assert refs == ["10 CFR §50.46"]


def test_extract_handles_mixed_citation_styles() -> None:
    """Mix of §-prefixed and title-prefixed forms in one passage."""
    text = "See 10 CFR 50.46 and §50.55a. Also 10 CFR §50.34 applies."
    refs = extract_cross_references(text)
    assert refs == ["10 CFR §50.34", "10 CFR §50.46", "10 CFR §50.55a"]


def test_extract_bare_number_not_matched() -> None:
    """Bare ``50.46`` without title or § must NOT be matched (false-positive guard)."""
    refs = extract_cross_references("The value 50.46 was recorded.")
    assert refs == []


# --- rewrite_cross_references_to_wikilinks -----------------------------------


def test_rewrite_cross_references_preserves_existing_wikilinks() -> None:
    text = "See § 50.36 and [[10 CFR §50.34]]."
    out = rewrite_cross_references_to_wikilinks(text)
    assert out.count("[[10 CFR §50.34]]") == 1
    assert "[[10 CFR §50.36]]" in out


def test_rewrite_title_cfr_without_section_symbol() -> None:
    """``10 CFR 50.46`` should be rewritten to ``[[10 CFR §50.46]]``."""
    out = rewrite_cross_references_to_wikilinks("Requirements of 10 CFR 50.46 apply.")
    assert "[[10 CFR §50.46]]" in out


def test_rewrite_title_cfr_no_spaces() -> None:
    """``10CFR50.46`` (no spaces) should also be rewritten."""
    out = rewrite_cross_references_to_wikilinks("See 10CFR50.46 details.")
    assert "[[10 CFR §50.46]]" in out


def test_rewrite_bare_number_not_matched() -> None:
    """Bare ``50.46`` must NOT be rewritten — avoid false positives."""
    text = "The value 50.46 was recorded."
    out = rewrite_cross_references_to_wikilinks(text)
    assert "[[" not in out
    assert out == text


def test_rewrite_title_cfr_with_section_symbol() -> None:
    """``10 CFR §50.46`` (with §) should still work as before."""
    out = rewrite_cross_references_to_wikilinks("See 10 CFR §50.46 and §50.55a.")
    assert "[[10 CFR §50.46]]" in out
    assert "[[10 CFR §50.55a]]" in out


def test_rewrite_with_subsections() -> None:
    """Subsections should be included in the match but stripped in the canonical key."""
    out = rewrite_cross_references_to_wikilinks("Per 10 CFR 50.46(b)(1) requirements.")
    assert "[[10 CFR §50.46]]" in out


def test_rewrite_idempotent() -> None:
    """Running rewrite twice on the same text should not double-wrap."""
    text = "See 10 CFR 50.46 and §50.55a."
    first_pass = rewrite_cross_references_to_wikilinks(text)
    second_pass = rewrite_cross_references_to_wikilinks(first_pass)
    assert first_pass == second_pass


def test_rewrite_different_titles() -> None:
    """Non-10 CFR titles should be preserved."""
    out = rewrite_cross_references_to_wikilinks("Per 40 CFR 61.92 limits.")
    assert "[[40 CFR §61.92]]" in out


# --- Known limitations -------------------------------------------------------


def test_plural_section_symbol_only_matches_first() -> None:
    """``§§ A and B`` is standard shorthand for two sections, but only the
    first is currently matched — the second lacks a § or title prefix.
    """
    refs = extract_cross_references("§§ 50.46 and 50.55")
    assert refs == ["10 CFR §50.46"]  # 50.55 not captured


def test_decimal_before_cfr_matches_fractional_title() -> None:
    """``2.5 CFR 50.46`` matches title=5 (the integer after the dot).

    This is a known limitation — CFR titles are always whole numbers, so
    this pattern does not occur in real NRC documents.
    """
    refs = extract_cross_references("ratio 2.5 CFR 50.46")
    assert refs == ["5 CFR §50.46"]


# --- Cross-corpus linking validation -----------------------------------------


def test_case_and_regulatory_wikilinks_use_same_canonical_format() -> None:
    """Wikilinks produced from case-style citations (``10 CFR 50.46``) and
    regulatory-style citations (``§50.46``) must produce identical canonical
    keys so Obsidian links resolve correctly across corpora.
    """
    case_style = rewrite_cross_references_to_wikilinks("10 CFR 50.46")
    reg_style = rewrite_cross_references_to_wikilinks("§50.46")
    assert case_style == reg_style == "[[10 CFR §50.46]]"

    # Extraction should also unify
    case_refs = extract_cross_references("10 CFR 50.46")
    reg_refs = extract_cross_references("§50.46")
    assert case_refs == reg_refs == ["10 CFR §50.46"]
