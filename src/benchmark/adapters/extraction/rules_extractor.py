"""Stage 1a: Deterministic structural segmentation.

Groups ``BenchmarkSourceSpan`` records by ``(parent_section_id, subsection_chain)``
and mints stable ``RegulatoryUnit`` records.  No LLM involvement — semantic
classification is deferred to Stage 1b.
"""

from __future__ import annotations

from collections import defaultdict

from benchmark.domain.enums import UnitKind
from benchmark.domain.models import BenchmarkSourceSpan, RegulatoryUnit


def _mint_unit_id(section_id: str, subsection_chain: tuple[str, ...]) -> str:
    """Mint a stable unit ID from section number and subsection chain."""
    parts = [section_id]
    parts.extend(subsection_chain)
    return "_".join(parts)


class RulesExtractor:
    """Deterministic regulatory unit extractor (Stage 1a).

    Groups spans by ``(parent_section_id, subsection_chain)`` and assigns
    a preliminary ``UnitKind`` based on structural cues:

    - Spans with cross-references → ``CROSS_REFERENCE``
    - All others → ``OBLIGATION`` (refined by LLM in Stage 1b)
    """

    def extract(self, spans: list[BenchmarkSourceSpan]) -> list[RegulatoryUnit]:
        # Group spans by (section_id, subsection_chain).
        groups: dict[tuple[str, tuple[str, ...]], list[BenchmarkSourceSpan]] = defaultdict(list)
        for span in spans:
            subsection = tuple(span.metadata.get("subsection_tokens", ()))
            key = (span.parent_section_id, subsection)
            groups[key].append(span)

        units: list[RegulatoryUnit] = []
        for (section_id, subsection_chain), group_spans in groups.items():
            # Collect cross-references from span metadata.
            cross_refs: list[str] = []
            for s in group_spans:
                cross_refs.extend(s.metadata.get("cross_references", ()))
            cross_refs_deduped = tuple(dict.fromkeys(cross_refs))

            kind = UnitKind.CROSS_REFERENCE if cross_refs_deduped else UnitKind.OBLIGATION

            units.append(
                RegulatoryUnit(
                    unit_id=_mint_unit_id(section_id, subsection_chain),
                    kind=kind,
                    spans=tuple(group_spans),
                    citation=group_spans[0].citation,
                    subsection_chain=subsection_chain,
                    parent_section_id=section_id,
                    corpus_snapshot_id=group_spans[0].corpus_snapshot_id,
                    cross_references=cross_refs_deduped,
                )
            )

        return units
