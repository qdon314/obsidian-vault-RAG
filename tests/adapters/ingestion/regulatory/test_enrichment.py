from __future__ import annotations

from rag.adapters.ingestion.regulatory.enrichment import enrich_regulatory_chunks
from tests.conftest import make_chunk


def test_enrich_regulatory_chunks_uses_section_path_for_specific_citation() -> None:
    chunk = make_chunk(
        section_heading="(1)",
        section_path="(a) > (1)",
        metadata={"corpus": "regulatory", "citation_key": "10 CFR §50.34"},
    )
    enriched = enrich_regulatory_chunks([chunk])
    assert enriched[0].metadata["citation"] == "10 CFR §50.34(a)(1)"
