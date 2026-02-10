"""Batch enrichment entry point for regulatory chunks.

This is the top-level function called by the ingestion pipeline after
chunking.  It delegates to ``metadata.enrich_regulatory_chunk_metadata``
for each chunk, using ``dataclasses.replace`` to produce new immutable
``Chunk`` objects with enriched metadata.
"""

from __future__ import annotations

from dataclasses import replace

from rag.adapters.ingestion.regulatory.metadata import enrich_regulatory_chunk_metadata
from rag.domain.models import Chunk


def enrich_regulatory_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Return a new list of chunks with regulatory citation metadata stamped.

    Non-regulatory chunks (``corpus != "regulatory"``) pass through unchanged.
    """
    enriched: list[Chunk] = []
    for chunk in chunks:
        metadata = enrich_regulatory_chunk_metadata(
            chunk.metadata,
            section_heading=chunk.section_heading,
            section_path=chunk.section_path,
        )
        enriched.append(replace(chunk, metadata=metadata))
    return enriched
