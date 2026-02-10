"""Citation manifest builder.

Scans a corpus directory of normalized markdown files and produces a JSON
manifest mapping each ``citation_key`` (e.g. ``"10 CFR §50.36"``) to the
relative file path that contains that section.  Downstream consumers use
this manifest for deterministic citation resolution without re-parsing
frontmatter at query time.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.adapters.ingestion.regulatory.normalizer import parse_frontmatter_citation_key


def build_citation_manifest(corpus_dir: Path) -> dict[str, str]:
    """Walk *corpus_dir* and build a ``citation_key → relative path`` mapping."""
    manifest: dict[str, str] = {}
    for md_file in sorted(corpus_dir.rglob("*.md")):
        citation_key = parse_frontmatter_citation_key(md_file)
        if citation_key:
            manifest[citation_key] = md_file.relative_to(corpus_dir).as_posix()
    return manifest


def save_citation_manifest(manifest: dict[str, str], path: Path) -> None:
    """Write *manifest* to *path* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
