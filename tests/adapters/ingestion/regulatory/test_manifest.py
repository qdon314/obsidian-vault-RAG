from __future__ import annotations

import json
from pathlib import Path

from rag.adapters.ingestion.regulatory.manifest import (
    build_citation_manifest,
    save_citation_manifest,
)


def test_build_citation_manifest_maps_citation_key_to_relative_path(tmp_path: Path) -> None:
    part_dir = tmp_path / "part-50"
    part_dir.mkdir()
    (part_dir / "50.34.md").write_text(
        '---\ncitation_key: "10 CFR §50.34"\n---\n# Content\n',
        encoding="utf-8",
    )
    manifest = build_citation_manifest(tmp_path)
    assert manifest == {"10 CFR §50.34": "part-50/50.34.md"}


def test_save_citation_manifest_writes_json(tmp_path: Path) -> None:
    path = tmp_path / "citation_manifest.json"
    save_citation_manifest({"10 CFR §50.34": "part-50/50.34.md"}, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["10 CFR §50.34"] == "part-50/50.34.md"
