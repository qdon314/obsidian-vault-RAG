from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from rag.adapters.ingestion.loaders.obsidian_markdown_loader import ObsidianMarkdownLoader
from rag.adapters.ingestion.loaders.text_loader import TextLoader
from rag.domain.models import Document, IngestReport

log = logging.getLogger(__name__)


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _stable_doc_id(uri: str, text_hash: str) -> str:
    # Stable across runs; changes when file content changes.
    return sha256(f"{uri}|{text_hash}".encode()).hexdigest()


def _flatten_frontmatter(md_meta: Mapping[str, Any]) -> dict[str, Any]:
    frontmatter = md_meta.get("frontmatter")
    if not isinstance(frontmatter, Mapping):
        return {}
    flattened: dict[str, Any] = {}
    for key, value in frontmatter.items():
        if isinstance(key, str):
            flattened[key] = value
    return flattened


def _walk_files(root: Path) -> list[Path]:
    """Recursively collect files, skipping hidden directories."""
    result: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune hidden directories in-place so os.walk won't descend into them
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            result.append(Path(dirpath) / fn)
    return result


def _iter_files(inputs: Sequence[str], *, recursive: bool) -> list[Path]:
    files: list[Path] = []

    for inp in inputs:
        p = Path(inp).expanduser()

        # Case 1: direct file or directory
        if p.exists():
            if p.is_dir():
                if recursive:
                    files.extend(_walk_files(p))
                else:
                    files.extend([x for x in p.glob("*") if x.is_file()])
            elif p.is_file():
                files.append(p)
            continue

        # Case 2: glob pattern (ONLY if relative)
        if p.is_absolute():
            # Absolute path that doesn't exist -> skip safely
            continue

        for m in Path(".").glob(inp):
            if m.is_dir():
                if recursive:
                    files.extend(_walk_files(m))
                else:
                    files.extend([x for x in m.glob("*") if x.is_file()])
            elif m.is_file():
                files.append(m)

    # Stable, deterministic ordering
    return sorted({f.resolve() for f in files}, key=lambda x: str(x))


@dataclass(frozen=True, slots=True)
class FilesystemIngestor:
    allowed_extensions: set[str] = field(
        default_factory=lambda: {".md", ".txt", ".py", ".json", ".yaml", ".yml"}
    )
    recursive: bool = True
    skip_hidden: bool = True
    source_name: str = "filesystem"

    # injected
    text_loader: TextLoader = field(default_factory=TextLoader)
    markdown_loader: ObsidianMarkdownLoader | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_one(self, path: Path, ext: str) -> tuple[str, dict[str, Any]] | None:
        """Load a single file. Thread-safe (read-only shared state)."""
        if ext == ".md" and self.markdown_loader is not None:
            return self.markdown_loader.load(path)
        text = self.text_loader.load(path)
        if text is None:
            return None
        return text, {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        inputs: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[list[Document], IngestReport]:
        base_meta = dict(metadata) if metadata else {}

        scanned = loaded = 0
        skipped_hidden = skipped_extension = skipped_too_large = skipped_empty = failed = 0
        by_ext: dict[str, int] = {}

        docs: list[Document] = []

        t0 = perf_counter()
        files = _iter_files(inputs, recursive=self.recursive)
        discovery_dt = perf_counter() - t0
        log.info("File discovery: %d files in %.2fs", len(files), discovery_dt)

        # --- Phase 1: filter ---
        eligible: list[tuple[Path, str]] = []
        for path in files:
            scanned += 1
            if self.skip_hidden and _is_hidden(path):
                skipped_hidden += 1
                continue
            ext = path.suffix.lower()
            if self.allowed_extensions and ext not in self.allowed_extensions:
                skipped_extension += 1
                continue
            eligible.append((path, ext))

        # --- Phase 2: parallel load ---
        t_load = perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                (path, ext, pool.submit(self._load_one, path, ext)) for path, ext in eligible
            ]
        load_dt = perf_counter() - t_load

        # --- Phase 3: process results ---
        for path, ext, future in futures:
            try:
                result = future.result()
            except Exception:
                failed += 1
                continue

            if result is None:
                if ext == ".md":
                    skipped_empty += 1
                else:
                    skipped_too_large += 1
                continue

            text, md_meta = result

            if not text or not text.strip():
                skipped_empty += 1
                continue

            uri = str(path)
            text_hash = _hash_text(text)
            doc_id = _stable_doc_id(uri, text_hash)

            try:
                stat = path.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
                size = stat.st_size
            except OSError:
                mtime = None
                size = None

            doc_meta = {
                **base_meta,
                "uri": uri,
                "title": path.name,
                "ext": ext,
                "mtime": mtime,
                "size_bytes": size,
                "content_hash": text_hash,
                **md_meta,
                **_flatten_frontmatter(md_meta),
            }

            docs.append(
                Document(
                    doc_id=doc_id,
                    text=text,
                    source=self.source_name,
                    uri=uri,
                    metadata=doc_meta,
                )
            )
            loaded += 1
            by_ext[ext] = by_ext.get(ext, 0) + 1

        log.info(
            "Ingest breakdown: discovery=%.2fs, loading=%.2fs, other=%.2fs",
            discovery_dt,
            load_dt,
            (perf_counter() - t0) - discovery_dt - load_dt,
        )

        report = IngestReport(
            scanned=scanned,
            loaded=loaded,
            skipped_hidden=skipped_hidden,
            skipped_extension=skipped_extension,
            skipped_too_large=skipped_too_large,
            skipped_empty=skipped_empty,
            failed=failed,
            by_extension=dict(by_ext),
        )
        return docs, report
