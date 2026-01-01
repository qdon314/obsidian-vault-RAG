from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from rag.domain.models import Document, IngestReport
from rag.adapters.ingestion.loaders.text_loader import TextLoader
from rag.adapters.ingestion.loaders.obsidian_markdown_loader import ObsidianMarkdownLoader


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _stable_doc_id(uri: str, text_hash: str) -> str:
    # Stable across runs; changes when file content changes.
    return sha256(f"{uri}|{text_hash}".encode("utf-8")).hexdigest()

def _iter_files(inputs: Sequence[str], *, recursive: bool) -> list[Path]:
    files: list[Path] = []

    for inp in inputs:
        p = Path(inp).expanduser()

        # Case 1: direct file or directory
        if p.exists():
            if p.is_dir():
                it = p.rglob("*") if recursive else p.glob("*")
                files.extend([x for x in it if x.is_file()])
            elif p.is_file():
                files.append(p)
            continue

        # Case 2: glob pattern (ONLY if relative)
        if p.is_absolute():
            # Absolute path that doesn't exist → skip safely
            continue

        for m in Path(".").glob(inp):
            if m.is_dir():
                it = m.rglob("*") if recursive else m.glob("*")
                files.extend([x for x in it if x.is_file()])
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

    text_loader: TextLoader = field(default_factory=TextLoader)
    markdown_loader: Optional[ObsidianMarkdownLoader] = None

    def ingest(
        self,
        inputs: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Tuple[list[Document], IngestReport]:
        base_meta = dict(metadata) if metadata else {}

        scanned = loaded = 0
        skipped_hidden = skipped_extension = skipped_too_large = skipped_empty = failed = 0
        by_ext: dict[str, int] = {}

        docs: list[Document] = []
        files = _iter_files(inputs, recursive=self.recursive)

        for path in files:
            scanned += 1

            try:
                if self.skip_hidden and _is_hidden(path):
                    skipped_hidden += 1
                    continue

                ext = path.suffix.lower()
                if self.allowed_extensions and ext not in self.allowed_extensions:
                    skipped_extension += 1
                    continue

                uri = str(path)

                # Load content + extra md metadata if applicable
                md_meta: dict[str, object] = {}
                if ext == ".md" and self.markdown_loader is not None:
                    loaded_md = self.markdown_loader.load(path)
                    if loaded_md is None:
                        skipped_empty += 1
                        continue
                    text, md_meta = loaded_md
                else:
                    # Let TextLoader enforce max_bytes; if it returns None treat as too large/unreadable
                    text = self.text_loader.load(path)
                    if text is None:
                        skipped_too_large += 1
                        continue

                if not text or not text.strip():
                    skipped_empty += 1
                    continue

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

            except Exception:
                failed += 1
                continue

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
