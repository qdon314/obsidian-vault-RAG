"""Validate an index manifest against the runtime environment."""

from __future__ import annotations

import logging

from rag.domain.errors import IndexIncompatibleError
from rag.domain.index_manifest import IndexManifest
from rag.ports import Embedder

log = logging.getLogger(__name__)


def validate_index(manifest: IndexManifest, embedder: Embedder) -> None:
    """Raise IndexIncompatibleError if runtime config is incompatible with index.

    Checks:
        1. Index build completed successfully.
        2. Embedding model matches between index and runtime.
    """
    if manifest.status != "complete":
        raise IndexIncompatibleError(
            f"Index '{manifest.index_name}' has status '{manifest.status}' (expected 'complete')"
        )

    index_model = manifest.embedding.get("model")
    if index_model and index_model != embedder.model_name:
        raise IndexIncompatibleError(
            f"Index '{manifest.index_name}' was built with embedding model "
            f"'{index_model}', but runtime embedder is '{embedder.model_name}'"
        )

    log.debug(
        "Index '%s' (id=%s, build=%s) validated OK",
        manifest.index_name,
        manifest.index_id,
        manifest.build_id,
    )
