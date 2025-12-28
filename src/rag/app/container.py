from __future__ import annotations

from dataclasses import dataclass

from rag.adapters.chunking.fixed import FixedChunker
from rag.ports import Chunker


@dataclass(frozen=True, slots=True)
class Container:
    """
    Lightweight dependency container.
    Holds instances of adapters implementing various ports.
    """
    chunker: Chunker


def build_container() -> Container:
    # Later: read these from config/env
    chunker = FixedChunker(chunk_size=1200, overlap=150)
    return Container(chunker=chunker)
