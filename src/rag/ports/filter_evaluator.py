from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from rag.domain.filters import Where


class FilterEvaluator(Protocol):
    def matches(self, where: Where, metadata: Mapping[str, object]) -> bool:
        ...
