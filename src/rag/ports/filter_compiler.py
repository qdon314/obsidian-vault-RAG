from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from rag.domain.filters import Where

TFilter_co = TypeVar("TFilter_co", covariant=True)


class FilterCompiler(Protocol, Generic[TFilter_co]):
    def compile(self, where: Where) -> TFilter_co | None: ...
