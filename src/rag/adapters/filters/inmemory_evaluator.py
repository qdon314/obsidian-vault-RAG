from __future__ import annotations

from collections.abc import Mapping

from rag.domain.filters import And, Contains, Eq, Filter, In, Not, Or, Prefix, Range, Where
from rag.ports import FilterEvaluator


class InMemoryFilterEvaluator(FilterEvaluator):
    def matches(self, where: Where, metadata: Mapping[str, object]) -> bool:
        if where is None:
            return True
        return _matches(where, metadata)

def _matches(node: Filter, meta: Mapping[str, object]) -> bool:
    if isinstance(node, Eq):
        return meta.get(node.field) == node.value

    if isinstance(node, In):
        return meta.get(node.field) in set(node.values)

    if isinstance(node, Contains):
        v = meta.get(node.field)
        return isinstance(v, (list, tuple, set)) and node.value in v

    if isinstance(node, Prefix):
        v = meta.get(node.field)
        return isinstance(v, str) and v.startswith(node.prefix)

    if isinstance(node, Range):
        v = meta.get(node.field)
        if v is None:
            return False
        if node.gte is not None and not (v >= node.gte): return False  # noqa: E701
        if node.lte is not None and not (v <= node.lte): return False  # noqa: E701
        if node.gt  is not None and not (v >  node.gt ): return False  # noqa: E701
        return not (node.lt is not None and not v < node.lt)

    if isinstance(node, And):
        return all(_matches(c, meta) for c in node.clauses)

    if isinstance(node, Or):
        return any(_matches(c, meta) for c in node.clauses)

    if isinstance(node, Not):
        return not _matches(node.clause, meta)

    raise TypeError(f"Unknown filter node: {type(node)}")
