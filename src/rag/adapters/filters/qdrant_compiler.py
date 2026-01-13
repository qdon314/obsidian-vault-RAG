from __future__ import annotations

from qdrant_client.models import (
    FieldCondition,
    MatchAny,
    MatchText,
    MatchValue,
)
from qdrant_client.models import (
    Filter as QdrantFilter,
)
from qdrant_client.models import (
    Range as QdrantRange,
)

from rag.domain.filters import And, Contains, Eq, Filter, In, Not, Or, Prefix, Range, Where
from rag.ports.filter_compiler import FilterCompiler


class QdrantFilterCompiler(FilterCompiler[QdrantFilter]):
    def compile(self, where: Where) -> QdrantFilter | None:
        if where is None:
            return None
        return _compile_filter(where)




def _compile_filter(f: Filter) -> QdrantFilter:
    match f:
        case And(clauses=clauses):
            # each clause becomes a *filter*; then we embed those as conditions via "must"
            return QdrantFilter(must=[_compile_as_condition(c) for c in clauses])

        case Or(clauses=clauses):
            return QdrantFilter(should=[_compile_as_condition(c) for c in clauses])

        case Not(clause=clause):
            return QdrantFilter(must_not=[_compile_as_condition(clause)])

        case _:
            # leaf or unknown: wrap as must condition
            return QdrantFilter(must=[_compile_condition(f)])


def _compile_as_condition(f: Filter):
    """
    Qdrant 'must/should/must_not' accepts 'Condition' (union type).
    A nested Filter is also a condition in Qdrant's model (client supports this).
    If your client version doesn't accept nested filters here, you'd need to normalize.
    """
    # If it's a boolean node, compile to a Filter (nested)
    if isinstance(f, (And, Or, Not)):
        return _compile_filter(f)
    return _compile_condition(f)


def _compile_condition(f: Filter) -> FieldCondition:
    match f:
        case Eq(field=fld, value=val):
            return FieldCondition(key=fld, match=MatchValue(value=val))

        case In(field=fld, values=vals):
            return FieldCondition(key=fld, match=MatchAny(any=list(vals)))

        case Contains(field=fld, value=val):
            # semantic "array contains value" (recommended)
            return FieldCondition(key=fld, match=MatchValue(value=val))

        case Range(field=fld, gte=gte, lte=lte, gt=gt, lt=lt):
            return FieldCondition(key=fld, range=QdrantRange(gte=gte, lte=lte, gt=gt, lt=lt))

        case Prefix(field=fld, prefix=pfx):
            return FieldCondition(key=fld, match=MatchText(text=pfx))

        case _:
            raise ValueError(f"Unsupported filter type: {type(f)}")
