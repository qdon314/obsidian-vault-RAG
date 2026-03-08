# eval/app_v2/engine/domain/warnings.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BundleWarningCode(StrEnum):
    MISSING_TRACES           = "missing_traces"
    MISSING_VERDICT          = "missing_verdict"
    PARTIAL_TRACE_PARSE      = "partial_trace_parse"
    PARTIAL_RESULTS_PARSE    = "partial_results_parse"
    SCHEMA_VERSION_UNKNOWN   = "schema_version_unknown"
    TRACE_TEXT_REDACTED      = "trace_text_redacted"
    ORPHAN_TRACE             = "orphan_trace"
    MISSING_TRACE_FOR_RESULT = "missing_trace_for_result"


@dataclass(frozen=True, slots=True)
class BundleWarning:
    code: BundleWarningCode
    message: str
    artifact_name: str | None = None
