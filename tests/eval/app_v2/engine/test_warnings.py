# tests/eval/app_v2/engine/test_warnings.py
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode


def test_bundle_warning_is_frozen():
    w = BundleWarning(code=BundleWarningCode.MISSING_TRACES, message="no traces")
    import dataclasses
    assert dataclasses.is_dataclass(w)
    try:
        w.message = "changed"  # type: ignore
        raise AssertionError("should be frozen")
    except (AttributeError, TypeError):
        pass


def test_bundle_warning_optional_artifact():
    w = BundleWarning(code=BundleWarningCode.ORPHAN_TRACE, message="orphan", artifact_name="traces.jsonl")
    assert w.artifact_name == "traces.jsonl"

    w2 = BundleWarning(code=BundleWarningCode.MISSING_VERDICT, message="no verdict")
    assert w2.artifact_name is None
