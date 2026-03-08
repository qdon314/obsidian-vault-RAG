# tests/eval/app_v2/engine/test_loader_base.py
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.base import LoadedArtifact


def test_loaded_artifact_no_warnings():
    a = LoadedArtifact(artifact_name="metrics.json", payload={"foo": 1}, warnings=())
    assert a.payload == {"foo": 1}
    assert a.warnings == ()


def test_loaded_artifact_with_warnings():
    w = BundleWarning(code=BundleWarningCode.SCHEMA_VERSION_UNKNOWN, message="unknown schema")
    a = LoadedArtifact(artifact_name="metrics.json", payload=None, warnings=(w,))
    assert len(a.warnings) == 1
