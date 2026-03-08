# eval/app_v2/engine/loaders/registry.py
from __future__ import annotations

from eval.app_v2.engine.loaders.base import ArtifactLoader
from eval.app_v2.engine.loaders.metrics import MetricsLoader
from eval.app_v2.engine.loaders.results import ResultsLoader
from eval.app_v2.engine.loaders.traces import TracesLoader
from eval.app_v2.engine.loaders.verdict import VerdictLoader

DEFAULT_LOADERS: tuple[ArtifactLoader, ...] = (
    MetricsLoader(),
    ResultsLoader(),
    TracesLoader(),
    VerdictLoader(),
)
