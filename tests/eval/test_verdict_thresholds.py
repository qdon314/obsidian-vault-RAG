from __future__ import annotations

from pathlib import Path

from rag.eval.verdict_thresholds import load_verdict_thresholds


class TestLoadVerdictThresholds:
    def test_defaults_without_config(self, tmp_path: Path) -> None:
        thresholds = load_verdict_thresholds(tmp_path / "missing.toml")
        assert thresholds.min_recall_at_10 == 0.60

    def test_scoped_thresholds(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.toml"
        config.write_text(
            "[eval.verdict]\n"
            "min_recall_at_10 = 0.60\n"
            "\n"
            "[eval.verdict.regulatory]\n"
            "min_recall_at_10 = 0.70\n"
            "min_evidence_bounded_rate = 0.85\n",
            encoding="utf-8",
        )
        thresholds = load_verdict_thresholds(config, scope="regulatory")
        assert thresholds.min_recall_at_10 == 0.70
        assert thresholds.min_evidence_bounded_rate == 0.85
        assert thresholds.min_mrr == 0.40

    def test_no_scope_returns_base(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.toml"
        config.write_text(
            "[eval.verdict]\n"
            "min_recall_at_10 = 0.55\n"
            "\n"
            "[eval.verdict.regulatory]\n"
            "min_recall_at_10 = 0.70\n",
            encoding="utf-8",
        )
        thresholds = load_verdict_thresholds(config)
        assert thresholds.min_recall_at_10 == 0.55

    def test_missing_scope_returns_base(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.toml"
        config.write_text("[eval.verdict]\nmin_recall_at_10 = 0.55\n", encoding="utf-8")
        thresholds = load_verdict_thresholds(config, scope="regulatory")
        assert thresholds.min_recall_at_10 == 0.55
