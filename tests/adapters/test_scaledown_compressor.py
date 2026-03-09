"""Tests for ScaleDownCompressor response parsing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag.adapters.compression.scaledown import ScaleDownCompressor
from rag.domain.models import ContextPack


def _make_pack(text: str = "CONTEXT:\n[1]\nsome chunk text\n") -> ContextPack:
    return ContextPack(
        query="test query",
        chunks=(),
        rendered_context=text,
        citations=(),
        token_budget=2000,
        metadata={"tokens_used_est": 852},
    )


SAMPLE_API_RESPONSE = {
    "results": {
        "success": True,
        "compressed_prompt": "CONTEXT:\n[1]\ncompressed chunk text\n",
        "original_prompt": "test query",
        "original_prompt_tokens": 852,
        "compressed_prompt_tokens": 794,
        "compression_ratio": 0.9319,
    },
    "model_used": "gpt-4-turbo",
    "total_original_tokens": 852,
    "total_compressed_tokens": 794,
    "num_pairs_processed": 1,
    "successful": True,
    "latency_ms": 367,
}


class TestScaleDownCompressorParsing:
    def _compressor(self) -> ScaleDownCompressor:
        return ScaleDownCompressor(api_key="test-key")

    def _mock_response(self, data: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = data
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_reads_compressed_text_from_results(self):
        """compressed_prompt is read from response["results"], not top-level."""
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", return_value=self._mock_response(SAMPLE_API_RESPONSE)):
            result = compressor.compress(pack, query="test query")

        assert result.successful is True
        assert result.context_pack.rendered_context == "CONTEXT:\n[1]\ncompressed chunk text\n"

    def test_uses_api_token_counts(self):
        """tokens_before and tokens_after come from the API response."""
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", return_value=self._mock_response(SAMPLE_API_RESPONSE)):
            result = compressor.compress(pack, query="test query")

        assert result.tokens_before == 852
        assert result.tokens_after == 794

    def test_savings_pct_derived_from_compression_ratio(self):
        """savings_pct = 1 - compression_ratio."""
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", return_value=self._mock_response(SAMPLE_API_RESPONSE)):
            result = compressor.compress(pack, query="test query")

        expected_savings = 1.0 - 0.9319
        assert abs(result.savings_pct - expected_savings) < 1e-4

    def test_compression_ratio_in_extra(self):
        """compression_ratio is surfaced in extra for observability."""
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", return_value=self._mock_response(SAMPLE_API_RESPONSE)):
            result = compressor.compress(pack, query="test query")

        assert "compression_ratio" in result.extra
        assert abs(result.extra["compression_ratio"] - 0.9319) < 1e-4

    def test_missing_results_key_falls_back_to_defaults(self):
        """When API response lacks 'results', compressed_text is empty and tokens unchanged."""
        compressor = self._compressor()
        pack = _make_pack()
        empty_response = {}  # no "results" key at all

        with patch("httpx.post", return_value=self._mock_response(empty_response)):
            result = compressor.compress(pack, query="test query")

        # Falls back to safe defaults — compressed text is empty, tokens unchanged
        assert result.successful is True
        assert result.context_pack.rendered_context == ""
        assert result.tokens_before == result.tokens_after

    def test_fail_open_on_error(self):
        """On network error, returns original pack with successful=False."""
        import httpx
        compressor = self._compressor()
        pack = _make_pack()

        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            result = compressor.compress(pack, query="test query")

        assert result.successful is False
        assert result.context_pack is pack
        assert result.tokens_before == result.tokens_after
