"""Tests for technical indicators.

These are unit tests of the pure-function indicator implementations.
No clock or API calls needed — all functions operate on given OHLCV rows.
"""

from __future__ import annotations

import pytest

from mcp_quant_agent.mcp_servers.analytics.indicators import (
    bollinger_bands,
    compute_indicators,
    macd,
    rsi,
    sma,
)


def _make_rows(n: int, start_close: float = 100.0) -> list[dict[str, object]]:
    return [
        {
            "date": f"2022-01-{i + 1:02d}",
            "open": start_close + i - 0.1,
            "high": start_close + i + 0.5,
            "low": start_close + i - 0.5,
            "close": start_close + i,
            "volume": 1_000_000,
        }
        for i in range(n)
    ]


class TestSMA:
    def test_returns_same_length(self) -> None:
        rows = _make_rows(30)
        result = sma(rows, window=20)
        assert len(result) == 30

    def test_first_window_minus_1_are_none(self) -> None:
        rows = _make_rows(30)
        result = sma(rows, window=20)
        for i in range(19):
            assert result[i]["sma"] is None

    def test_value_at_exactly_window(self) -> None:
        rows = _make_rows(5)
        result = sma(rows, window=3)
        # indices 0..4, window hits at index 2
        assert result[2]["sma"] is not None
        # closes are 100, 101, 102 → mean = 101
        assert abs(float(result[2]["sma"]) - 101.0) < 1e-6  # type: ignore[arg-type]

    def test_dates_preserved(self) -> None:
        rows = _make_rows(5)
        result = sma(rows)
        for i, r in enumerate(result):
            assert r["date"] == rows[i]["date"]


class TestRSI:
    def test_returns_same_length(self) -> None:
        rows = _make_rows(30)
        result = rsi(rows)
        assert len(result) == 30

    def test_first_window_are_none(self) -> None:
        rows = _make_rows(20)
        result = rsi(rows, window=14)
        for i in range(14):
            assert result[i]["rsi"] is None

    def test_rsi_in_valid_range(self) -> None:
        rows = _make_rows(30)
        result = rsi(rows, window=14)
        for r in result:
            if r["rsi"] is not None:
                val = float(r["rsi"])  # type: ignore[arg-type]
                assert 0.0 <= val <= 100.0


class TestMACD:
    def test_returns_same_length(self) -> None:
        rows = _make_rows(30)
        result = macd(rows)
        assert len(result) == 30

    def test_has_required_keys(self) -> None:
        rows = _make_rows(30)
        result = macd(rows)
        assert all("macd" in r and "signal" in r and "histogram" in r for r in result)

    def test_histogram_equals_macd_minus_signal(self) -> None:
        rows = _make_rows(50)
        result = macd(rows)
        for r in result:
            assert (
                abs(float(r["macd"]) - float(r["signal"]) - float(r["histogram"]))
                < 1e-9
            )  # type: ignore[arg-type]


class TestBollingerBands:
    def test_returns_same_length(self) -> None:
        rows = _make_rows(30)
        result = bollinger_bands(rows)
        assert len(result) == 30

    def test_first_window_minus_1_are_none(self) -> None:
        rows = _make_rows(30)
        result = bollinger_bands(rows, window=20)
        for i in range(19):
            assert result[i]["upper"] is None

    def test_upper_gt_middle_gt_lower(self) -> None:
        rows = _make_rows(30)
        result = bollinger_bands(rows, window=20)
        for r in result[19:]:  # only check non-None entries
            assert float(r["upper"]) >= float(r["middle"]) >= float(r["lower"])  # type: ignore[arg-type]


class TestComputeIndicators:
    def test_raises_on_insufficient_data(self) -> None:
        rows = _make_rows(20)
        with pytest.raises(ValueError, match="26"):
            compute_indicators(rows)  # type: ignore[arg-type]

    def test_returns_expected_keys(self) -> None:
        rows = _make_rows(50)
        result = compute_indicators(rows)  # type: ignore[arg-type]
        for key in ("date", "close", "rsi_14", "macd", "sma_20"):
            assert key in result

    def test_latest_date_correct(self) -> None:
        rows = _make_rows(30)
        result = compute_indicators(rows)  # type: ignore[arg-type]
        assert result["date"] == rows[-1]["date"]
