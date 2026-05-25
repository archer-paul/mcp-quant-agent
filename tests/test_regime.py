"""Tests for market regime detection."""

from __future__ import annotations

from mcp_quant_agent.mcp_servers.analytics.regime import (
    Regime,
    get_current_regime,
    label_regimes,
)


def _make_rows(
    n: int,
    start_close: float = 100.0,
    daily_return: float = 0.005,
) -> list[dict[str, object]]:
    """Generate synthetic OHLCV bars with a given daily return trend."""
    import datetime as dt

    rows = []
    start = dt.date(2022, 1, 3)
    for i in range(n):
        close = start_close * ((1 + daily_return) ** i)
        day = start + dt.timedelta(days=i)
        rows.append(
            {
                "date": day.isoformat(),
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return rows


class TestLabelRegimes:
    def test_returns_same_length(self) -> None:
        rows = _make_rows(100)
        result = label_regimes(rows)
        assert len(result) == 100

    def test_early_rows_none(self) -> None:
        rows = _make_rows(100)
        result = label_regimes(rows, vol_window=20, trend_window=60)
        # First 60 rows (trend_window) have None regime
        for r in result[:60]:
            assert r["regime"] is None

    def test_trending_up_labels_bull(self) -> None:
        rows = _make_rows(150, daily_return=0.008)  # strong uptrend
        result = label_regimes(rows)
        labeled = [r for r in result if r["regime"] is not None]
        # Most of the labeled rows should be bull or high_vol
        bull_count = sum(1 for r in labeled if r["regime"] == Regime.BULL.value)
        assert bull_count > len(labeled) * 0.3

    def test_trending_down_labels_bear(self) -> None:
        rows = _make_rows(150, daily_return=-0.005)  # downtrend
        result = label_regimes(rows)
        labeled = [r for r in result if r["regime"] is not None]
        bear_count = sum(1 for r in labeled if r["regime"] == Regime.BEAR.value)
        assert bear_count > 0

    def test_has_required_keys(self) -> None:
        rows = _make_rows(80)
        result = label_regimes(rows)
        for r in result:
            assert "date" in r
            assert "close" in r
            assert "regime" in r

    def test_regime_values_valid(self) -> None:
        rows = _make_rows(100)
        result = label_regimes(rows)
        valid = {r.value for r in Regime} | {None}
        for r in result:
            assert r["regime"] in valid


class TestGetCurrentRegime:
    def test_returns_string_or_none(self) -> None:
        rows = _make_rows(100)
        regime = get_current_regime(rows)
        valid = {r.value for r in Regime} | {None}
        assert regime in valid

    def test_empty_rows_returns_none(self) -> None:
        assert get_current_regime([]) is None

    def test_too_few_rows_returns_none(self) -> None:
        rows = _make_rows(20)  # less than trend_window (60)
        assert get_current_regime(rows) is None
