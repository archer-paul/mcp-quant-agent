"""Tests for the bar validation layer (bar_validation.py).

These tests verify that malformed bars are caught *before* they enter the
backtest engine.  They cover the exact failure modes seen in run #1:

1. ``close`` is a multi-element ``pd.Series`` (yfinance MultiIndex format).
2. ``close`` is a single-element ``pd.Series`` (acceptable — extract scalar).
3. Negative price (close < 0).
4. ``high < low`` inconsistency.
5. ``close > high`` inconsistency.
6. ``close < low`` inconsistency.
7. Anti-spike: close outside [open*0.5, open*2.0].
8. Negative volume.
9. A valid bar passes without raising.
10. ``_to_float_scalar`` raises on NaN / Inf.
11. ``get_price_history`` raises when the cache contains a corrupt bar
    (regression guard so old cached files can never silently produce bad trades).

Run with::

    pytest tests/test_bar_validation.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mcp_quant_agent.mcp_servers.data.bar_validation import (
    _to_float_scalar,
    validate_bar,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_bar(**overrides: Any) -> dict[str, Any]:
    """Return a valid OHLCV bar, optionally overriding fields."""
    bar: dict[str, Any] = {
        "date": "2023-03-15",
        "open": 150.0,
        "high": 152.0,
        "low": 149.0,
        "close": 151.0,
        "volume": 50_000_000,
    }
    bar.update(overrides)
    return bar


# ---------------------------------------------------------------------------
# _to_float_scalar
# ---------------------------------------------------------------------------


class TestToFloatScalar:
    def test_plain_float(self) -> None:
        assert _to_float_scalar(3.14, "close", "AAPL", "2023-01-03") == pytest.approx(3.14)

    def test_int(self) -> None:
        assert _to_float_scalar(100, "open", "AAPL", "2023-01-03") == pytest.approx(100.0)

    def test_single_element_series_extracted(self) -> None:
        """Single-element Series → extract scalar silently."""
        s = pd.Series([151.25])
        result = _to_float_scalar(s, "close", "AAPL", "2023-01-03")
        assert result == pytest.approx(151.25)

    def test_multi_element_series_raises(self) -> None:
        """Multi-element Series → ValueError (the run #1 root cause)."""
        s = pd.Series([151.25, 372.52])
        with pytest.raises(ValueError, match="2-element Series"):
            _to_float_scalar(s, "close", "AAPL", "2023-01-03")

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="Non-finite"):
            _to_float_scalar(float("nan"), "close", "AAPL", "2023-01-03")

    def test_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="Non-finite"):
            _to_float_scalar(float("inf"), "high", "AAPL", "2023-01-03")

    def test_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot convert"):
            _to_float_scalar("not-a-number", "close", "AAPL", "2023-01-03")


# ---------------------------------------------------------------------------
# validate_bar — good path
# ---------------------------------------------------------------------------


class TestValidateBarGoodPath:
    def test_valid_bar_passes(self) -> None:
        validate_bar(_good_bar(), "AAPL")  # must not raise

    def test_close_equals_high(self) -> None:
        validate_bar(_good_bar(close=152.0), "AAPL")  # close == high is OK

    def test_close_equals_low(self) -> None:
        validate_bar(_good_bar(close=149.0), "AAPL")  # close == low is OK

    def test_zero_volume_ok(self) -> None:
        validate_bar(_good_bar(volume=0), "AAPL")  # zero volume is valid

    def test_no_ticker_arg(self) -> None:
        validate_bar(_good_bar())  # default ticker="" must not raise


# ---------------------------------------------------------------------------
# validate_bar — bad input (the guards)
# ---------------------------------------------------------------------------


class TestValidateBarBadInput:
    # ── Series injection (run #1 bug) ─────────────────────────────────────────

    def test_close_multi_series_raises(self) -> None:
        """Multi-element Series for close must be caught before it's traded."""
        bar = _good_bar(close=pd.Series([151.0, 372.52]))
        with pytest.raises(ValueError, match="2-element Series"):
            validate_bar(bar, "AAPL")

    def test_open_multi_series_raises(self) -> None:
        bar = _good_bar(open=pd.Series([150.0, 300.0]))
        with pytest.raises(ValueError, match="2-element Series"):
            validate_bar(bar, "AAPL")

    # ── Non-positive prices ───────────────────────────────────────────────────

    def test_negative_close_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            validate_bar(_good_bar(close=-1.0), "AAPL")

    def test_zero_open_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            validate_bar(_good_bar(open=0.0, close=0.0, high=0.0, low=0.0), "AAPL")

    # ── OHLC consistency ─────────────────────────────────────────────────────

    def test_high_lt_low_raises(self) -> None:
        with pytest.raises(ValueError, match="high=.*< low="):
            validate_bar(_good_bar(high=148.0, low=149.0, close=148.5), "AAPL")

    def test_close_above_high_raises(self) -> None:
        with pytest.raises(ValueError, match="close=.*> high="):
            validate_bar(_good_bar(close=153.0, high=152.0), "AAPL")

    def test_close_below_low_raises(self) -> None:
        with pytest.raises(ValueError, match="close=.*< low="):
            validate_bar(_good_bar(close=148.0, low=149.0), "AAPL")

    # ── Anti-spike ────────────────────────────────────────────────────────────

    def test_close_more_than_2x_open_raises(self) -> None:
        """close > open*2 is a gross error (e.g. Series head extraction)."""
        with pytest.raises(ValueError, match="anti-spike bounds"):
            validate_bar(_good_bar(open=100.0, close=201.0, high=201.0), "AAPL")

    def test_close_less_than_half_open_raises(self) -> None:
        """close < open*0.5 is equally impossible intraday."""
        with pytest.raises(ValueError, match="anti-spike bounds"):
            validate_bar(_good_bar(open=100.0, high=100.0, low=49.0, close=49.0), "AAPL")

    def test_close_exactly_2x_open_passes(self) -> None:
        """Boundary: close == open*2 is accepted (limit inclusive)."""
        validate_bar(_good_bar(open=100.0, high=200.0, low=99.0, close=200.0), "AAPL")

    def test_close_exactly_half_open_passes(self) -> None:
        """Boundary: close == open*0.5 is accepted (limit inclusive)."""
        validate_bar(_good_bar(open=100.0, high=100.0, low=50.0, close=50.0), "AAPL")

    # ── Volume ────────────────────────────────────────────────────────────────

    def test_negative_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="volume=.*must be >= 0"):
            validate_bar(_good_bar(volume=-1), "AAPL")

    # ── Missing field ─────────────────────────────────────────────────────────

    def test_missing_close_raises(self) -> None:
        bar = {k: v for k, v in _good_bar().items() if k != "close"}
        with pytest.raises(ValueError, match="missing required field 'close'"):
            validate_bar(bar, "AAPL")


# ---------------------------------------------------------------------------
# Integration: get_price_history raises on corrupt cache
# ---------------------------------------------------------------------------


class TestGetPriceHistoryValidatesCacheData:
    """Regression guard: cached bars written by pre-validation code must be caught."""

    def test_corrupt_cache_raises_valueerror(self, tmp_path: Path) -> None:
        """get_price_history should raise if the cache holds a bar with close>high.

        The corrupt bar date equals the requested end_date so no live re-fetch
        is attempted (which would silently overwrite the corrupt data).
        """
        import datetime as dt

        from mcp_quant_agent.clock import SimulationClock, set_clock
        from mcp_quant_agent.mcp_servers.data import yfinance_source
        from mcp_quant_agent.mcp_servers.data.cache import PriceCache

        # t_now >= corrupt bar date so the clock filter does not discard it.
        set_clock(SimulationClock(dt.datetime(2023, 6, 30)))

        # corrupt bar date == end_date → cache.latest_cached_date == end_date
        # → no re-fetch triggered (the condition latest < end_date is False).
        bar_date = "2023-03-15"
        corrupt_bars = [
            {
                "date": bar_date,
                "open": 150.0,
                "high": 151.0,
                "low": 149.0,
                "close": 999.0,   # close > high — impossible, should be caught
                "volume": 50_000_000,
            }
        ]
        cache = PriceCache(cache_dir=tmp_path)
        cache.write("AAPL", "1d", corrupt_bars)

        # Patch the module-level cache so get_price_history uses tmp_path
        orig_cache = yfinance_source._cache
        yfinance_source._cache = cache
        try:
            with pytest.raises(ValueError, match="close=999"):
                yfinance_source.get_price_history(
                    "AAPL",
                    start_date="2023-01-01",
                    end_date=bar_date,   # ← same as latest cached, no re-fetch
                    use_cache=True,
                )
        finally:
            yfinance_source._cache = orig_cache
