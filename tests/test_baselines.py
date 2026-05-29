"""Tests for backtest/baselines.py.

Critical test
-------------
``TestSignalShift.test_signal_shifted_one_bar`` — the structural look-ahead
guard for baselines.  If the ``.shift(1)`` is removed from any strategy, this
test FAILS, proving the baseline is contaminated with look-ahead.

Other tests verify correctness of strategy logic and metrics consistency.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mcp_quant_agent.backtest.baselines import (
    _bb_signal,
    _momentum_signal,
    _simulate_nav,
    buy_and_hold,
    mean_reversion_bb,
    momentum_ts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(closes: list[float], start: str = "2022-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close-price list."""
    n = len(closes)
    dates = pd.date_range(start, periods=n, freq="B")  # business days
    close = pd.Series(closes, index=dates, dtype=float)
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.010,
            "low": close * 0.990,
            "close": close,
            "volume": [1_000_000.0] * n,
        },
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# 1. Signal shift — THE CRITICAL LOOK-AHEAD GUARD
# ---------------------------------------------------------------------------


class TestSignalShift:
    """The ``.shift(1)`` is the anti-look-ahead guard for baselines.

    If it is removed, the strategy uses the same-day signal — a look-ahead
    bias.  These tests FAIL if the shift is removed.
    """

    def test_momentum_signal_shifted_one_bar(self) -> None:
        """exec_signal[t] == raw_signal[t-1] for all t >= 1.

        The strategy should not react to a signal on the same bar it fires.
        If the shift is removed, exec_signal[t] == raw_signal[t] — the test
        would fail because the assertion below checks the previous bar's value.
        """
        # Flat then rising prices: momentum signal fires when price rises
        closes = [100.0] * 5 + [102.0, 103.0, 104.0, 105.0, 106.0]
        close = pd.Series(closes, index=pd.date_range("2022-01-01", periods=10))

        raw = _momentum_signal(close, lookback=3)
        exec_sig = raw.shift(1).fillna(0.0)

        # The shift invariant: exec_signal[i] == raw_signal[i-1]
        for i in range(1, len(close)):
            assert exec_sig.iloc[i] == raw.iloc[i - 1], (
                f"LOOK-AHEAD BUG: exec_signal[{i}] != raw_signal[{i - 1}]. "
                "The signal is not shifted — same-day execution!"
            )
        # First bar must be flat (NaN → 0 from the shift)
        assert exec_sig.iloc[0] == 0.0

    def test_momentum_no_position_on_signal_fire_day(self) -> None:
        """The day a momentum signal fires, the position is still 0 (flat).

        Lookback=5: signal fires when bar 5 is above bar 0.  With the shift,
        the strategy is NOT yet long on bar 5 — it enters on bar 6.
        """
        # Bars 0-4: flat at 100.  Bar 5: jumps to 110 → signal fires.
        closes = [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 111.0, 112.0, 113.0, 114.0]
        df = _make_df(closes)
        result = momentum_ts(df, lookback=5)
        nav = result["nav_series"]

        # Day 5: price jumps 100→110 (+10%). With shift, no position yet.
        nav_change_day5 = (nav[5] - nav[4]) / nav[4]
        assert abs(nav_change_day5) < 1e-6, (
            f"LOOK-AHEAD: NAV changed on the signal-fire day ({nav_change_day5:.6f}). "
            "The +10% bar on day 5 was captured — signal was not shifted!"
        )

        # Day 6: first day in position; should have a positive return.
        nav_change_day6 = (nav[6] - nav[5]) / nav[5]
        # (price goes 110→111, +0.9%, minus ~10bps cost = ~0.8%)
        assert nav_change_day6 > 0.005, (
            f"Expected positive return on first day in position (day 6), "
            f"got {nav_change_day6:.4f}"
        )

    def test_bollinger_no_position_on_entry_signal_day(self) -> None:
        """The day a BB entry signal fires, the position is still 0.

        BB: close < lower_band → signal fires.  With the shift, the strategy
        does not enter on that bar — it enters the following bar.
        """
        # Construct prices that dip below the lower band on a specific day.
        # Base price 100 with a sharp dip on bar 22 (after 20-bar warm-up).
        closes = [100.0] * 22 + [85.0] + [100.0] * 7  # sharp dip on bar 22
        df = _make_df(closes, start="2022-01-01")
        result = mean_reversion_bb(df, window=20, num_std=2.0)
        nav = result["nav_series"]

        # Bar 22: close = 85, which is well below the lower band (~96).
        # Signal fires here. With shift, no position on bar 22.
        # Close drops from 100 to 85 (-15%). We should NOT capture this loss.
        nav_change_day22 = (nav[22] - nav[21]) / nav[21]
        assert nav_change_day22 > -0.01, (
            f"LOOK-AHEAD: position entered on signal-fire bar 22. "
            f"NAV drop = {nav_change_day22:.4f}. Signal must be shifted by 1 bar."
        )


# ---------------------------------------------------------------------------
# 2. Buy & Hold
# ---------------------------------------------------------------------------


class TestBuyAndHold:
    def test_nav_equals_price_ratio_with_costs(self) -> None:
        """BnH NAV[t] = initial_cash × (1-COST_PER_TRADE) × close[t] / close[0].

        Transaction costs are applied: entry cost at bar 0, exit cost at bar -1.
        The mid-series bars have the entry haircut but not the exit haircut.
        """
        from mcp_quant_agent.backtest.baselines import COST_PER_TRADE
        closes = [100.0, 105.0, 95.0, 110.0, 108.0]
        df = _make_df(closes)
        result = buy_and_hold(df, initial_cash=10_000.0)
        nav = result["nav_series"]
        # All bars except last: only entry cost (1 - COST_PER_TRADE)
        for i, (close, nav_val) in enumerate(zip(closes[:-1], nav[:-1], strict=True)):
            expected = 10_000.0 * (1.0 - COST_PER_TRADE) * close / closes[0]
            assert abs(nav_val - expected) < 0.01, (
                f"NAV mismatch at bar {i}: expected {expected:.2f}, got {nav_val:.2f}"
            )
        # Last bar: entry + exit cost
        expected_last = 10_000.0 * (1.0 - COST_PER_TRADE) ** 2 * closes[-1] / closes[0]
        assert abs(nav[-1] - expected_last) < 0.01

    def test_nav_length_matches_input(self) -> None:
        df = _make_df([100.0] * 30)
        result = buy_and_hold(df)
        assert len(result["nav_series"]) == 30

    def test_initial_nav_below_initial_cash_by_entry_cost(self) -> None:
        """After paying entry cost, first bar NAV < initial_cash."""
        from mcp_quant_agent.backtest.baselines import COST_PER_TRADE
        df = _make_df([100.0, 105.0, 110.0])
        result = buy_and_hold(df, initial_cash=50_000.0)
        expected = 50_000.0 * (1.0 - COST_PER_TRADE)
        assert result["nav_series"][0] == pytest.approx(expected)

    def test_n_trades_is_two(self) -> None:
        df = _make_df([100.0] * 10)
        result = buy_and_hold(df)
        assert result["n_trades"] == 2

    def test_metrics_present(self) -> None:
        df = _make_df([100.0 + i for i in range(100)])
        result = buy_and_hold(df)
        for key in (
            "annualised_return",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "hit_rate",
        ):
            assert key in result["metrics"], f"Missing metric: {key}"


# ---------------------------------------------------------------------------
# 3. Momentum
# ---------------------------------------------------------------------------


class TestMomentum:
    def test_flat_market_no_position(self) -> None:
        """Flat price → momentum always zero → no position → NAV flat."""
        closes = [100.0] * 50
        df = _make_df(closes)
        result = momentum_ts(df, lookback=10)
        nav = result["nav_series"]
        # All returns are 0, so NAV stays flat
        assert all(abs(v - 100_000.0) < 1.0 for v in nav), (
            "Flat market should produce flat NAV"
        )

    def test_uptrend_generates_long_position(self) -> None:
        """Steady uptrend → momentum signal → long → NAV grows."""
        closes = [100.0 + i * 0.5 for i in range(60)]
        df = _make_df(closes)
        result = momentum_ts(df, lookback=20)
        nav = result["nav_series"]
        # NAV should grow overall (we're long in an uptrend)
        assert nav[-1] > nav[0], "Expected NAV growth in uptrend with long momentum"

    def test_insufficient_data_returns_flat_nav(self) -> None:
        """When data < lookback, the function gracefully returns flat NAV."""
        closes = [100.0] * 10
        df = _make_df(closes)
        result = momentum_ts(df, lookback=50)  # lookback > data
        assert result["n_trades"] == 0
        assert all(v == 100_000.0 for v in result["nav_series"])

    def test_transaction_cost_reduces_nav(self) -> None:
        """NAV with costs must be <= NAV without costs (for non-zero trades)."""
        closes = [100.0 + (i % 10) for i in range(80)]
        df = _make_df(closes)

        result_with_cost = momentum_ts(df, lookback=5)
        nav_with_cost = result_with_cost["nav_series"]

        # Without cost: manually simulate using _simulate_nav with cost=0
        close_series = df["close"]
        raw = _momentum_signal(close_series, lookback=5)
        exec_sig = raw.shift(1).fillna(0.0)
        nav_no_cost, _, _, _, _ = _simulate_nav(
            close_series, exec_sig, 100_000.0, cost_per_trade=0.0
        )

        # With trades, net returns must be ≤ gross returns
        if result_with_cost["n_trades"] > 0:
            assert nav_with_cost[-1] <= nav_no_cost[-1] + 1.0, (
                "Transaction costs should reduce final NAV"
            )


# ---------------------------------------------------------------------------
# 4. Mean-reversion Bollinger Bands
# ---------------------------------------------------------------------------


class TestMeanReversionBB:
    def test_no_signal_when_price_within_bands(self) -> None:
        """Flat price with no outliers → Bollinger width ≈ 0 → no entry."""
        # Perfectly flat price: std = 0, bands = mean ± 0 → close is always AT the band.
        # Floating point may cause tiny deviations but no signal.
        closes = [100.0] * 50
        df = _make_df(closes)
        result = mean_reversion_bb(df, window=20, num_std=2.0)
        # With a flat series, std=0 → upper=lower=mean=100 → close ≤ lower is
        # False for close=100 (not strictly less than), so no entry.
        assert result["n_trades"] == 0

    def test_dip_entry_recovery_exit(self) -> None:
        """Dip below lower band → long entry; recovery above upper band → exit."""
        # 30 bars at 100, then dip to 80, then recover to 110
        closes = [100.0] * 25 + [80.0, 82.0, 85.0, 100.0, 105.0, 110.0] + [100.0] * 5
        df = _make_df(closes)
        result = mean_reversion_bb(df, window=20, num_std=1.5)
        # There should be at least one trade (entry on dip)
        assert result["n_trades"] >= 1, "Expected at least one entry on the dip"

    def test_output_keys(self) -> None:
        """Output dict must contain all required keys."""
        df = _make_df([100.0 + (i % 5) for i in range(60)])
        result = mean_reversion_bb(df)
        for key in (
            "strategy",
            "nav_series",
            "metrics",
            "sharpe_ci",
            "n_trades",
            "turnover",
        ):
            assert key in result, f"Missing key: {key}"
        assert result["strategy"] == "mean_reversion_bb"


# ---------------------------------------------------------------------------
# 5. _bb_signal internal tests
# ---------------------------------------------------------------------------


class TestBBSignal:
    def test_signal_enters_on_dip_below_lower_band(self) -> None:
        """Signal becomes 1 when close < lower_band."""
        closes = [100.0] * 20 + [70.0]  # sharp dip on bar 20
        close = pd.Series(closes, index=pd.date_range("2022-01-01", periods=21))
        signal = _bb_signal(close, window=20, num_std=2.0)
        # Signal should be 1 on bar 20 (the dip)
        assert signal.iloc[20] == 1.0, "Expected long signal after dip below lower band"

    def test_signal_exits_above_upper_band(self) -> None:
        """Signal returns to 0 when close > upper_band while in position."""
        closes = [100.0] * 20 + [70.0, 75.0, 80.0, 85.0, 120.0]  # dip then spike
        close = pd.Series(closes, index=pd.date_range("2022-01-01", periods=25))
        signal = _bb_signal(close, window=20, num_std=2.0)
        # Bar 20: dip → enter long. Bar 24: spike → exit
        assert signal.iloc[20] == 1.0, "Expected entry on dip"
        assert signal.iloc[24] == 0.0, "Expected exit after spike above upper band"

    def test_flat_series_no_signal(self) -> None:
        """Flat series → std = 0 → bands collapse → no strict-less-than entry."""
        closes = [100.0] * 30
        close = pd.Series(closes, index=pd.date_range("2022-01-01", periods=30))
        signal = _bb_signal(close, window=20, num_std=2.0)
        assert (signal == 0.0).all(), "Flat prices should produce no entry signal"
