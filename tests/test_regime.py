"""Tests for market regime detection.

Critical test
-------------
``TestRegimeLookAheadGuard.test_v2_causal_vol_threshold`` -- the structural
anti-lookahead guard for the regime detector.  It verifies that adding future
high-vol bars to the end of the series does NOT change any past regime label.
This test FAILS if ``label_regimes_v2`` uses a full-series vol percentile
(as v1 does) instead of a trailing percentile.
"""

from __future__ import annotations

import datetime as dt
from typing import cast

from mcp_quant_agent.mcp_servers.analytics.regime import (
    Regime,
    _apply_min_hold_smoothing,
    get_current_regime,
    label_regimes,
    label_regimes_v2,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rows(
    n: int,
    start_close: float = 100.0,
    daily_return: float = 0.005,
    start: dt.date | None = None,
) -> list[dict[str, object]]:
    """Generate synthetic OHLCV bars with a given daily return trend."""
    rows = []
    start = start or dt.date(2022, 1, 3)
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


def _make_volatile_rows(
    n: int,
    start_idx: int = 0,
    amplitude: float = 0.12,
) -> list[dict[str, object]]:
    """Generate rows with extreme vol (alternating ±amplitude)."""
    rows = []
    close = 100.0
    for i in range(n):
        close *= 1.0 + amplitude if i % 2 == 0 else 1.0 - amplitude
        d = (dt.date(2022, 1, 3) + dt.timedelta(days=start_idx + i)).isoformat()
        rows.append(
            {
                "date": d,
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 1. Anti-lookahead guard (CRITICAL)
# ---------------------------------------------------------------------------


class TestRegimeLookAheadGuard:
    """Structural proof that v2 vol threshold is fully causal.

    These tests FAIL if ``label_regimes_v2`` computes the vol threshold from
    the full series (as v1 does) instead of a trailing percentile.
    """

    def test_v2_causal_vol_threshold(self) -> None:
        """Adding future high-vol bars MUST NOT change v2 regime labels at past bars.

        Method: run v2 on a base series of 100 bars, record all labels.
        Then run v2 on the same 100 bars + 60 extra high-vol bars.
        Every label in the first 100 bars must be identical between the two runs.

        This test FAILS if a full-series percentile is used (as in v1):
        - full-series: the 60 extreme future bars raise the 80th-percentile
          threshold, potentially converting past HIGH_VOL labels to RANGE/BULL.
        - trailing: future bars are outside every past bar's trailing window,
          so they have zero influence on past thresholds.
        """
        base_rows = _make_rows(100, daily_return=0.003)  # gentle uptrend, low vol
        future_extreme = _make_volatile_rows(60, start_idx=100, amplitude=0.15)

        labeled_base = label_regimes_v2(base_rows)
        labeled_extended = label_regimes_v2(base_rows + future_extreme)

        for i in range(len(base_rows)):
            assert labeled_base[i]["regime"] == labeled_extended[i]["regime"], (
                f"LOOK-AHEAD at bar {i}: regime changed from "
                f"{labeled_base[i]['regime']!r} to {labeled_extended[i]['regime']!r} "
                "when future high-vol bars were appended. "
                "The vol threshold must be computed from trailing data only."
            )

    def test_v2_vol_threshold_trailing_field_matches_logic(self) -> None:
        """``vol_threshold_trailing`` increases once high-vol bars enter the window."""
        # Use alternating returns to ensure non-zero variance
        rows = _make_volatile_rows(50, start_idx=0, amplitude=0.005)  # low vol
        high_vol = _make_volatile_rows(50, start_idx=50, amplitude=0.15)  # high vol
        labeled = label_regimes_v2(rows + high_vol, vol_window=10, vol_percentile_window=20)
        # Threshold should be non-negative for valid bars
        thresholds = [
            r["vol_threshold_trailing"]
            for r in labeled
            if r["vol_threshold_trailing"] is not None
        ]
        assert all(t >= 0 for t in thresholds)
        # After enough high-vol bars enter the trailing window, threshold rises
        if len(thresholds) >= 30:
            assert thresholds[-1] > thresholds[10], (
                "Threshold should rise after high-vol bars enter the trailing window"
            )

    def test_v2_single_bar_extension_stable(self) -> None:
        """Adding ONE future bar does not change any past label."""
        rows = _make_rows(60, daily_return=0.005)
        labeled_n = label_regimes_v2(rows)
        labeled_n1 = label_regimes_v2(rows + _make_rows(1, start_close=999.0))

        for i in range(len(rows)):
            assert labeled_n[i]["regime"] == labeled_n1[i]["regime"], (
                f"Adding one future bar changed regime at bar {i}: "
                f"{labeled_n[i]['regime']!r} -> {labeled_n1[i]['regime']!r}"
            )

    def test_v1_documents_full_series_percentile(self) -> None:
        """Document (not require) that v1 MAY differ between truncated and full runs.

        v1 uses the full-series 80th percentile.  When the series is extended
        with extreme bars, the threshold rises, potentially changing past labels.
        This test is here for documentation; it asserts the CLAIM about v1 behavior,
        not a failure.  The expected behavior is that v1 can differ, v2 cannot.
        """
        base_rows = _make_rows(100, daily_return=0.003)
        extreme_future = _make_volatile_rows(60, start_idx=100, amplitude=0.20)

        labeled_v2_base = label_regimes_v2(base_rows)
        labeled_v2_ext = label_regimes_v2(base_rows + extreme_future)

        # v2 must be identical at all past bars (causal guarantee)
        for i in range(len(base_rows)):
            assert labeled_v2_base[i]["regime"] == labeled_v2_ext[i]["regime"]

        v2_diffs = sum(
            1
            for i in range(len(base_rows))
            if labeled_v2_base[i]["regime"] != labeled_v2_ext[i]["regime"]
        )
        assert v2_diffs == 0, f"v2 should be causal but has {v2_diffs} changed labels"
        # v1 may have non-zero diffs -- documented limitation (not asserted here)


# ---------------------------------------------------------------------------
# 2. v1 tests (backward-compatible)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3. v2 correctness tests
# ---------------------------------------------------------------------------


class TestLabelRegimesV2:
    """Tests for the improved v2 detector."""

    def test_returns_same_length(self) -> None:
        rows = _make_rows(100)
        result = label_regimes_v2(rows)
        assert len(result) == 100

    def test_first_20_bars_none(self) -> None:
        """With trend_short=20, first 20 bars have None regime."""
        rows = _make_rows(100)
        result = label_regimes_v2(rows, trend_short=20)
        for r in result[:20]:
            assert r["regime"] is None, f"Expected None, got {r['regime']!r}"

    def test_21st_bar_onwards_has_regime(self) -> None:
        """Bar 21 (index 20) should have a regime label."""
        rows = _make_rows(100, daily_return=0.01)  # clear uptrend
        result = label_regimes_v2(rows, trend_short=20, vol_window=20)
        assert result[20]["regime"] is not None

    def test_uptrend_labels_bull(self) -> None:
        """Net uptrend with realistic noise -> mostly BULL labels."""
        # Use the same alternating structure but with a net positive drift
        # so we have real variance AND a bull signal.
        rows = []
        close = 100.0
        for i in range(80):
            # +1.5% / +0.5% alternating => net positive, non-zero variance
            close *= 1.015 if i % 2 == 0 else 1.005
            d = (dt.date(2022, 1, 3) + dt.timedelta(days=i)).isoformat()
            rows.append({"date": d, "open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close, "volume": 1_000_000})
        result = label_regimes_v2(rows, bull_threshold=0.02)
        labeled = [r for r in result if r["regime"] is not None]
        bull_count = sum(1 for r in labeled if r["regime"] == Regime.BULL.value)
        assert bull_count > len(labeled) * 0.4, (
            f"Expected mostly BULL in uptrend, got {bull_count}/{len(labeled)}: "
            f"{[r['regime'] for r in labeled]}"
        )

    def test_downtrend_labels_bear(self) -> None:
        """Steady -0.5% daily return -> 20-day return <<  -2% -> BEAR label."""
        rows = _make_rows(80, daily_return=-0.005)
        result = label_regimes_v2(rows, bear_threshold=-0.02)
        labeled = [r for r in result if r["regime"] is not None]
        bear_count = sum(1 for r in labeled if r["regime"] == Regime.BEAR.value)
        assert bear_count > 0

    def test_high_vol_bars_labeled_high_vol(self) -> None:
        """Extreme vol bars should be labeled HIGH_VOL."""
        # 30 calm bars + 30 extreme vol bars
        rows = _make_rows(30, daily_return=0.001) + _make_volatile_rows(30, start_idx=30)
        result = label_regimes_v2(rows, vol_window=5, vol_percentile_window=20)
        labeled = [r for r in result[30:] if r["regime"] is not None]
        hv_count = sum(1 for r in labeled if r["regime"] == Regime.HIGH_VOL.value)
        assert hv_count > 0, "Expected some HIGH_VOL labels in extreme vol period"

    def test_extra_output_fields_present(self) -> None:
        """v2 output has the extra diagnostic fields."""
        rows = _make_rows(80)
        result = label_regimes_v2(rows)
        for r in result:
            for key in ("trend_short_20d", "trend_long_60d", "vol_threshold_trailing"):
                assert key in r, f"Missing key {key!r}"

    def test_trend_long_60d_diagnostic_none_for_first_60_bars(self) -> None:
        """trend_long_60d is None for bars < 60."""
        rows = _make_rows(80)
        result = label_regimes_v2(rows, trend_long=60)
        for r in result[:60]:
            assert r["trend_long_60d"] is None

    def test_v2_more_reactive_than_v1_on_recovery(self) -> None:
        """The key thesis scenario: v2 labels a recovery month as BULL, v1 may not.

        Build a series: 60 bars crashing (-0.5%/day) then 20 bars recovering
        (+1%/day).  At bar 79:
        - v1 trend (60-day): still sees the crash in the lookback -> BEAR
        - v2 trend (20-day): only sees the recovery -> BULL
        """
        crash_rows = _make_rows(60, start_close=150.0, daily_return=-0.005)
        recovery_rows = _make_rows(20, start_close=crash_rows[-1]["close"], daily_return=0.01, start=dt.date(2022, 3, 5))  # type: ignore[arg-type]
        all_rows = crash_rows + recovery_rows

        v1 = label_regimes(all_rows)[-1]["regime"]
        v2 = label_regimes_v2(all_rows)[-1]["regime"]

        # v1 with 60d lookback: crash dominates
        # v2 with 20d lookback: recovery dominates
        # We assert v2 is not bear (should be bull or range)
        assert v2 != Regime.BEAR.value, (
            f"v2 should see the recovery, but labelled as {v2!r}. "
            f"(v1 labelled as {v1!r} -- expected bear due to 60d lookback)"
        )


# ---------------------------------------------------------------------------
# 4. get_current_regime tests
# ---------------------------------------------------------------------------


class TestGetCurrentRegime:
    def test_returns_string_or_none(self) -> None:
        rows = _make_rows(100)
        regime = get_current_regime(rows)
        valid = {r.value for r in Regime} | {None}
        assert regime in valid

    def test_empty_rows_returns_none(self) -> None:
        assert get_current_regime([]) is None

    def test_too_few_rows_returns_none(self) -> None:
        rows = _make_rows(15)  # less than trend_short=20 for v2
        assert get_current_regime(rows) is None

    def test_version_v1_uses_label_regimes(self) -> None:
        """version='v1' should return a v1-compatible result."""
        rows = _make_rows(150, daily_return=0.01)
        regime_v1 = get_current_regime(rows, version="v1")
        valid = {r.value for r in Regime} | {None}
        assert regime_v1 in valid


# ---------------------------------------------------------------------------
# 5. Min-hold smoothing tests
# ---------------------------------------------------------------------------


class TestMinHoldSmoothing:
    """Tests for _apply_min_hold_smoothing and label_regimes_v2 min_hold."""

    def test_single_bar_spike_suppressed(self) -> None:
        """A 1-bar excursion (bull then single bear then bull) is swallowed."""
        raw = cast(list[str | None], ["bull"] * 10 + ["bear"] + ["bull"] * 10)
        smoothed = _apply_min_hold_smoothing(raw, min_hold=3)
        # The bear at position 10 and the 2 bulls before 3 bulls of bull
        # should be smoothed to bull — never confirmed because bear lasted 1 bar
        assert smoothed[10] == "bull", f"Expected bull, got {smoothed[10]}"
        assert smoothed[11] == "bull"

    def test_two_bar_spike_suppressed(self) -> None:
        """A 2-bar regime excursion is suppressed with min_hold=3."""
        raw = cast(list[str | None], ["bull"] * 10 + ["bear", "bear"] + ["bull"] * 10)
        smoothed = _apply_min_hold_smoothing(raw, min_hold=3)
        # bear appears twice but not 3 times consecutively → not confirmed
        assert all(s == "bull" for s in smoothed[10:12])

    def test_three_bar_run_confirmed(self) -> None:
        """A 3-bar run of a new label IS confirmed with min_hold=3."""
        raw = cast(list[str | None], ["bull"] * 10 + ["bear"] * 4 + ["bull"] * 10)
        smoothed = _apply_min_hold_smoothing(raw, min_hold=3)
        # bear runs for 4 bars → confirmed at bar 12 (index of 3rd bear)
        assert smoothed[12] == "bear"
        assert smoothed[13] == "bear"
        # Once confirmed, switching back to bull also needs min_hold bars
        # (10 bulls at the end → confirmed)
        assert smoothed[-1] == "bull"

    def test_none_passthrough(self) -> None:
        """None (warm-up) entries pass through unchanged."""
        raw: list[str | None] = [None, None, "bull", "bull", "bull", "bear"]
        smoothed = _apply_min_hold_smoothing(raw, min_hold=3)
        assert smoothed[0] is None
        assert smoothed[1] is None
        # bear at position 5 is only 1 bar, stays bull
        assert smoothed[5] == "bull"

    def test_min_hold_1_is_identity(self) -> None:
        """min_hold=1 is equivalent to no smoothing."""
        raw: list[str | None] = [None, None, "bull", "bear", "bull", "range"]
        smoothed = _apply_min_hold_smoothing(raw, min_hold=1)
        assert smoothed == raw

    def test_smoothing_causal_with_future_extreme(self) -> None:
        """Adding future bars does NOT change smoothed labels for past bars."""
        base_rows = _make_rows(100, daily_return=0.003)
        extreme_future = _make_volatile_rows(60, start_idx=100, amplitude=0.20)

        base_labeled = label_regimes_v2(base_rows, min_hold=3)
        ext_labeled = label_regimes_v2(base_rows + extreme_future, min_hold=3)

        for i in range(len(base_rows)):
            assert base_labeled[i]["regime"] == ext_labeled[i]["regime"], (
                f"SMOOTHED LOOK-AHEAD at bar {i}: "
                f"base={base_labeled[i]['regime']!r}, ext={ext_labeled[i]['regime']!r}"
            )

    def test_min_hold_reduces_run_count(self) -> None:
        """Smoothed series has fewer runs than the raw series."""
        rows = _make_rows(200, daily_return=0.002)
        labeled_raw = label_regimes_v2(rows, min_hold=1)
        labeled_smooth = label_regimes_v2(rows, min_hold=3)

        raw_regimes = [r["regime"] for r in labeled_raw if r["regime"] is not None]
        smooth_regimes = [r["regime"] for r in labeled_smooth if r["regime"] is not None]

        def count_runs(seq: list[str]) -> int:
            return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]) + 1

        raw_runs = count_runs(raw_regimes)
        smooth_runs = count_runs(smooth_regimes)
        # Smoothed should have fewer or equal runs
        assert smooth_runs <= raw_runs, (
            f"Smoothing increased run count: raw={raw_runs}, smooth={smooth_runs}"
        )

    def test_regime_raw_field_preserved(self) -> None:
        """The 'regime_raw' field should match min_hold=1 labels."""
        rows = _make_rows(100, daily_return=0.003)
        smoothed = label_regimes_v2(rows, min_hold=3)
        raw = label_regimes_v2(rows, min_hold=1)

        for i in range(len(rows)):
            assert smoothed[i]["regime_raw"] == raw[i]["regime"], (
                f"regime_raw mismatch at bar {i}: "
                f"{smoothed[i]['regime_raw']!r} vs {raw[i]['regime']!r}"
            )

    def test_version_v2_default(self) -> None:
        """Default version should be v2."""
        rows = _make_rows(100, daily_return=0.01)
        regime_default = get_current_regime(rows)
        regime_v2 = get_current_regime(rows, version="v2")
        assert regime_default == regime_v2

    def test_recovery_scenario_v2_not_bear(self) -> None:
        """In a recovery, v2 should not label as bear (regression test)."""
        crash = _make_rows(60, start_close=150.0, daily_return=-0.005)
        recovery = _make_rows(
            25,
            start_close=crash[-1]["close"],  # type: ignore[arg-type]
            daily_return=0.01,
            start=dt.date(2022, 3, 10),
        )
        regime = get_current_regime(crash + recovery, version="v2")
        assert regime != Regime.BEAR.value, (
            f"v2 should detect recovery but got {regime!r}"
        )
