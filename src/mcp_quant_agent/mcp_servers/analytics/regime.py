"""Market regime detection -- labels each bar as bull / bear / range / high_vol.

Two detector versions
---------------------
* **v1** (``label_regimes``) -- original detector.
  - Trend: 60-bar momentum.
  - Vol threshold: 80th-percentile of the FULL series.  This is a minor
    look-ahead bias: adding future high-vol bars raises the threshold and
    can re-label past HIGH_VOL bars.  Kept for backward-compatibility and
    comparison.

* **v2** (``label_regimes_v2``) -- improved, fully causal.  Default.
  - Trend: **20-bar momentum** (one trading month).  More reactive: correctly
    labels Jan 2023 (a +17% recovery month) as BULL, not BEAR.
  - Vol threshold: **trailing 252-bar percentile** -- computed from past bars
    only at each point.  No look-ahead.  See ``TestRegimeLookAheadGuard``.

The causal guarantee
--------------------
v2 satisfies: ``label_regimes_v2(rows[:t])[-1]["regime"]
              == label_regimes_v2(rows)[:t][-1]["regime"]``
for all t.  Adding future bars never changes past labels.  The test
``test_v2_causal_vol_threshold`` verifies this structurally.

Priority order (both versions): HIGH_VOL > BULL > BEAR > RANGE.

Window-choice rationale (v2)
-----------------------------
- trend_short = 20d: one calendar month, reactive to recent reversals.
- vol_window = 20d: standard realised vol window.
- vol_percentile_window = 252d: one trading year; provides a stable
  rolling baseline.  For the first 252 bars the window is shorter
  (documented limitation).
- bull/bear threshold = +/-2%: tighter than v1's 5% to reduce RANGE labels
  in trending markets.

Reference: ATLAS (arXiv:2510.15949) and LiveTradeBench show that agent
performance and CoT quality vary significantly across regimes -- specifically
that CoT quality degrades in high-volatility periods.  Correct regime labels
are therefore critical to the thesis evaluation.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any


class Regime(StrEnum):
    """Market regime label.

    Priority for multi-condition labelling: HIGH_VOL > BULL > BEAR > RANGE.
    """

    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"
    HIGH_VOL = "high_vol"


# ---------------------------------------------------------------------------
# Warmup constants — single source of truth for both BacktestEngine and
# PMBacktestEngine.  Import here; do NOT hardcode 380 / 130 elsewhere.
# ---------------------------------------------------------------------------

# The v2 regime detector needs:
#   - vol_window (20 bars) to produce the first non-None rolling vol
#   - vol_percentile_window (252 bars) of non-None rolling vols to fully
#     populate the HIGH_VOL threshold
#
# First non-None vol is at bar vol_window=20.
# For bar i of the full series to have a fully-populated threshold, we need:
#   i - vol_percentile_window + 1 <= vol_window
#   i >= vol_window + vol_percentile_window - 1 = 20 + 252 - 1 = 271
#
# These are the actual DEFAULT parameters — confirmed no caller overrides them:
#   trend_short (primary signal):   20 trading days  (~29 calendar days)
#   vol_window  (rolling vol):      20 trading days
#   vol_percentile_window (HIGH_VOL threshold):  252 trading days  (~365 calendar days)
#
# The controlling requirement is vol_percentile_window, not trend_short.
# The "132 calendar days" is NOT a parameter in this codebase.

#: Minimum trading-day warmup for a fully-populated v2 regime label at bar 0.
#: Derived as vol_window + vol_percentile_window - 1 = 20 + 252 - 1 = 271.
REGIME_WARMUP_TRADING_DAYS: int = 271

#: Calendar-day equivalent of REGIME_WARMUP_TRADING_DAYS.
#: Factor = 365.25/252 ≈ 1.450.  Safety margin of +20 days covers public holidays.
#: ceil(271 × 1.450) + 20 = 393 + 20 = 413 → 420 (rounded to multiple of 7).
#:
#: Both BacktestEngine and PMBacktestEngine import this constant.
#: Changing vol_percentile_window in label_regimes_v2 MUST update this constant.
REGIME_WARMUP_CALENDAR_DAYS: int = 420


# ---------------------------------------------------------------------------
# v1 -- original detector (backward-compatible, kept for comparison)
# ---------------------------------------------------------------------------


def label_regimes(
    rows: list[dict[str, Any]],
    vol_window: int = 20,
    trend_window: int = 60,
    high_vol_pct: float = 0.80,
    bull_threshold: float = 0.05,
    bear_threshold: float = -0.05,
) -> list[dict[str, Any]]:
    """Attach a regime label to each bar -- original detector (v1).

    .. note::
       The ``high_vol_pct`` threshold is computed from the **full series'**
       percentile -- a minor look-ahead bias.  Prefer ``label_regimes_v2``
       for production use.  v1 is kept for comparison.

    Parameters
    ----------
    rows:
        OHLCV bars (oldest first), already filtered by t_now.
    vol_window:
        Rolling window for realised volatility (default 20 bars).
    trend_window:
        Lookback for momentum signal (default 60 bars).
    high_vol_pct:
        Percentile threshold for HIGH_VOL using **full series** (minor look-ahead).
    bull_threshold, bear_threshold:
        Momentum thresholds (+5% / -5%).

    Returns
    -------
    list[dict[str, Any]]
        One dict per bar: ``date``, ``close``, ``rolling_vol_ann``,
        ``momentum_60d``, ``regime`` (None if insufficient lookback).

    Examples
    --------
    >>> rows = [{"date": f"2022-{m:02d}-01", "close": 100.0 + m} for m in range(1, 90)]
    >>> labeled = label_regimes(rows)
    >>> labeled[-1]["regime"] in {"bull", "bear", "range", "high_vol"}
    True
    """
    closes = [float(r["close"]) for r in rows]
    n = len(closes)

    # Step 1: log returns
    log_rets: list[float | None] = [None]
    for i in range(1, n):
        if closes[i - 1] > 0:
            log_rets.append(math.log(closes[i] / closes[i - 1]))
        else:
            log_rets.append(None)

    # Step 2: rolling realised vol (annualised)
    ann_factor = math.sqrt(252)
    rolling_vol: list[float | None] = []
    for i in range(n):
        if i < vol_window:
            rolling_vol.append(None)
        else:
            window = [r for r in log_rets[i - vol_window + 1 : i + 1] if r is not None]
            if len(window) < 2:
                rolling_vol.append(None)
            else:
                mean = sum(window) / len(window)
                var = sum((x - mean) ** 2 for x in window) / (len(window) - 1)
                rolling_vol.append(math.sqrt(var) * ann_factor)

    # Step 3: high-vol threshold (80th percentile of FULL series -- minor look-ahead)
    valid_vols = [v for v in rolling_vol if v is not None]
    if valid_vols:
        sorted_vols = sorted(valid_vols)
        idx = min(int(high_vol_pct * len(sorted_vols)), len(sorted_vols) - 1)
        vol_threshold = sorted_vols[idx]
    else:
        vol_threshold = float("inf")

    # Step 4: momentum signal
    momentum: list[float | None] = []
    for i in range(n):
        if i < trend_window or closes[i - trend_window] <= 0:
            momentum.append(None)
        else:
            momentum.append(closes[i] / closes[i - trend_window] - 1.0)

    # Step 5: assign regime
    result: list[dict[str, Any]] = []
    for i in range(n):
        vol = rolling_vol[i]
        mom = momentum[i]

        if vol is None or mom is None:
            regime: str | None = None
        elif vol >= vol_threshold:
            regime = Regime.HIGH_VOL.value
        elif mom >= bull_threshold:
            regime = Regime.BULL.value
        elif mom <= bear_threshold:
            regime = Regime.BEAR.value
        else:
            regime = Regime.RANGE.value

        result.append(
            {
                "date": rows[i]["date"],
                "close": closes[i],
                "rolling_vol_ann": round(vol, 6) if vol is not None else None,
                "momentum_60d": round(mom, 6) if mom is not None else None,
                "regime": regime,
            }
        )
    return result


# ---------------------------------------------------------------------------
# v2 -- improved detector: reactive trend + causal vol threshold
# ---------------------------------------------------------------------------


def _apply_min_hold_smoothing(
    raw_labels: list[str | None],
    min_hold: int,
) -> list[str | None]:
    """Apply causal minimum-hold smoothing to a raw regime label sequence.

    A regime change is only confirmed after the new label has appeared for
    ``min_hold`` consecutive non-None bars.  This suppresses single-bar and
    brief noise excursions (which account for ~30-36% of all regime runs on
    AAPL/MSFT/NVDA 2022-2024).

    Fully causal: ``smoothed[t]`` depends only on ``raw[0..t]``.
    The anti-lookahead guarantee is preserved: adding future bars does NOT
    change any past smoothed label.

    Parameters
    ----------
    raw_labels:
        Unsmoothed labels, ``None`` for warm-up bars with insufficient history.
    min_hold:
        Minimum number of consecutive non-None bars the new label must hold
        before being adopted (default 3 in ``label_regimes_v2``).

    Returns
    -------
    list[str | None]
        Same length as ``raw_labels``.  None entries are preserved.

    Examples
    --------
    >>> _apply_min_hold_smoothing(["bull"]*5 + ["bear"] + ["bull"]*5, min_hold=3)
    ['bull', 'bull', 'bull', 'bull', 'bull', 'bull', 'bull', 'bull', 'bull', 'bull', 'bull']
    """
    n = len(raw_labels)
    smoothed: list[str | None] = [None] * n
    confirmed: str | None = None  # the currently confirmed (smoothed) label

    for i in range(n):
        label = raw_labels[i]
        if label is None:
            smoothed[i] = None
            continue

        if confirmed is None:
            # Bootstrap: first non-None label is adopted immediately.
            confirmed = label
            smoothed[i] = confirmed
            continue

        if label == confirmed:
            smoothed[i] = confirmed
        else:
            # Check if `label` has held for `min_hold` consecutive non-None bars.
            # Walk backward from i through non-None labels.
            consecutive = 0
            j = i
            while j >= 0 and consecutive < min_hold:
                if raw_labels[j] is None:
                    break  # gap in data — don't cross warm-up boundary
                if raw_labels[j] == label:
                    consecutive += 1
                    j -= 1
                else:
                    break
            if consecutive >= min_hold:
                confirmed = label
            smoothed[i] = confirmed

    return smoothed


def label_regimes_v2(
    rows: list[dict[str, Any]],
    vol_window: int = 20,
    trend_short: int = 20,
    trend_long: int = 60,
    vol_percentile_window: int = 252,
    high_vol_pct: float = 0.80,
    bull_threshold: float = 0.02,
    bear_threshold: float = -0.02,
    min_hold: int = 3,
) -> list[dict[str, Any]]:
    """Improved regime detector -- reactive short-term trend + causal vol threshold.

    Fixes two issues in ``label_regimes`` (v1):

    1. **Short-term trend (20d)** replaces the 60-day window.  The 60-day
       window reached back into the Nov 2022 crash, labelling the +17% Jan 2023
       recovery as BEAR.  The 20-day window correctly captures the current
       direction.

    2. **Causal vol threshold** -- at each bar t the HIGH_VOL threshold is the
       ``high_vol_pct``-th percentile of the past ``vol_percentile_window``
       bars' rolling vol, not the full series.  Future bars never affect
       past thresholds.

    Parameters
    ----------
    rows:
        OHLCV bars (oldest first), already filtered by t_now.
    vol_window:
        Rolling window for realised volatility (default 20 bars).
    trend_short:
        Short-term momentum lookback in trading days (default 20).
        Primary signal used for BULL / BEAR labelling.
    trend_long:
        Medium-term momentum lookback in trading days (default 60).
        **Diagnostic only** -- included in output but NOT used for labelling.
    vol_percentile_window:
        Trailing window (bars) for computing the HIGH_VOL threshold
        (default 252 = one trading year).  Fully causal.
    high_vol_pct:
        Percentile for HIGH_VOL threshold (default 0.80).
    bull_threshold, bear_threshold:
        Short-term momentum thresholds (+2% / -2%).
    min_hold:
        Causal minimum-hold smoothing window (default 3).  A regime change is
        only confirmed after ``min_hold`` consecutive bars show the new label.
        This suppresses single-bar and 2-bar noise excursions (~30-36% of runs
        on AAPL/MSFT/NVDA 2022-2024).  Pass ``min_hold=1`` to disable.

    Returns
    -------
    list[dict[str, Any]]
        One dict per bar: ``date``, ``close``, ``rolling_vol_ann``,
        ``vol_threshold_trailing``, ``trend_short_20d``, ``trend_long_60d``,
        ``regime_raw`` (unsmoothed), ``regime`` (smoothed, or raw if min_hold=1).
        ``regime`` is None for warm-up bars with insufficient lookback.

    Structural guarantee
    --------------------
    ``label_regimes_v2(rows[:t])[-1]["regime"]
    == label_regimes_v2(rows)[t-1]["regime"]``
    for all t, for any ``min_hold``.  See ``TestRegimeLookAheadGuard`` for the
    formal test.  The smoothing function is also causal by construction.

    Examples
    --------
    >>> rows = [{"date": f"2022-{m:02d}-01", "close": 100.0 + m} for m in range(1, 60)]
    >>> labeled = label_regimes_v2(rows)
    >>> labeled[-1]["regime"] in {"bull", "bear", "range", "high_vol", None}
    True
    """
    closes = [float(r["close"]) for r in rows]
    n = len(closes)

    # ── Step 1: daily log-returns ────────────────────────────────────────────
    log_rets: list[float | None] = [None]
    for i in range(1, n):
        if closes[i - 1] > 0:
            log_rets.append(math.log(closes[i] / closes[i - 1]))
        else:
            log_rets.append(None)

    # ── Step 2: 20-bar rolling realised vol (annualised) ─────────────────────
    ann_factor = math.sqrt(252)
    rolling_vol: list[float | None] = []
    for i in range(n):
        if i < vol_window:
            rolling_vol.append(None)
        else:
            window = [r for r in log_rets[i - vol_window + 1 : i + 1] if r is not None]
            if len(window) < 2:
                rolling_vol.append(None)
            else:
                mean = sum(window) / len(window)
                var = sum((x - mean) ** 2 for x in window) / (len(window) - 1)
                rolling_vol.append(math.sqrt(var) * ann_factor)

    # ── Step 3: causal vol threshold (trailing percentile, NO look-ahead) ────
    # At bar i: use only rolling_vol[max(0, i-vol_percentile_window+1) : i+1].
    # Future bars are never included -- this is the structural anti-lookahead
    # guarantee for the regime detector.
    vol_thresholds: list[float] = []
    for i in range(n):
        start = max(0, i - vol_percentile_window + 1)
        past_vols = [v for v in rolling_vol[start : i + 1] if v is not None]
        if past_vols:
            sorted_past = sorted(past_vols)
            idx = min(int(high_vol_pct * len(sorted_past)), len(sorted_past) - 1)
            vol_thresholds.append(sorted_past[idx])
        else:
            vol_thresholds.append(float("inf"))  # no vol data yet

    # ── Step 4: short-term momentum (primary signal) ─────────────────────────
    momentum_short: list[float | None] = []
    for i in range(n):
        if i < trend_short or closes[i - trend_short] <= 0:
            momentum_short.append(None)
        else:
            momentum_short.append(closes[i] / closes[i - trend_short] - 1.0)

    # ── Step 5: long-term momentum (diagnostic output only) ──────────────────
    momentum_long: list[float | None] = []
    for i in range(n):
        if i < trend_long or closes[i - trend_long] <= 0:
            momentum_long.append(None)
        else:
            momentum_long.append(closes[i] / closes[i - trend_long] - 1.0)

    # ── Step 6: assign raw regime (priority: HIGH_VOL > BULL > BEAR > RANGE) ──
    raw_regimes: list[str | None] = []
    for i in range(n):
        vol = rolling_vol[i]
        mom_s = momentum_short[i]
        vt = vol_thresholds[i]

        if vol is None or mom_s is None:
            raw_regime: str | None = None
        elif vol > vt:
            # Strictly greater than the trailing percentile threshold.
            # Using > (not >=) so that a bar exactly AT the threshold is not
            # labelled HIGH_VOL (avoids all-HIGH_VOL in constant-vol series).
            raw_regime = Regime.HIGH_VOL.value
        elif mom_s >= bull_threshold:
            raw_regime = Regime.BULL.value
        elif mom_s <= bear_threshold:
            raw_regime = Regime.BEAR.value
        else:
            raw_regime = Regime.RANGE.value
        raw_regimes.append(raw_regime)

    # ── Step 7: apply causal min-hold smoothing (if requested) ───────────────
    final_regimes = (
        _apply_min_hold_smoothing(raw_regimes, min_hold)
        if min_hold > 1
        else raw_regimes
    )

    result: list[dict[str, Any]] = []
    for i in range(n):
        vol = rolling_vol[i]
        mom_s = momentum_short[i]
        mom_l = momentum_long[i]
        vt = vol_thresholds[i]
        result.append(
            {
                "date": rows[i]["date"],
                "close": closes[i],
                "rolling_vol_ann": round(vol, 6) if vol is not None else None,
                "vol_threshold_trailing": round(vt, 6) if vt != float("inf") else None,
                "trend_short_20d": round(mom_s, 6) if mom_s is not None else None,
                "trend_long_60d": (
                    round(momentum_long[i], 6) if mom_l is not None else None  # type: ignore[arg-type]
                ),
                "regime_raw": raw_regimes[i],  # unsmoothed, for diagnostics
                "regime": final_regimes[i],
            }
        )
    return result


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------


def get_current_regime(
    rows: list[dict[str, Any]],
    version: str = "v2",
    **kwargs: Any,
) -> str | None:
    """Return the regime label for the most recent bar.

    Parameters
    ----------
    rows:
        OHLCV bars filtered to <= t_now (oldest first).
    version:
        ``"v2"`` (default, recommended) or ``"v1"`` (legacy, minor look-ahead).
    **kwargs:
        Forwarded to the chosen detector function.

    Returns
    -------
    str | None
        One of ``"bull"``, ``"bear"``, ``"range"``, ``"high_vol"``, or None.
    """
    if not rows:
        return None
    if version == "v1":
        labeled = label_regimes(rows, **kwargs)
    else:
        labeled = label_regimes_v2(rows, **kwargs)
    return str(labeled[-1]["regime"]) if labeled[-1]["regime"] is not None else None
