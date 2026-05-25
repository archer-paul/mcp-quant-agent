"""Market regime detection — labels each bar as bull / bear / range / high_vol.

Methodology
-----------
Regime labelling uses two signals computed on historical OHLCV data:

1. **Realised volatility** — rolling standard deviation of daily log-returns
   (default window: 20 bars, annualised × √252).  High-vol threshold: 95th
   percentile of the full-period vol distribution.

2. **Trend** — rolling momentum (current close / close N days ago − 1, default
   N = 60 bars).  Thresholds: +5% → bull, −5% → bear, otherwise → range.

Priority order:  HIGH_VOL > BULL > BEAR > RANGE.
When vol is above the high-vol threshold, the regime is HIGH_VOL regardless of
the trend direction.

Why not HMM?
------------
A hidden-Markov model is conceptually cleaner but requires fitting parameters
on historical data — which, done naively, leaks future information (the fitted
transitions see the whole history).  The rule-based approach is fully causal at
every bar.  HMM is noted in docs/DECISIONS.md as a future extension.

Reference: ATLAS (arXiv:2510.15949) and LiveTradeBench show that agent
performance varies significantly by regime — specifically that CoT quality
degrades in high-volatility periods.  This module's output feeds the
regime-segmented evaluation in ``eval/regime.py``.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any


class Regime(StrEnum):
    """Market regime label.

    Ordered by "priority" for multi-condition labelling:
    HIGH_VOL > BULL > BEAR > RANGE.
    """

    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"
    HIGH_VOL = "high_vol"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def label_regimes(
    rows: list[dict[str, Any]],
    vol_window: int = 20,
    trend_window: int = 60,
    high_vol_pct: float = 0.80,
    bull_threshold: float = 0.05,
    bear_threshold: float = -0.05,
) -> list[dict[str, Any]]:
    """Attach a regime label to each bar in *rows*.

    Parameters
    ----------
    rows:
        OHLCV bars (oldest first), already filtered by t_now.  Must contain
        ``"date"`` and ``"close"`` keys.
    vol_window:
        Rolling window for realised volatility (default 20 bars).
    trend_window:
        Lookback for momentum signal (default 60 bars).
    high_vol_pct:
        Percentile threshold for "high volatility" label.  E.g. 0.80 means
        a bar is HIGH_VOL when its rolling vol is above the 80th percentile of
        all non-NaN vols in the series.
    bull_threshold:
        Momentum above this → BULL (default +5% over the trend window).
    bear_threshold:
        Momentum below this → BEAR (default −5% over the trend window).

    Returns
    -------
    list[dict[str, Any]]
        One dict per input row: ``{"date", "close", "regime"}``.
        ``regime`` is ``None`` for bars with insufficient lookback.

    Examples
    --------
    >>> rows = [{"date": f"2022-{m:02d}-01", "close": 100.0 + m} for m in range(1, 90)]
    >>> labeled = label_regimes(rows)
    >>> labeled[-1]["regime"] in {"bull", "bear", "range", "high_vol"}
    True
    """
    closes = [float(r["close"]) for r in rows]
    n = len(closes)

    # ── Step 1: compute daily log-returns ─────────────────────────────────────
    log_rets: list[float | None] = [None]
    for i in range(1, n):
        if closes[i - 1] > 0:
            log_rets.append(math.log(closes[i] / closes[i - 1]))
        else:
            log_rets.append(None)

    # ── Step 2: rolling realised vol (annualised) ─────────────────────────────
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

    # ── Step 3: high-vol threshold (percentile of non-None vols) ─────────────
    valid_vols = [v for v in rolling_vol if v is not None]
    if valid_vols:
        sorted_vols = sorted(valid_vols)
        idx = int(high_vol_pct * len(sorted_vols))
        idx = min(idx, len(sorted_vols) - 1)
        vol_threshold = sorted_vols[idx]
    else:
        vol_threshold = float("inf")  # no threshold → never high-vol

    # ── Step 4: momentum signal ───────────────────────────────────────────────
    momentum: list[float | None] = []
    for i in range(n):
        if i < trend_window:
            momentum.append(None)
        elif closes[i - trend_window] > 0:
            momentum.append(closes[i] / closes[i - trend_window] - 1.0)
        else:
            momentum.append(None)

    # ── Step 5: assign regime ─────────────────────────────────────────────────
    result: list[dict[str, Any]] = []
    for i in range(n):
        vol = rolling_vol[i]
        mom = momentum[i]

        if vol is None or mom is None:
            regime = None
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


def get_current_regime(rows: list[dict[str, Any]], **kwargs: Any) -> str | None:
    """Return the regime label for the most recent bar in *rows*.

    Parameters
    ----------
    rows:
        OHLCV bars filtered to ≤ t_now.
    **kwargs:
        Forwarded to :func:`label_regimes`.

    Returns
    -------
    str | None
        One of ``"bull"``, ``"bear"``, ``"range"``, ``"high_vol"``, or
        ``None`` if there is insufficient lookback.
    """
    if not rows:
        return None
    labeled = label_regimes(rows, **kwargs)
    return str(labeled[-1]["regime"]) if labeled[-1]["regime"] is not None else None
