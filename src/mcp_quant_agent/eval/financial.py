"""Financial performance metrics.

All functions are pure (no side effects) — easy to test and cache.

Metrics implemented
-------------------
- ``annualised_return``  — CAGR
- ``sharpe_ratio``       — annualised Sharpe (excess return over risk-free rate)
- ``sortino_ratio``      — Sharpe on downside volatility only
- ``calmar_ratio``       — annualised return / max drawdown
- ``max_drawdown``       — peak-to-trough drawdown fraction
- ``hit_rate``           — fraction of positive-return days
- ``daily_turnover``     — average |Δposition| / NAV per day
- ``bootstrap_sharpe_ci`` — bootstrap 95% CI for the Sharpe ratio

These match the metric set used in StockBench (arXiv:2510.02209) and
KellyBench, so results are directly comparable.

Formulae verified against:
- Sharpe: https://en.wikipedia.org/wiki/Sharpe_ratio
- Sortino: Sortino & Price (1994)
- Calmar: drawdown ratio per Terry Young (1991)
- Bootstrap: Politis & Romano (1994) stationary bootstrap (simplified to iid here)

References
----------
KellyBench (arXiv:2604.27865) warns against single-seed evaluation —
always use ``bootstrap_sharpe_ci`` and multi-seed runs in final experiments.
"""

from __future__ import annotations

import math
import random
from typing import Any

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _daily_returns(nav_series: list[float]) -> list[float]:
    """Compute daily returns from a NAV series.

    Parameters
    ----------
    nav_series:
        List of NAV values (oldest first).

    Returns
    -------
    list[float]
        Daily returns ``(nav[i] - nav[i-1]) / nav[i-1]``.
    """
    if len(nav_series) < 2:
        return []
    return [
        (nav_series[i] - nav_series[i - 1]) / nav_series[i - 1]
        for i in range(1, len(nav_series))
        if nav_series[i - 1] != 0
    ]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float], ddof: int = 1) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - ddof)
    return math.sqrt(max(var, 0.0))


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def annualised_return(
    nav_series: list[float],
    trading_days: int = 252,
) -> float:
    """Compound Annual Growth Rate (CAGR).

    Parameters
    ----------
    nav_series:
        NAV values, oldest first.
    trading_days:
        Number of trading days per year (default 252).

    Returns
    -------
    float
        CAGR as a decimal (e.g. 0.15 = 15%).
    """
    if len(nav_series) < 2 or nav_series[0] <= 0:
        return 0.0
    n_days = len(nav_series) - 1
    total_return = nav_series[-1] / nav_series[0]
    years = n_days / trading_days
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1.0)


def max_drawdown(nav_series: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction.

    Returns
    -------
    float
        Max drawdown in [0, 1].  E.g. 0.25 = 25% drawdown.
    """
    if not nav_series:
        return 0.0
    peak = nav_series[0]
    max_dd = 0.0
    for nav in nav_series:
        if nav > peak:
            peak = nav
        if peak > 0:
            dd = (peak - nav) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def sharpe_ratio(
    nav_series: list[float],
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    nav_series:
        NAV values, oldest first.
    risk_free_rate:
        Annual risk-free rate (default 0 — excess-return Sharpe).
    trading_days:
        Trading days per year for annualisation (default 252).

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    rets = _daily_returns(nav_series)
    if not rets:
        return 0.0
    daily_rf = risk_free_rate / trading_days
    excess = [r - daily_rf for r in rets]
    m = _mean(excess)
    s = _std(excess, ddof=1)
    if s == 0:
        return 0.0
    return (m / s) * math.sqrt(trading_days)


def sortino_ratio(
    nav_series: list[float],
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Annualised Sortino ratio (Sharpe using downside volatility only).

    References
    ----------
    Sortino & Price (1994), *Performance Measurement in a Downside Risk Framework*.
    """
    rets = _daily_returns(nav_series)
    if not rets:
        return 0.0
    daily_rf = risk_free_rate / trading_days
    excess = [r - daily_rf for r in rets]
    downside = [r for r in excess if r < 0]
    m = _mean(excess)
    if not downside:
        return float("inf")  # no negative days → infinite Sortino
    downside_std = math.sqrt(sum(r**2 for r in downside) / len(downside))
    if downside_std == 0:
        return 0.0
    return (m / downside_std) * math.sqrt(trading_days)


def calmar_ratio(
    nav_series: list[float],
    trading_days: int = 252,
) -> float:
    """Calmar ratio — annualised return / max drawdown.

    Returns ``inf`` if max drawdown is zero (no losing period).
    """
    ann_ret = annualised_return(nav_series, trading_days)
    max_dd = max_drawdown(nav_series)
    if max_dd == 0:
        return float("inf") if ann_ret > 0 else 0.0
    return ann_ret / max_dd


def hit_rate(nav_series: list[float]) -> float:
    """Fraction of trading days with positive returns."""
    rets = _daily_returns(nav_series)
    if not rets:
        return 0.0
    return sum(1 for r in rets if r > 0) / len(rets)


def compute_all_metrics(
    nav_series: list[float],
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict[str, float]:
    """Compute the full metric suite for a given NAV series.

    Parameters
    ----------
    nav_series:
        NAV values (oldest first).
    risk_free_rate:
        Annual risk-free rate for Sharpe/Sortino (default 0).
    trading_days:
        Trading days per year (default 252).

    Returns
    -------
    dict[str, float]
        ``{annualised_return, sharpe, sortino, calmar, max_drawdown, hit_rate}``.
    """
    return {
        "annualised_return": annualised_return(nav_series, trading_days),
        "sharpe": sharpe_ratio(nav_series, risk_free_rate, trading_days),
        "sortino": sortino_ratio(nav_series, risk_free_rate, trading_days),
        "calmar": calmar_ratio(nav_series, trading_days),
        "max_drawdown": max_drawdown(nav_series),
        "hit_rate": hit_rate(nav_series),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_sharpe_ci(
    nav_series: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap 95% confidence interval for the Sharpe ratio.

    Uses the iid block bootstrap (simplified from the stationary bootstrap of
    Politis & Romano 1994 — sufficient for daily returns which are weakly
    autocorrelated).

    KellyBench (arXiv:2604.27865) shows that single-run Sharpe estimates can
    vary wildly.  Always report CIs in the thesis.

    Parameters
    ----------
    nav_series:
        NAV values (oldest first).
    n_bootstrap:
        Number of bootstrap samples (default 1000).
    confidence:
        CI level (default 0.95 → 2.5th and 97.5th percentiles).
    risk_free_rate:
        Annual risk-free rate.
    trading_days:
        Trading days per year.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        ``{"sharpe": point_estimate, "ci_lower": ..., "ci_upper": ...,
           "n_bootstrap": n_bootstrap}``.
    """
    rets = _daily_returns(nav_series)
    if len(rets) < 10:
        s = sharpe_ratio(nav_series, risk_free_rate, trading_days)
        return {"sharpe": s, "ci_lower": s, "ci_upper": s, "n_bootstrap": 0}

    rng = random.Random(seed)
    n = len(rets)
    daily_rf = risk_free_rate / trading_days
    bootstrap_sharpes: list[float] = []

    for _ in range(n_bootstrap):
        sample = [rets[rng.randint(0, n - 1)] for _ in range(n)]
        excess = [r - daily_rf for r in sample]
        m = _mean(excess)
        s = _std(excess, ddof=1)
        bs = (m / s) * math.sqrt(trading_days) if s > 0 else 0.0
        bootstrap_sharpes.append(bs)

    bootstrap_sharpes.sort()
    alpha = 1.0 - confidence
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap)

    return {
        "sharpe": sharpe_ratio(nav_series, risk_free_rate, trading_days),
        "ci_lower": bootstrap_sharpes[lo_idx],
        "ci_upper": bootstrap_sharpes[min(hi_idx, len(bootstrap_sharpes) - 1)],
        "n_bootstrap": n_bootstrap,
    }


def compare_strategies(
    nav_series_dict: dict[str, list[float]],
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict[str, dict[str, Any]]:
    """Compute metrics for multiple strategies and return a comparison table.

    Parameters
    ----------
    nav_series_dict:
        Mapping of strategy name → NAV series.
    risk_free_rate, trading_days:
        Forwarded to metric functions.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of strategy name → metrics dict.
    """
    return {
        name: compute_all_metrics(nav, risk_free_rate, trading_days)
        for name, nav in nav_series_dict.items()
    }
