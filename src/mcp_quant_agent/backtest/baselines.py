"""Systematic baseline strategies — vectorised via vectorbt (or pure pandas).

Baselines to implement (see docs/PLAN.md J3):
1. **Buy & Hold** — buy at start, hold to end.
2. **Time-series momentum** — go long when 12-month momentum > 0, flat otherwise.
3. **Mean-reversion (Bollinger)** — go long when price < lower band, short when > upper.

These baselines are the yardstick against which the agent is evaluated.
Multi-seed + bootstrap confidence intervals are computed in ``eval/financial.py``.

Status: STUB — implementation in J3 sprint.

References
----------
- vectorbt docs: https://vectorbt.dev
- StockBench (arXiv:2510.02209) uses similar baseline set.
- KellyBench warns against single-seed evaluation — always bootstrap.
"""

from __future__ import annotations

from typing import Any


def buy_and_hold(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_cash: float = 100_000.0,
) -> dict[str, Any]:
    """Buy-and-hold baseline.

    Buys at the open of *start_date* and holds until the close of *end_date*.
    Returns performance metrics via ``eval/financial.py``.

    Status: STUB — not yet implemented.

    Raises
    ------
    NotImplementedError
        Until the implementation sprint (J3).
    """
    raise NotImplementedError(
        "buy_and_hold() not yet implemented — see docs/PLAN.md J3."
    )


def momentum_strategy(
    ticker: str,
    start_date: str,
    end_date: str,
    lookback_days: int = 252,
    initial_cash: float = 100_000.0,
) -> dict[str, Any]:
    """Time-series momentum baseline.

    Go long when the *lookback_days*-return is positive, flat otherwise.
    No shorting (long-only variant matching most StockBench comparisons).

    Status: STUB.

    Raises
    ------
    NotImplementedError
        Until the implementation sprint (J3).
    """
    raise NotImplementedError(
        "momentum_strategy() not yet implemented — see docs/PLAN.md J3."
    )


def mean_reversion_bollinger(
    ticker: str,
    start_date: str,
    end_date: str,
    window: int = 20,
    num_std: float = 2.0,
    initial_cash: float = 100_000.0,
) -> dict[str, Any]:
    """Mean-reversion via Bollinger Bands.

    Buy signal: close < lower band (oversold).
    Sell signal: close > upper band (overbought).

    Status: STUB.

    Raises
    ------
    NotImplementedError
        Until the implementation sprint (J3).
    """
    raise NotImplementedError(
        "mean_reversion_bollinger() not yet implemented — see docs/PLAN.md J3."
    )


def run_all_baselines(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 100_000.0,
) -> dict[str, Any]:
    """Run all three baselines for each ticker and return a results table.

    Returns a dict mapping strategy name → ticker → performance metrics.

    Status: STUB.

    Raises
    ------
    NotImplementedError
        Until the implementation sprint (J3).
    """
    raise NotImplementedError("run_all_baselines() not yet implemented.")
