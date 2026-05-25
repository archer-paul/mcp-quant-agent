"""Systematic baseline strategies — vectorised with pandas.

Three baselines serve as the yardstick against which the LLM agent is evaluated.

Strategies
----------
1. **Buy & Hold** — buy at first bar's close, hold to last bar's close.
2. **Time-series momentum** — go long when ``lookback``-day return > 0, flat
   otherwise.  Default lookback: 252 days (≈ 12 months).
3. **Mean-reversion (Bollinger Bands)** — enter long when close < lower band,
   exit when close > upper band, hold through the position.

Design constraints
------------------
- **Signal shift**: signals are computed from the close of bar *t* and executed
  at the open of bar *t + 1* (``exec_signal = signal.shift(1)``).  Any test
  that removes or bypasses this shift would fail
  ``TestSignalShift.test_signal_shifted_one_bar`` — the structural look-ahead
  guard for baselines.
- **Transaction costs**: 5 bps commission + 5 bps slippage per one-way trade
  (10 bps round-trip ≈ institutional mid-cap equity cost).
- **Metrics**: uses ``eval/financial.py`` exclusively — no duplication.
- **Data source**: ``yfinance_source._fetch_raw_bars`` with ``auto_adjust=False``
  (no split leakage).

Library choice
--------------
Pandas over vectorbt: the daily-bar arithmetic is simple enough that vectorbt
adds no benefit over a pandas Series.  vectorbt is retained as an optional
extra for future portfolio-level work.  See ``docs/DECISIONS.md``.

References
----------
- StockBench (arXiv:2510.02209) uses a comparable baseline set.
- KellyBench warns that single-seed Sharpe estimates are unreliable —
  always use ``bootstrap_sharpe_ci`` from ``eval/financial.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from mcp_quant_agent.eval.financial import bootstrap_sharpe_ci, compute_all_metrics
from mcp_quant_agent.mcp_servers.data.yfinance_source import _fetch_raw_bars

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMISSION_BPS: float = 5.0  # one-way commission (basis points)
SLIPPAGE_BPS: float = 5.0  # one-way slippage  (basis points)
#: Total one-way cost as a fraction (10 bps = 0.0010)
COST_PER_TRADE: float = (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetch unadjusted OHLCV data and return a DataFrame indexed by date.

    Returns ``None`` if no data is available.
    """
    try:
        bars = _fetch_raw_bars(ticker, start_date, end_date, "1d")
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", ticker, exc)
        return None
    if not bars:
        return None
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


# ---------------------------------------------------------------------------
# Signal helpers (internal, but named with single underscore for testability)
# ---------------------------------------------------------------------------


def _momentum_signal(close: pd.Series, lookback: int) -> pd.Series:
    """Raw momentum signal (NOT shifted — shift happens in the strategy).

    Returns a binary Series: 1.0 when the ``lookback``-day return is positive
    (go long), 0.0 otherwise (flat).  First ``lookback`` bars are NaN (no signal
    available yet → treated as 0 by the caller after ``.shift(1).fillna(0)``).
    """
    return (close / close.shift(lookback) - 1.0).gt(0.0).astype(float)


def _bb_signal(close: pd.Series, window: int, num_std: float) -> pd.Series:
    """Raw Bollinger-band mean-reversion signal (NOT shifted).

    Entry: close < lower band → long (1.0)
    Exit:  close > upper band → flat (0.0)
    Hold:  between the bands while in position.

    Uses population std (``ddof=0``) to match the standard BB definition.
    The forward loop ensures causal signal generation (no look-ahead within
    the signal computation itself).
    """
    rolling_mean = close.rolling(window, min_periods=window).mean()
    rolling_std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std

    signal = pd.Series(0.0, index=close.index, dtype=float)
    in_position = False
    for i in range(len(close)):
        c = close.iloc[i]
        lo = lower.iloc[i]
        hi = upper.iloc[i]
        if pd.isna(lo):
            # Insufficient data for the rolling window — stay flat
            signal.iloc[i] = 0.0
            continue
        if not in_position and c < lo:
            in_position = True
        elif in_position and c > hi:
            in_position = False
        signal.iloc[i] = 1.0 if in_position else 0.0
    return signal


# ---------------------------------------------------------------------------
# NAV simulation
# ---------------------------------------------------------------------------


def _simulate_nav(
    close: pd.Series,
    exec_signal: pd.Series,
    initial_cash: float,
    cost_per_trade: float = COST_PER_TRADE,
) -> tuple[list[float], int, float]:
    """Simulate a strategy NAV given a pre-shifted execution signal.

    Parameters
    ----------
    close:
        Daily close prices.
    exec_signal:
        Pre-shifted signal Series (already ``signal.shift(1).fillna(0)``).
        Values are 0.0 (flat) or 1.0 (long).
    initial_cash:
        Starting NAV.
    cost_per_trade:
        One-way friction per dollar traded (default: 10 bps total).

    Returns
    -------
    (nav_series, n_trades, daily_turnover)
        - ``nav_series``: list of floats (equity curve, same length as close)
        - ``n_trades``: number of position changes
        - ``daily_turnover``: mean absolute daily position change
    """
    daily_ret = close.pct_change()

    # Strategy gross return = position × price change
    strat_ret = exec_signal * daily_ret

    # Transaction cost: charged when position changes (buy or sell)
    pos_change = exec_signal.diff().abs().fillna(0.0)
    cost = pos_change * cost_per_trade

    # Net return per day
    net_ret = (strat_ret - cost).fillna(0.0)

    # NAV cumulative product (starts at initial_cash)
    nav = initial_cash * (1.0 + net_ret).cumprod()
    nav.iloc[0] = initial_cash  # fix first bar (pct_change is NaN there)

    n_trades = int((pos_change > 0.0).sum())
    daily_turnover = float(pos_change.mean())
    return nav.tolist(), n_trades, daily_turnover


# ---------------------------------------------------------------------------
# Strategy functions (operate on a pre-loaded DataFrame)
# ---------------------------------------------------------------------------


def buy_and_hold(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
) -> dict[str, Any]:
    """Buy-and-hold baseline.

    Buys at the close of the first bar and holds to the close of the last bar.
    No transaction costs for intermediate bars (only the two bookend trades).

    Parameters
    ----------
    df:
        OHLCV DataFrame indexed by date (from :func:`_load_prices`).
    initial_cash:
        Starting capital.

    Returns
    -------
    dict with keys: ``nav_series``, ``metrics``, ``sharpe_ci``, ``n_trades``,
    ``turnover``, ``strategy``.
    """
    close = df["close"]
    # NAV = initial_cash × (close[t] / close[0])
    nav_series = (initial_cash * close / close.iloc[0]).tolist()
    metrics = compute_all_metrics(nav_series)
    sharpe_ci = bootstrap_sharpe_ci(nav_series)
    return {
        "strategy": "buy_and_hold",
        "nav_series": nav_series,
        "metrics": metrics,
        "sharpe_ci": sharpe_ci,
        "n_trades": 2,  # buy at start + sell at end
        "turnover": round(2.0 / max(len(nav_series) - 1, 1), 6),
    }


def momentum_ts(
    df: pd.DataFrame,
    lookback: int = 252,
    initial_cash: float = 100_000.0,
) -> dict[str, Any]:
    """Time-series momentum baseline.

    Go long when the ``lookback``-day return is positive; flat otherwise.
    Signal is computed at the **close of bar t** and executed at **bar t+1**
    (``exec_signal = signal.shift(1)``).

    Parameters
    ----------
    df:
        OHLCV DataFrame indexed by date.
    lookback:
        Return lookback in trading days (default 252 ≈ 12 months).
    initial_cash:
        Starting capital.
    """
    close = df["close"]
    if len(close) <= lookback:
        logger.warning(
            "momentum_ts: only %d bars, need > %d for lookback. Returning flat NAV.",
            len(close),
            lookback,
        )
        nav_series = [initial_cash] * len(close)
        metrics = compute_all_metrics(nav_series)
        return {
            "strategy": "momentum_ts",
            "nav_series": nav_series,
            "metrics": metrics,
            "sharpe_ci": bootstrap_sharpe_ci(nav_series),
            "n_trades": 0,
            "turnover": 0.0,
        }

    raw_signal = _momentum_signal(close, lookback)
    exec_signal = raw_signal.shift(1).fillna(0.0)  # ← THE SHIFT: no same-day execution
    nav_series, n_trades, turnover = _simulate_nav(close, exec_signal, initial_cash)
    metrics = compute_all_metrics(nav_series)
    return {
        "strategy": "momentum_ts",
        "nav_series": nav_series,
        "metrics": metrics,
        "sharpe_ci": bootstrap_sharpe_ci(nav_series),
        "n_trades": n_trades,
        "turnover": round(turnover, 6),
    }


def mean_reversion_bb(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    initial_cash: float = 100_000.0,
) -> dict[str, Any]:
    """Mean-reversion via Bollinger Bands.

    Enter long when close < lower band; exit when close > upper band.
    Signal is computed at **close of bar t** and executed at **bar t+1**.

    Parameters
    ----------
    df:
        OHLCV DataFrame indexed by date.
    window:
        Rolling window for the moving average (default 20 days).
    num_std:
        Number of standard deviations for the bands (default 2.0).
    initial_cash:
        Starting capital.
    """
    close = df["close"]
    raw_signal = _bb_signal(close, window, num_std)
    exec_signal = raw_signal.shift(1).fillna(0.0)  # ← THE SHIFT
    nav_series, n_trades, turnover = _simulate_nav(close, exec_signal, initial_cash)
    metrics = compute_all_metrics(nav_series)
    return {
        "strategy": "mean_reversion_bb",
        "nav_series": nav_series,
        "metrics": metrics,
        "sharpe_ci": bootstrap_sharpe_ci(nav_series),
        "n_trades": n_trades,
        "turnover": round(turnover, 6),
    }


# ---------------------------------------------------------------------------
# Portfolio-level runner
# ---------------------------------------------------------------------------


def run_all_baselines(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 100_000.0,
    momentum_lookback: int = 252,
    bb_window: int = 20,
    bb_num_std: float = 2.0,
) -> dict[str, Any]:
    """Run all three baselines for each ticker and return a results table.

    Parameters
    ----------
    tickers:
        List of equity tickers.
    start_date, end_date:
        ISO-8601 backtest window.
    initial_cash:
        Starting capital per ticker.
    momentum_lookback:
        Lookback for time-series momentum (default 252 = 12 months).
    bb_window, bb_num_std:
        Bollinger Band parameters.

    Returns
    -------
    dict with structure::

        {
            "per_ticker": {
                "<TICKER>": {
                    "buy_and_hold": {...},
                    "momentum_ts": {...},
                    "mean_reversion_bb": {...},
                },
                ...
            },
            "equal_weight": {
                "buy_and_hold": {...},
                "momentum_ts": {...},
                "mean_reversion_bb": {...},
            },
            "meta": {"tickers": [...], "start_date": ..., "end_date": ...},
        }
    """
    strategy_fns: dict[str, Callable[[pd.DataFrame], dict[str, Any]]] = {
        "buy_and_hold": lambda p: buy_and_hold(p, initial_cash),
        "momentum_ts": lambda p: momentum_ts(p, momentum_lookback, initial_cash),
        "mean_reversion_bb": lambda p: mean_reversion_bb(p, bb_window, bb_num_std, initial_cash),
    }

    per_ticker: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        logger.info(
            "Loading price data for %s (%s → %s)...", ticker, start_date, end_date
        )
        df = _load_prices(ticker, start_date, end_date)
        if df is None or len(df) < bb_window + 1:
            logger.warning(
                "Skipping %s — insufficient data (%s bars).",
                ticker,
                0 if df is None else len(df),
            )
            continue
        ticker_results: dict[str, Any] = {}
        for name, fn in strategy_fns.items():
            try:
                ticker_results[name] = fn(df)
            except Exception as exc:
                logger.error("Strategy %s on %s failed: %s", name, ticker, exc)
        per_ticker[ticker] = ticker_results

    # Equal-weighted aggregate: average NAV series across tickers
    equal_weight: dict[str, Any] = {}
    valid_tickers = list(per_ticker.keys())
    if valid_tickers:
        for strategy_name in strategy_fns:
            nav_lists = [
                per_ticker[t][strategy_name]["nav_series"]
                for t in valid_tickers
                if strategy_name in per_ticker[t]
            ]
            if nav_lists:
                min_len = min(len(n) for n in nav_lists)
                avg_nav = [
                    sum(n[i] for n in nav_lists) / len(nav_lists)
                    for i in range(min_len)
                ]
                eq_metrics = compute_all_metrics(avg_nav)
                eq_ci = bootstrap_sharpe_ci(avg_nav)
                equal_weight[strategy_name] = {
                    "strategy": strategy_name,
                    "nav_series": avg_nav,
                    "metrics": eq_metrics,
                    "sharpe_ci": eq_ci,
                    "n_trades": int(
                        sum(
                            per_ticker[t][strategy_name].get("n_trades", 0)
                            for t in valid_tickers
                        )
                        / max(len(valid_tickers), 1)
                    ),
                    "turnover": round(
                        sum(
                            per_ticker[t][strategy_name].get("turnover", 0.0)
                            for t in valid_tickers
                        )
                        / max(len(valid_tickers), 1),
                        6,
                    ),
                }

    return {
        "per_ticker": per_ticker,
        "equal_weight": equal_weight,
        "meta": {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "cost_per_trade_bps": (COMMISSION_BPS + SLIPPAGE_BPS),
        },
    }
