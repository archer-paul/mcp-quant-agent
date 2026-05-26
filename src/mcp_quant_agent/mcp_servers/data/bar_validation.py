"""Bar-level data quality validation — the data-integrity guard.

Purpose
-------
External price sources (yfinance, Finnhub) occasionally return malformed
data: ``pd.Series`` objects instead of scalars (yfinance MultiIndex format
change), negative prices, high < low, or single-bar spikes that are
mathematically impossible.

This module provides ``validate_bar`` which is called on **every bar before
it enters the backtest engine or cache**.  Any violation raises ``ValueError``
with the ticker, date, field name, and bad value so the problem is immediately
diagnosable.

**Fail loud** is the project convention (see CLAUDE.md).  A silent skip
(``continue`` on ``TypeError``) is far worse than a crash because it
creates a stale-data position with no warning.

Post-mortem context
-------------------
Run #1 (gpt-4-1-mini_20260526_160051) produced a catastrophic NAV collapse
because yfinance started returning ``pd.Series`` objects for individual OHLCV
columns in multi-ticker download results.  The previous code caught
``TypeError`` and *skipped* the bar (logged only as WARNING).  Some bars
were accepted with wrong float values extracted from the head of a Series.
This validation layer makes that class of bug structurally impossible.

See ``docs/DECISIONS.md`` — "2026-05 — Bar validation layer".

Invariants
----------
1. ``open``, ``high``, ``low``, ``close`` are finite, positive scalar floats.
2. ``high >= low``.
3. ``high >= close >= low``.
4. ``close`` in ``[open * 0.5, open * 2.0]`` — spike guard (gross errors only;
   real 50 % intraday moves are extremely rare but theoretically valid).
5. ``volume`` is a non-negative integer-compatible value.
"""

from __future__ import annotations

import math
from typing import Any


def _to_float_scalar(value: Any, field: str, ticker: str, date: str) -> float:
    """Extract a finite float from *value*, rejecting multi-element Series.

    Handles:

    * Plain ``float`` / ``int`` / numpy scalar → direct ``float()`` cast.
    * Single-element ``pd.Series`` → extract via ``.iloc[0]``.
    * Multi-element ``pd.Series`` → **raise** (this is the bug from run #1:
      yfinance MultiIndex format returning Series instead of scalars).
    * NaN / Inf → raise.
    * Any non-convertible type → raise with context.

    Parameters
    ----------
    value:
        The raw value from the data source row.
    field:
        Column name (e.g. ``"close"``), used in the error message.
    ticker:
        Equity ticker (e.g. ``"AAPL"``), used in the error message.
    date:
        ISO-8601 date string, used in the error message.

    Raises
    ------
    ValueError
        On multi-element Series, NaN/Inf, or unconvertible type.
    """
    try:
        import pandas as pd  # noqa: PLC0415

        if isinstance(value, pd.Series):
            if len(value) == 1:
                value = value.iloc[0]
            else:
                raise ValueError(
                    f"Expected scalar for '{field}' on {ticker}/{date}, "
                    f"got {len(value)}-element Series: {value.values!r}. "
                    "This typically means yfinance returned a MultiIndex "
                    "DataFrame — check the download call and column flattening."
                )
    except ImportError:
        pass  # pandas unavailable — fall through to float() below

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot convert '{field}'={value!r} to float "
            f"for {ticker}/{date}: {exc}"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite value for '{field}' on {ticker}/{date}: {result}"
        )
    return result


def validate_bar(bar: dict[str, Any], ticker: str = "") -> None:
    """Validate a single OHLCV bar dict.  Raises ``ValueError`` on violation.

    Called by ``_fetch_raw_bars`` (write path) **and** by ``get_price_history``
    after cache reads (to catch any bars written by older, pre-validation code).

    Parameters
    ----------
    bar:
        Dict with keys ``date``, ``open``, ``high``, ``low``, ``close``,
        ``volume``.  Extra keys are ignored.
    ticker:
        Ticker symbol included in all error messages for fast diagnosis.

    Raises
    ------
    ValueError
        With message containing ticker, date, field name, and bad value.

    Examples
    --------
    >>> validate_bar({"date": "2022-01-03", "open": 100.0, "high": 101.0,
    ...               "low": 99.0, "close": 100.5, "volume": 1_000_000}, "AAPL")
    >>> # passes silently

    >>> import pandas as pd
    >>> validate_bar({"date": "2022-01-03", "open": 100.0, "high": 101.0,
    ...               "low": 99.0,
    ...               "close": pd.Series([100.5, 200.0]),
    ...               "volume": 1_000_000}, "AAPL")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Expected scalar for 'close' on AAPL/2022-01-03 ...
    """
    date = str(bar.get("date", "unknown"))

    # ── 1. Extract and type-check all price fields ───────────────────────────
    price_fields = ("open", "high", "low", "close")
    prices: dict[str, float] = {}
    for field in price_fields:
        raw = bar.get(field)
        if raw is None:
            raise ValueError(
                f"Bar validation failed for {ticker}/{date}: "
                f"missing required field '{field}'"
            )
        v = _to_float_scalar(raw, field, ticker, date)
        if v <= 0:
            raise ValueError(
                f"Bar validation failed for {ticker}/{date}: "
                f"'{field}'={v} must be > 0 (got non-positive price)"
            )
        prices[field] = v

    # ── 2. Volume ────────────────────────────────────────────────────────────
    raw_vol = bar.get("volume", 0)
    vol = _to_float_scalar(raw_vol, "volume", ticker, date)
    if vol < 0:
        raise ValueError(
            f"Bar validation failed for {ticker}/{date}: "
            f"volume={vol} must be >= 0"
        )

    o, h, lo, c = prices["open"], prices["high"], prices["low"], prices["close"]

    # ── 3. OHLC consistency ──────────────────────────────────────────────────
    if h < lo:
        raise ValueError(
            f"Bar validation failed for {ticker}/{date}: "
            f"high={h} < low={lo}"
        )
    if c > h:
        raise ValueError(
            f"Bar validation failed for {ticker}/{date}: "
            f"close={c} > high={h}"
        )
    if c < lo:
        raise ValueError(
            f"Bar validation failed for {ticker}/{date}: "
            f"close={c} < low={lo}"
        )

    # ── 4. Anti-spike guard (gross errors, not real gaps) ────────────────────
    lo_limit = o * 0.5
    hi_limit = o * 2.0
    if not (lo_limit <= c <= hi_limit):
        raise ValueError(
            f"Bar validation failed for {ticker}/{date}: "
            f"close={c:.4f} outside anti-spike bounds "
            f"[open*0.5={lo_limit:.4f}, open*2={hi_limit:.4f}]. "
            "This rejects implausible within-day price moves (>100% or <-50%). "
            "If the move is genuine, widen the bounds in bar_validation.py."
        )
