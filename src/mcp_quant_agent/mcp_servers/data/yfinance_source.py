"""yfinance data source with t_now filter and parquet cache.

Design decisions (see docs/DECISIONS.md for full rationale)
------------------------------------------------------------
1. **``auto_adjust=False``**: yfinance's default (``auto_adjust=True``) applies
   split/dividend adjustments based on *current* corporate-action factors.  In a
   backtest set in 2022, this encodes splits that happened in 2023–2025 — a
   look-ahead bias.  We disable auto-adjustment and work with unadjusted prices.
   Consequence: returns are not adjusted for splits, which slightly distorts
   long-period comparisons.  This is documented and preferable to silent leakage.

2. **t_now filter applied in ``get_price_history``**: even if the raw fetch or the
   parquet cache returns bars beyond t_now, they are stripped by
   ``clock.filter_rows`` before the caller sees them.

3. **Cache-then-filter**: the parquet cache stores raw data (no t_now filter at
   write time).  This allows re-using the same cache files for backtest runs at
   different ``t_now`` values.

4. **Incremental cache update**: on cache miss (or stale cache tail), we fetch
   only the missing date range and merge.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
import yfinance as yf

from mcp_quant_agent.clock import get_clock
from mcp_quant_agent.mcp_servers.data.bar_validation import (
    _to_float_scalar,
    validate_bar,
)
from mcp_quant_agent.mcp_servers.data.cache import PriceCache

logger = logging.getLogger(__name__)

# Module-level cache instance (re-uses the default cache_dir from settings)
_cache: PriceCache | None = None


def _get_cache() -> PriceCache:
    global _cache
    if _cache is None:
        _cache = PriceCache()
    return _cache


def _yfinance_end_exclusive(end_date: str, interval: str = "1d") -> str:
    """Convert the wrapper's inclusive daily end date to yfinance's exclusive end."""
    if interval == "1d" and len(end_date) == 10:
        return (dt.date.fromisoformat(end_date) + dt.timedelta(days=1)).isoformat()
    return end_date


# ---------------------------------------------------------------------------
# Raw fetch (side-effect: hits yfinance API — mock this in tests)
# ---------------------------------------------------------------------------


def _check_price_continuity(
    bars: list[dict[str, Any]],
    ticker: str,
    max_ratio: float = 5.0,
) -> None:
    """Raise ValueError if any adjacent-bar close-price ratio exceeds *max_ratio*.

    A ratio > 5× between consecutive daily bars indicates a split-adjusted /
    unadjusted price mix (e.g. NVDA 10:1 split in Jun 2024 causing old cached
    bars at ~$130/share to appear next to newly-fetched bars at ~$13/share).
    Such contamination silently destroys backtest P&L — we fail loudly instead.

    Parameters
    ----------
    bars:
        List of OHLCV bar dicts sorted ascending by date.
    ticker:
        Ticker symbol for error messages.
    max_ratio:
        Maximum allowable close-price ratio between consecutive bars.
        Default 5.0 catches split/unsplit mixes; real overnight gaps are
        far smaller (even extreme circuit-breaker moves rarely exceed 2×).

    Raises
    ------
    ValueError
        If any consecutive pair has a price ratio exceeding max_ratio.
    """
    for i in range(1, len(bars)):
        prev_close = float(bars[i - 1].get("close", 0.0))
        curr_close = float(bars[i].get("close", 0.0))
        if prev_close <= 0 or curr_close <= 0:
            continue
        ratio = max(curr_close / prev_close, prev_close / curr_close)
        if ratio > max_ratio:
            raise ValueError(
                f"Price discontinuity for {ticker}: "
                f"{bars[i - 1]['date']} close={prev_close:.4f} → "
                f"{bars[i]['date']} close={curr_close:.4f} "
                f"(ratio={ratio:.1f}x, threshold={max_ratio}x). "
                "This indicates a split-adjusted/unadjusted price mix in the "
                "parquet cache.  Delete data/cache/prices/ and re-run to force "
                "a fresh fetch with consistent split-adjusted prices."
            )


def _fetch_raw_bars(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> list[dict[str, Any]]:
    """Fetch raw OHLCV bars from yfinance.

    This is the only function in this module that calls the yfinance API.
    It is kept deliberately thin so tests can patch it without mocking the
    entire yfinance package.

    Parameters
    ----------
    ticker:
        Equity ticker symbol.
    start:
        ISO-8601 start date (inclusive), e.g. ``"2022-01-03"``.
    end:
        ISO-8601 end date (exclusive for yfinance — we use the day after the
        desired last date when calling the API, then filter afterward).
    interval:
        yfinance interval string: ``"1d"``, ``"1h"``, ``"5m"``, etc.

    Returns
    -------
    list[dict[str, Any]]
        Raw OHLCV bars.  May include bars beyond ``t_now`` — that is the
        caller's responsibility to filter.

    Notes
    -----
    ``auto_adjust=False`` is non-negotiable.  Do NOT change this to
    ``True`` — it would introduce split/dividend leakage into the backtest.
    See docs/DECISIONS.md.
    """
    df: pd.DataFrame = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,   # CRITICAL: see module docstring
        progress=False,
        repair=False,
    )
    if df.empty:
        logger.debug("_fetch_raw_bars: no data for %s %s→%s", ticker, start, end)
        return []

    # yfinance 0.2.x can return MultiIndex columns when a single ticker is
    # requested with certain parameter combinations.  Flatten to single-level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)  # noqa: PD011

    # De-duplicate column names that can arise after MultiIndex flattening
    # (e.g. yfinance returning 'Close' twice for different price types).
    # Keeping only the first occurrence ensures row["Close"] is a scalar,
    # not a Series — the root cause of the run #1 price-data bug.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    df = df.reset_index()
    bars: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        date_val = row.get("Date") or row.get("Datetime")
        if date_val is None:
            continue
        # Normalise to "YYYY-MM-DD" string
        if hasattr(date_val, "date"):
            date_str = date_val.date().isoformat()
        else:
            date_str = str(date_val)[:10]

        # Fail loud: _to_float_scalar raises on Series/NaN/non-finite.
        # Never silently skip a malformed bar — a skipped bar creates a
        # stale-data position with no diagnostic.  If this raises, the
        # yfinance format has changed and needs investigation.
        bar: dict[str, Any] = {
            "date": date_str,
            "open":   round(_to_float_scalar(row["Open"],   "open",   ticker, date_str), 4),
            "high":   round(_to_float_scalar(row["High"],   "high",   ticker, date_str), 4),
            "low":    round(_to_float_scalar(row["Low"],    "low",    ticker, date_str), 4),
            "close":  round(_to_float_scalar(row["Close"],  "close",  ticker, date_str), 4),
            "volume": int(max(0, _to_float_scalar(row["Volume"], "volume", ticker, date_str))),
        }
        validate_bar(bar, ticker)  # raises ValueError on OHLC inconsistency or spike
        bars.append(bar)

    # Guard against split-adjusted / unadjusted price mix (e.g. NVDA Jun 2024
    # 10:1 split causing alternating $13 / $130 prices in the same series).
    _check_price_continuity(bars, ticker)

    return bars


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_price_history(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Return OHLCV bars for *ticker* from *start_date* to *end_date*, filtered to t_now.

    This is the main entry point for price data.  It:

    1. Checks the parquet cache for existing data.
    2. Fetches any missing tail via ``_fetch_raw_bars``.
    3. Updates the cache with new data.
    4. Applies the t_now filter via ``clock.filter_rows``.

    Parameters
    ----------
    ticker:
        Equity ticker symbol (e.g. ``"AAPL"``).
    start_date:
        ISO-8601 start date, e.g. ``"2022-01-03"``.
    end_date:
        ISO-8601 end date (the last date we want, inclusive).
    interval:
        Bar interval.  Defaults to ``"1d"`` (daily).
    use_cache:
        If ``False``, bypass the parquet cache (useful for one-off fetches).

    Returns
    -------
    list[dict[str, Any]]
        OHLCV bars with ``date ≤ t_now``, sorted ascending by date.
        Keys: ``date``, ``open``, ``high``, ``low``, ``close``, ``volume``.

    Raises
    ------
    RuntimeError
        If no simulation clock has been set.
    """
    clock = get_clock()

    if use_cache:
        cache = _get_cache()
        latest = cache.latest_cached_date(ticker, interval)

        if latest is None or latest < end_date:
            # Cache miss or stale tail — fetch from API
            fetch_start = start_date if latest is None else latest
            raw = _fetch_raw_bars(
                ticker,
                fetch_start,
                _yfinance_end_exclusive(end_date, interval),
                interval,
            )
            if raw:
                cache.merge_and_write(ticker, interval, raw)

        bars = cache.read_all(ticker, interval)
        # Apply date range filter first (don't return data outside requested window)
        bars = [b for b in bars if str(b["date"]) >= start_date]
    else:
        bars = _fetch_raw_bars(
            ticker,
            start_date,
            _yfinance_end_exclusive(end_date, interval),
            interval,
        )

    # Validate bars read from cache — guards against data written by older
    # code before the bar validation layer existed (e.g. run #1 corrupt cache).
    # Raises ValueError if any bar fails; delete the parquet cache and re-fetch.
    for b in bars:
        validate_bar(b, ticker)

    # THE anti-look-ahead filter: strip any bar beyond t_now
    return clock.filter_rows(bars, date_key="date")


def get_latest_price(ticker: str) -> float:
    """Return the most recent close price at or before t_now.

    Fetches the last 10 days of data (to ensure we get at least one bar on
    weekdays) and returns the closing price of the most recent bar.

    Raises
    ------
    ValueError
        If no price data is available for *ticker* at or before t_now.
    RuntimeError
        If no simulation clock has been set.
    """
    clock = get_clock()
    end_date = clock.t_now.date().isoformat()
    # Go back up to 10 calendar days to find the last trading day
    import datetime as dt

    start_date = (clock.t_now.date() - dt.timedelta(days=10)).isoformat()
    bars = get_price_history(ticker, start_date, end_date)
    if not bars:
        raise ValueError(
            f"No price data for {ticker} at or before t_now={end_date}. "
            "Check that the ticker is valid and the data source is accessible."
        )
    return float(bars[-1]["close"])
