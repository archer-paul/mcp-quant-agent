"""Finnhub data source — news, earnings, and company info with t_now filter.

Source: Salvi reference code (``gen-ai-imperial/code/week3_agent/market_data.py``).
Adaptations:
- Added t_now filter (``clock.filter_rows``) on all returned items.
- Added type annotations and docstrings.
- News datetime is normalised to ISO-8601 string for clock comparison.
- Added ``_fetch_raw_news`` as a patchable boundary for tests.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from mcp_quant_agent.clock import get_clock
from mcp_quant_agent.config import settings

logger = logging.getLogger(__name__)

_news_cache: Any | None = None


def _get_news_cache() -> Any:
    """Return the module-level NewsCache instance."""
    global _news_cache
    if _news_cache is None:
        from mcp_quant_agent.mcp_servers.data.cache import NewsCache

        _news_cache = NewsCache()
    return _news_cache


def _get_finnhub_client() -> Any:
    """Return a Finnhub client, raising if the API key is not configured."""
    try:
        import finnhub
    except ImportError as exc:
        raise RuntimeError(
            "finnhub-python is not installed. "
            "Run: uv pip install 'mcp-quant-agent[dev]'"
        ) from exc

    key = settings.finnhub_api_key
    if not key:
        raise RuntimeError(
            "FINNHUB_API_KEY is not set. "
            "Add it to your .env file (see .env.example)."
        )
    return finnhub.Client(api_key=key)


# ---------------------------------------------------------------------------
# Raw fetch functions (patchable in tests)
# ---------------------------------------------------------------------------


def _fetch_raw_news(
    ticker: str,
    from_date: str,
    to_date: str,
) -> list[dict[str, Any]]:
    """Fetch company news from Finnhub.

    This is the only function in this module that calls the Finnhub API.
    It is kept thin so tests can patch it without importing finnhub.

    Returns
    -------
    list[dict[str, Any]]
        News items with ``datetime`` key (ISO-8601 string).
        May include items beyond ``t_now`` — the caller filters them.
    """
    client = _get_finnhub_client()
    raw: list[dict[str, Any]] = client.company_news(
        ticker, _from=from_date, to=to_date
    )
    items: list[dict[str, Any]] = []
    for item in raw[:50]:  # cap at 50 items
        try:
            ts = dt.datetime.fromtimestamp(item["datetime"]).isoformat(
                timespec="seconds"
            )
        except (KeyError, TypeError, ValueError):
            continue
        items.append(
            {
                "datetime": ts,
                "headline": str(item.get("headline", "")),
                "summary": str(item.get("summary", ""))[:300],
                "source": str(item.get("source", "")),
                "url": str(item.get("url", "")),
                "ticker": ticker,
            }
        )
    return items


def _fetch_raw_earnings(ticker: str) -> list[dict[str, Any]]:
    """Fetch upcoming earnings calendar from Finnhub.

    Returns raw earnings items.  May include future earnings — caller filters.
    """
    client = _get_finnhub_client()
    today = dt.date.today()
    from_date = today.isoformat()
    to_date = (today + dt.timedelta(days=90)).isoformat()
    raw = client.earnings_calendar(_from=from_date, to=to_date, symbol=ticker)
    items: list[dict[str, Any]] = []
    for item in raw.get("earningsCalendar", []):
        items.append(
            {
                "date": str(item.get("date", "")),
                "ticker": str(item.get("symbol", ticker)),
                "eps_estimate": item.get("epsEstimate"),
                "revenue_estimate": item.get("revenueEstimate"),
            }
        )
    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_news_items(
    ticker: str,
    from_date: str,
    to_date: str,
) -> list[dict[str, Any]]:
    """Return news items for *ticker* from *from_date* to *to_date*, filtered to t_now.

    Parameters
    ----------
    ticker:
        Equity ticker (e.g. ``"AAPL"``).
    from_date:
        ISO-8601 start date (e.g. ``"2022-06-01"``).
    to_date:
        ISO-8601 end date.  Items beyond t_now are stripped.

    Returns
    -------
    list[dict[str, Any]]
        News items with ``datetime ≤ t_now``.
        Keys: ``datetime``, ``headline``, ``summary``, ``source``, ``url``, ``ticker``.

    Raises
    ------
    RuntimeError
        If no simulation clock is set or the Finnhub key is missing.
    """
    clock = get_clock()
    raw = _fetch_raw_news(ticker, from_date, to_date)
    return clock.filter_rows(raw, date_key="datetime")


def get_news_items_cache_first(
    ticker: str,
    from_date: str,
    to_date: str,
    *,
    offline: bool = True,
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Return timestamped news from cache first, filtered to ``t_now``.

    In offline mode, a cache miss raises loudly and never calls Finnhub. This
    is the PM-backtest path: no network fetch should happen while replaying a
    historical decision loop.
    """
    clock = get_clock()
    news_cache = cache if cache is not None else _get_news_cache()

    cached = news_cache.read_range_filtered(ticker, from_date, to_date)
    if cached or news_cache.exists(ticker):
        return cached

    if offline:
        raise RuntimeError(
            f"News cache miss for {ticker} {from_date}->{to_date}. "
            "Offline mode forbids Finnhub fetch during backtests."
        )

    raw = _fetch_raw_news(ticker, from_date, to_date)
    if raw:
        news_cache.merge_and_write(ticker, raw)
    return clock.filter_rows(raw, date_key="datetime")


def get_recent_news(
    ticker: str,
    days_back: int = 7,
) -> list[dict[str, Any]]:
    """Return news from the last *days_back* calendar days (relative to t_now).

    Convenience wrapper around :func:`get_news_items` that computes the date
    range from the current t_now.
    """
    clock = get_clock()
    end_date = clock.t_now.date().isoformat()
    start_date = (clock.t_now.date() - dt.timedelta(days=days_back)).isoformat()
    return get_news_items(ticker, start_date, end_date)


def get_earnings_calendar(ticker: str) -> list[dict[str, Any]]:
    """Return upcoming earnings dates, filtered to dates ≤ t_now (i.e. announced).

    In a backtest, future earnings announcements must be invisible.  We filter
    the calendar by t_now so only announcements that have already happened are
    visible to the agent.

    Note: Finnhub's earnings calendar only covers ~90 days into the future from
    the real today.  In deep backtest scenarios, this may return an empty list.
    """
    clock = get_clock()
    items = _fetch_raw_earnings(ticker)
    # Earnings calendar uses "date" key (date only, no time component)
    return clock.filter_rows(items, date_key="date")
