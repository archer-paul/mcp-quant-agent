"""FastMCP data server — exposes price history, news, and t_now to MCP clients.

Run standalone::

    python -m mcp_quant_agent.mcp_servers.data.server

Register with Claude Desktop::

    claude mcp add --transport stdio --scope user \\
        --env FINNHUB_API_KEY=<key> \\
        mcp-quant-data -- python -m mcp_quant_agent.mcp_servers.data.server

Anti-look-ahead
---------------
All tools call the underlying source functions (``yfinance_source``,
``finnhub_source``) which enforce the t_now filter via the module-level
clock.  The clock must be set before the server is used in a backtest; in
live mode, it is set to the current time at server startup.
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from mcp_quant_agent.clock import SimulationClock, set_clock
from mcp_quant_agent.mcp_servers.data.finnhub_source import (
    get_earnings_calendar as _get_earnings,
    get_news_items as _get_news,
    get_recent_news as _recent_news,
)
from mcp_quant_agent.mcp_servers.data.yfinance_source import (
    get_latest_price as _get_latest_price,
    get_price_history as _get_price_history,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("mcp-quant-data")


# ---------------------------------------------------------------------------
# Price tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_price_history(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> str:
    """Return OHLCV price bars for *ticker* from *start_date* to *end_date*.

    All bars are filtered to dates ≤ t_now — no future data leaks through.

    Args:
        ticker: Equity ticker symbol (e.g. "AAPL", "MSFT").
        start_date: ISO-8601 start date, e.g. "2022-01-03".
        end_date: ISO-8601 end date (inclusive), e.g. "2022-06-15".
        interval: Bar interval: "1d" (daily, default), "1h", "5m", etc.

    Returns:
        JSON array of {date, open, high, low, close, volume} objects,
        sorted ascending by date, with date <= t_now.
    """
    bars = _get_price_history(ticker, start_date, end_date, interval)
    return json.dumps(bars, indent=2)


@mcp.tool()
async def get_latest_price(ticker: str) -> str:
    """Return the most recent closing price at or before t_now.

    Args:
        ticker: Equity ticker symbol (e.g. "AAPL").

    Returns:
        JSON object {"ticker": "AAPL", "close": 150.25, "date": "2022-06-15"}.
    """
    price = _get_latest_price(ticker)
    from mcp_quant_agent.clock import t_now

    return json.dumps(
        {
            "ticker": ticker,
            "close": price,
            "as_of": t_now().date().isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# News tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_news(
    ticker: str,
    from_date: str,
    to_date: str,
) -> str:
    """Return company news for *ticker*, filtered to news dated ≤ t_now.

    Uses Finnhub. Requires FINNHUB_API_KEY in environment.

    Args:
        ticker: Equity ticker symbol.
        from_date: ISO-8601 start date (e.g. "2022-06-08").
        to_date: ISO-8601 end date.

    Returns:
        JSON array of {datetime, headline, summary, source, url} objects.
    """
    items = _get_news(ticker, from_date, to_date)
    return json.dumps(items, indent=2)


@mcp.tool()
async def get_recent_news(ticker: str, days_back: int = 7) -> str:
    """Return the last *days_back* days of news relative to t_now.

    Args:
        ticker: Equity ticker symbol.
        days_back: How many calendar days to look back (default 7).

    Returns:
        JSON array of {datetime, headline, summary, source, url} objects.
    """
    items = _recent_news(ticker, days_back=days_back)
    return json.dumps(items, indent=2)


# ---------------------------------------------------------------------------
# Clock tools (for orchestrator / backtest engine)
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_t_now() -> str:
    """Return the current simulation time (t_now).

    In backtest mode this is the current bar date; in live mode it is
    approximately the current wall-clock time.

    Returns:
        JSON object {"t_now": "2022-06-15T00:00:00"}.
    """
    from mcp_quant_agent.clock import t_now

    return json.dumps({"t_now": t_now().isoformat()})


@mcp.tool()
async def get_earnings_calendar(ticker: str) -> str:
    """Return announced earnings dates for *ticker* (≤ t_now only).

    Args:
        ticker: Equity ticker symbol.

    Returns:
        JSON array of {date, ticker, eps_estimate, revenue_estimate}.
    """
    items = _get_earnings(ticker)
    return json.dumps(items, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _init_live_clock() -> None:
    """Set the clock to the current wall-clock time for live / standalone use."""
    import datetime as dt

    set_clock(SimulationClock(dt.datetime.utcnow()))
    logger.info("mcp-quant-data: live clock set to %s", dt.datetime.utcnow().isoformat())


def main() -> None:
    """Start the MCP data server (live mode — clock = now)."""
    _init_live_clock()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
