"""FastMCP execution server — paper-trading order management.

Run standalone::

    python -m mcp_quant_agent.mcp_servers.execution.server

The server maintains a single Portfolio instance in memory.  State resets
on server restart or via the ``reset_portfolio`` tool (for backtest loops).

Price sourcing
--------------
``place_order`` requires an explicit ``price`` argument.  The orchestrator
is responsible for obtaining the current t_now price via the data MCP server
and passing it here.  This design:
1. Makes the execution server independent of the data server (no circular deps).
2. Makes price assumptions explicit in the agent's reasoning (traceable).
3. Keeps the tool simple and testable.
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from mcp_quant_agent.mcp_servers.execution.paper_trading import Portfolio

logger = logging.getLogger(__name__)

mcp = FastMCP("mcp-quant-execution")

# Module-level portfolio instance (the explicitly-designed in-memory state)
_portfolio: Portfolio = Portfolio()


# ---------------------------------------------------------------------------
# Order tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def place_order(
    ticker: str,
    side: str,
    quantity: float,
    price: float,
) -> str:
    """Place a paper-trading market order at the given price.

    Args:
        ticker: Equity ticker symbol (e.g. "AAPL").
        side: "buy" or "sell".
        quantity: Number of shares (must be positive).
        price: Fill price — the current close at t_now from the data server.
               The orchestrator is responsible for providing this.

    Returns:
        JSON fill confirmation:
        {status, timestamp, ticker, side, quantity, price, notional}.
    """
    try:
        fill = _portfolio.place_order(ticker, side, quantity, price)
    except ValueError as exc:
        return json.dumps({"status": "rejected", "reason": str(exc)})

    return json.dumps(
        {
            "status": "filled",
            "timestamp": fill.timestamp,
            "ticker": fill.ticker,
            "side": fill.side,
            "quantity": fill.quantity,
            "price": fill.price,
            "notional": fill.notional,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Portfolio query tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_portfolio(current_prices_json: str) -> str:
    """Return current portfolio snapshot: positions, cash, PnL, NAV.

    Args:
        current_prices_json: JSON object mapping ticker → current close price,
                             e.g. '{"AAPL": 150.25, "MSFT": 320.10}'.
                             Obtain these from the data server's get_latest_price.

    Returns:
        JSON object with:
        {cash, positions, total_market_value, total_unrealised_pnl,
         nav, total_return_pct, num_trades}.
    """
    current_prices: dict[str, float] = json.loads(current_prices_json)
    summary = _portfolio.get_portfolio_summary(current_prices)
    return json.dumps(summary, indent=2)


@mcp.tool()
async def get_pnl(current_prices_json: str) -> str:
    """Return PnL summary and full trade history.

    Args:
        current_prices_json: JSON object mapping ticker → current close price.

    Returns:
        JSON object with {portfolio_summary, trade_history}.
    """
    current_prices: dict[str, float] = json.loads(current_prices_json)
    summary = _portfolio.get_portfolio_summary(current_prices)
    history = _portfolio.get_order_history()
    return json.dumps(
        {"portfolio_summary": summary, "trade_history": history},
        indent=2,
    )


@mcp.tool()
async def get_nav_series() -> str:
    """Return the recorded NAV time series (equity curve).

    Returns:
        JSON array of {timestamp, nav} objects.
    """
    return json.dumps(_portfolio.get_nav_series(), indent=2)


# ---------------------------------------------------------------------------
# Backtest control tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def reset_portfolio(initial_cash: float = 100_000.0) -> str:
    """Reset the portfolio to a clean state.

    Use this at the start of a new backtest run.

    Args:
        initial_cash: Starting capital (default 100,000).

    Returns:
        JSON confirmation {status: "reset", initial_cash: ...}.
    """
    _portfolio.reset(initial_cash=initial_cash)
    return json.dumps({"status": "reset", "initial_cash": initial_cash})


@mcp.tool()
async def record_nav(current_prices_json: str) -> str:
    """Record the current NAV to the equity curve.

    Call once per bar at bar close to build the equity curve for performance
    metric computation after the backtest.

    Args:
        current_prices_json: JSON object mapping ticker → current close price.

    Returns:
        JSON object {timestamp, nav}.
    """
    current_prices: dict[str, float] = json.loads(current_prices_json)
    nav = _portfolio.record_nav(current_prices)
    latest = _portfolio.nav_history[-1] if _portfolio.nav_history else {}
    return json.dumps({"timestamp": latest.get("timestamp"), "nav": nav})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
