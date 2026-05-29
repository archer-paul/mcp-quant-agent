"""Paper-trading engine — in-memory portfolio management.

Source: Salvi reference code (``gen-ai-imperial/code/week3_agent/paper_trading.py``).
Adaptations:
- Order timestamps use the simulation clock (``t_now``), not ``datetime.now()``.
  This makes the trade history consistent with the backtest timeline and
  allows accurate performance attribution by date.
- NAV time series is recorded after every order for equity-curve computation.
- ``reset()`` method added for clean backtest restarts.
- Full type annotations.
- ``place_order`` accepts an explicit ``price`` argument (required for
  backtesting where the "market price" is the t_now close, not a live fetch).

Design note on price sourcing
-------------------------------
The execution server accepts an explicit *price* per order.  The orchestrator
is responsible for fetching the current price from the data MCP server (which
already applies the t_now filter) and passing it here.  This is more transparent
than having the execution layer do a live price lookup, and avoids the execution
server depending on the data server.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Fill:
    """A single executed order (fill)."""

    timestamp: str  # ISO-8601 simulation time (t_now at fill)
    ticker: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float  # fill price (close at t_now)
    notional: float  # quantity × price
    commission: float = 0.0  # placeholder for realistic friction modelling


@dataclass
class Portfolio:
    """In-memory paper-trading portfolio.

    State
    -----
    - ``cash``               — available cash
    - ``positions``          — current holdings: ticker → quantity
    - ``cost_basis``         — weighted-average cost per share: ticker → price
    - ``order_history``      — chronological list of fills
    - ``nav_history``        — (timestamp, nav) tuples for equity-curve reconstruction
    - ``initial_cash``       — for total-return computation
    - ``commission_bps``     — one-way transaction cost in basis points (applies to
                               both buys and sells, deducted from cash).  Set to 0
                               for cost-free mode (legacy behaviour).  The canonical
                               value is 10 bps (5 commission + 5 slippage) to match
                               the baselines in ``backtest/baselines.py``.
    - ``total_commission``   — cumulative commissions paid (for cost-drag reporting)
    - ``total_turnover``     — cumulative notional traded (for turnover reporting)

    All state resets on ``reset()`` or when a new ``Portfolio`` is created.
    No hidden global state — the portfolio is always passed explicitly.
    """

    cash: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)
    cost_basis: dict[str, float] = field(default_factory=dict)
    order_history: list[Fill] = field(default_factory=list)
    nav_history: list[dict[str, Any]] = field(default_factory=list)
    initial_cash: float = 100_000.0
    commission_bps: float = 0.0  # one-way cost; 10 bps = 5 comm + 5 slippage
    total_commission: float = field(default=0.0, init=False)
    total_turnover: float = field(default=0.0, init=False)

    def reset(self, initial_cash: float = 100_000.0) -> None:
        """Reset the portfolio to a clean state for a new backtest run."""
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions = {}
        self.cost_basis = {}
        self.order_history = []
        self.nav_history = []
        self.total_commission = 0.0
        self.total_turnover = 0.0

    # ── orders ────────────────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        simulation_time: str | None = None,
    ) -> Fill:
        """Execute a market order at *price*.

        Parameters
        ----------
        ticker:
            Equity ticker symbol.
        side:
            ``"buy"`` or ``"sell"``.
        quantity:
            Number of shares (must be positive).
        price:
            Fill price — the closing price at t_now from the data layer.
        simulation_time:
            ISO-8601 timestamp for the fill.  Defaults to the current module-
            level clock time (``t_now()``).  Pass explicitly in tests.

        Returns
        -------
        Fill
            The executed fill.

        Raises
        ------
        ValueError
            If ``side`` is not ``"buy"`` or ``"sell"``, ``quantity ≤ 0``,
            or there is insufficient cash / position.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        notional = round(quantity * price, 4)
        commission = round(notional * self.commission_bps / 10_000.0, 4)

        if simulation_time is None:
            try:
                from mcp_quant_agent.clock import t_now

                simulation_time = t_now().isoformat()
            except RuntimeError:
                simulation_time = dt.datetime.utcnow().isoformat()

        if side == "buy":
            total_cost = notional + commission
            if total_cost > self.cash:
                raise ValueError(
                    f"Insufficient cash: need ${total_cost:,.2f} "
                    f"(${notional:,.2f} notional + ${commission:,.2f} commission at "
                    f"{self.commission_bps:.0f} bps), have ${self.cash:,.2f}. "
                    "Reduce quantity or check portfolio state."
                )
            self.cash -= notional + commission
            prev_qty = self.positions.get(ticker, 0.0)
            prev_cost = self.cost_basis.get(ticker, 0.0)
            new_qty = prev_qty + quantity
            # Weighted-average cost basis (excludes commission — that hits cash directly)
            self.cost_basis[ticker] = (
                (prev_cost * prev_qty + price * quantity) / new_qty
                if new_qty > 0
                else price
            )
            self.positions[ticker] = new_qty

        else:  # sell
            held = self.positions.get(ticker, 0.0)
            if quantity > held:
                raise ValueError(
                    f"Cannot sell {quantity:.2f} shares of {ticker}: "
                    f"only holding {held:.2f}."
                )
            self.cash += notional - commission  # commission deducted from proceeds
            new_qty = held - quantity
            if new_qty < 1e-9:  # treat as fully closed
                self.positions.pop(ticker, None)
                self.cost_basis.pop(ticker, None)
            else:
                self.positions[ticker] = new_qty

        self.total_commission = round(self.total_commission + commission, 4)
        self.total_turnover = round(self.total_turnover + notional, 4)
        fill = Fill(
            timestamp=simulation_time,
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=price,
            notional=notional,
            commission=commission,
        )
        self.order_history.append(fill)
        return fill

    def record_nav(
        self,
        current_prices: dict[str, float],
        timestamp: str | None = None,
    ) -> float:
        """Compute and record the current NAV.

        Call this once per bar in the backtest loop (after any order,
        or simply at bar close) to build the equity curve.

        Parameters
        ----------
        current_prices:
            Mapping of ticker → current close price (at t_now).
        timestamp:
            Simulation timestamp.  Defaults to current t_now.

        Returns
        -------
        float
            Current NAV.
        """
        if timestamp is None:
            try:
                from mcp_quant_agent.clock import t_now

                timestamp = t_now().isoformat()
            except RuntimeError:
                timestamp = dt.datetime.utcnow().isoformat()

        market_value = sum(
            self.positions.get(ticker, 0.0) * price
            for ticker, price in current_prices.items()
        )
        nav = round(self.cash + market_value, 4)
        self.nav_history.append({"timestamp": timestamp, "nav": nav})
        return nav

    # ── queries ───────────────────────────────────────────────────────────────

    def get_portfolio_summary(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """Return a portfolio snapshot with market values and PnL.

        Parameters
        ----------
        current_prices:
            Mapping of ticker → current close price.

        Returns
        -------
        dict[str, Any]
            ``{cash, positions, total_market_value, total_unrealised_pnl,
            nav, total_return_pct, num_trades}``.
        """
        positions_detail: list[dict[str, Any]] = []
        total_mv = 0.0
        total_upnl = 0.0

        for ticker, qty in sorted(self.positions.items()):
            price = current_prices.get(ticker)
            if price is None:
                continue
            mv = round(qty * price, 4)
            cost = self.cost_basis.get(ticker, 0.0)
            upnl = round((price - cost) * qty, 4)
            total_mv += mv
            total_upnl += upnl
            positions_detail.append(
                {
                    "ticker": ticker,
                    "quantity": qty,
                    "avg_cost": round(cost, 4),
                    "current_price": price,
                    "market_value": mv,
                    "unrealised_pnl": upnl,
                }
            )

        nav = round(self.cash + total_mv, 4)
        return {
            "cash": round(self.cash, 4),
            "positions": positions_detail,
            "total_market_value": round(total_mv, 4),
            "total_unrealised_pnl": round(total_upnl, 4),
            "nav": nav,
            "total_return_pct": round((nav / self.initial_cash - 1.0) * 100.0, 4),
            "num_trades": len(self.order_history),
            "total_commission": round(self.total_commission, 4),
            "total_turnover": round(self.total_turnover, 4),
            "commission_bps": self.commission_bps,
        }

    def get_order_history(self) -> list[dict[str, Any]]:
        """Return the full trade history as a list of dicts."""
        return [
            {
                "timestamp": f.timestamp,
                "ticker": f.ticker,
                "side": f.side,
                "quantity": f.quantity,
                "price": f.price,
                "notional": f.notional,
                "commission": f.commission,
            }
            for f in self.order_history
        ]

    def get_nav_series(self) -> list[dict[str, Any]]:
        """Return the recorded NAV time series (equity curve)."""
        return list(self.nav_history)
