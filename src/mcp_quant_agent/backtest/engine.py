"""Event-driven backtest engine — drip-feed agent loop.

The engine advances the simulation clock bar by bar, calling the orchestrator at
each step.  It is the single place where:
1. The simulation clock is advanced (``clock.advance_to(bar_date)``).
2. The orchestrator is invoked (``run_single_step``).
3. NAV is recorded (``portfolio.record_nav``).
4. Metrics are computed at the end of the run.

Anti-look-ahead guarantee
--------------------------
The clock is advanced to the bar date *before* the orchestrator call.  The
orchestrator's MCP tool calls will only see data ≤ the current bar date.
This is the structural guarantee: it is impossible for the orchestrator to
observe bars beyond its "current" date.

Status: STUB — implementation in J5 sprint.

Design notes
------------
- Event-driven over vectorised because the agent loop must be sequential (each
  decision depends on the previous fill/portfolio state).
- vectorbt is used only for baseline computation; this engine is pure Python.
- Each iteration logs a structured trace to Langfuse (see orchestrator.py).
"""

from __future__ import annotations

from typing import Any


class BacktestEngine:
    """Event-driven backtest engine.

    Parameters
    ----------
    tickers:
        List of equity tickers to trade.
    start_date, end_date:
        ISO-8601 backtest window.
    initial_cash:
        Starting portfolio cash.
    model:
        OpenAI model for agent decisions.

    Status: STUB.
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100_000.0,
        model: str = "gpt-4.1-mini",
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.model = model

    def run(self) -> dict[str, Any]:
        """Run the full backtest and return performance metrics.

        Implementation order (per CLAUDE.md):
        1. Set simulation clock to start_date.
        2. Pre-fetch and cache all price data for the full period.
        3. For each bar date in [start_date, end_date]:
           a. advance_to(bar_date)
           b. run_single_step(ticker) for each ticker
           c. record_nav(current_prices)
        4. Compute metrics via eval/financial.py.
        5. Label regimes via eval/regime.py.
        6. Return structured results.

        Raises
        ------
        NotImplementedError
            Until the implementation sprint (J5).
        """
        raise NotImplementedError(
            "BacktestEngine.run() not yet implemented — see docs/PLAN.md J5."
        )
