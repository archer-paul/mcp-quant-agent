"""Event-driven backtest engine — drip-feed agent loop.

The engine advances the simulation clock bar by bar, calls the orchestrator,
routes orders to the in-memory portfolio, and records the equity curve.

Anti-look-ahead guarantee
--------------------------
The clock is advanced to the bar date *before* the orchestrator is invoked.
The orchestrator's data calls (``get_price_history``, ``get_news_items``) all
go through the clock filter, so the agent can only observe data ≤ the current
bar date.  This is the structural guarantee.

Usage (stub backbone — no API key needed)::

    from mcp_quant_agent.backtest.engine import BacktestEngine
    engine = BacktestEngine(
        tickers=["AAPL", "MSFT"],
        start_date="2022-01-01",
        end_date="2022-01-31",
        use_stub=True,
    )
    results = engine.run()
    print(results["metrics"])

Usage (real backbone)::

    engine = BacktestEngine(
        tickers=["AAPL"], start_date="2022-01-01", end_date="2022-03-31",
        model="gpt-4.1-mini", use_stub=False,
    )
    # Raises RuntimeError if OPENAI_API_KEY is not set.
    results = engine.run()
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

logger = logging.getLogger(__name__)


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
        OpenAI model ID for agent decisions (ignored when ``use_stub=True``).
    use_stub:
        If True, use the deterministic stub backbone (no API key required).
        Stub results must NEVER be cited in the thesis.
    use_llm_cache:
        Cache LLM responses to disk (by prompt hash).  Set to False for final
        thesis runs to ensure fresh responses.

    Raises
    ------
    RuntimeError
        At ``run()`` time if ``use_stub=False`` and ``OPENAI_API_KEY`` is unset.
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100_000.0,
        model: str = "gpt-4.1-mini",
        use_stub: bool = False,
        use_llm_cache: bool = True,
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.model = model
        self.use_stub = use_stub
        self.use_llm_cache = use_llm_cache

    def run(self) -> dict[str, Any]:
        """Run the full backtest and return performance metrics + equity curve.

        Steps
        -----
        1. Validate API key (fail loud if needed).
        2. Set simulation clock to ``start_date``.
        3. Pre-fetch and cache all price data for the full period.
        4. Build the LangGraph orchestrator.
        5. For each trading date in [start_date, end_date]:
           a. ``clock.advance_to(date)``
           b. Run one ``graph.invoke`` per ticker.
           c. Record NAV via ``portfolio.record_nav(current_prices)``.
        6. Compute performance metrics.
        7. Return results dict.

        Returns
        -------
        dict with keys:
            ``nav_series``, ``metrics``, ``sharpe_ci``, ``n_bars``,
            ``n_decisions``, ``tickers``, ``start_date``, ``end_date``,
            ``backbone``, ``decisions_log`` (last 20 entries).
        """
        from mcp_quant_agent.agents.orchestrator import AgentState, build_graph
        from mcp_quant_agent.clock import SimulationClock, set_clock
        from mcp_quant_agent.eval.financial import (
            bootstrap_sharpe_ci,
            compute_all_metrics,
        )
        from mcp_quant_agent.mcp_servers.data.yfinance_source import _fetch_raw_bars
        from mcp_quant_agent.mcp_servers.execution.paper_trading import Portfolio

        # ── 1. Validate API key ───────────────────────────────────────────────
        if not self.use_stub:
            from mcp_quant_agent.config import settings

            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set in .env / environment.\n"
                    "Set it or run with use_stub=True for a smoke test.\n"
                    "WARNING: stub results are for plumbing tests ONLY."
                )

        # ── 2. Set up clock and portfolio ─────────────────────────────────────
        clock = SimulationClock(self.start_date)
        set_clock(clock)
        portfolio = Portfolio(cash=self.initial_cash, initial_cash=self.initial_cash)

        # ── 3. Pre-fetch price data ───────────────────────────────────────────
        logger.info(
            "Pre-fetching price data for %s (%s → %s)...",
            self.tickers,
            self.start_date,
            self.end_date,
        )
        price_data: dict[str, list[dict[str, Any]]] = {}
        for ticker in self.tickers:
            try:
                bars = _fetch_raw_bars(ticker, self.start_date, self.end_date, "1d")
                price_data[ticker] = bars
                logger.info("  %s: %d bars fetched.", ticker, len(bars))
            except Exception as exc:
                logger.error("  Failed to fetch %s: %s", ticker, exc)
                price_data[ticker] = []

        # ── 4. Build orchestrator ─────────────────────────────────────────────
        graph = build_graph(
            model=self.model,
            use_stub=self.use_stub,
            portfolio=portfolio,
            use_cache=self.use_llm_cache,
        )

        # ── 5. Collect trading dates (union of all tickers' bar dates) ────────
        all_date_strs = sorted(
            {bar["date"] for bars in price_data.values() for bar in bars}
        )
        if not all_date_strs:
            logger.error("No price data available — cannot run backtest.")
            return {
                "nav_series": [],
                "metrics": {},
                "sharpe_ci": {},
                "n_bars": 0,
                "n_decisions": 0,
                "tickers": self.tickers,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "backbone": "stub" if self.use_stub else self.model,
                "decisions_log": [],
                "error": "No price data available.",
            }

        # Build a quick lookup: ticker → {date_str → bar}
        price_lookup: dict[str, dict[str, dict[str, Any]]] = {
            ticker: {bar["date"]: bar for bar in bars}
            for ticker, bars in price_data.items()
        }

        # ── 6. Bar-by-bar loop ────────────────────────────────────────────────
        nav_series: list[float] = []
        decisions_log: list[dict[str, Any]] = []
        n_decisions = 0

        logger.info(
            "Backtest: %d bars × %d tickers (backbone=%s).",
            len(all_date_strs),
            len(self.tickers),
            "stub" if self.use_stub else self.model,
        )

        for date_str in all_date_strs:
            date = dt.date.fromisoformat(date_str)
            clock.advance_to(date)

            # Agent step for each ticker
            for ticker in self.tickers:
                if ticker not in price_lookup or date_str not in price_lookup[ticker]:
                    continue  # no bar for this ticker on this date

                initial_state: AgentState = {
                    "ticker": ticker,
                    "t_now_str": date_str,
                    "bars": [],
                    "news": [],
                    "indicators": {},
                    "portfolio": {},
                    "reasoning": "",
                    "decision": {},
                    "fill": None,
                    "regime": None,
                    "errors": [],
                }
                try:
                    result = graph.invoke(initial_state)
                    n_decisions += 1
                    decision = result.get("decision", {})
                    decisions_log.append(
                        {
                            "date": date_str,
                            "ticker": ticker,
                            "action": decision.get("action", "hold"),
                            "quantity": decision.get("quantity", 0),
                            "fill": result.get("fill"),
                            "regime": result.get("regime"),
                            "errors": result.get("errors", []),
                        }
                    )
                except Exception as exc:
                    logger.error("Agent error on %s/%s: %s", date_str, ticker, exc)
                    decisions_log.append(
                        {
                            "date": date_str,
                            "ticker": ticker,
                            "action": "error",
                            "quantity": 0,
                            "fill": None,
                            "errors": [str(exc)],
                        }
                    )

            # Record NAV (after all tickers for this date)
            current_prices = {
                ticker: float(price_lookup[ticker][date_str]["close"])
                for ticker in self.tickers
                if ticker in price_lookup and date_str in price_lookup[ticker]
            }
            nav = portfolio.record_nav(current_prices, timestamp=date_str)
            nav_series.append(nav)

        # ── 7. Compute metrics ────────────────────────────────────────────────
        metrics = compute_all_metrics(nav_series) if len(nav_series) > 1 else {}
        sharpe_ci = bootstrap_sharpe_ci(nav_series) if len(nav_series) > 10 else {}

        backbone_label = "stub" if self.use_stub else self.model

        logger.info(
            "Backtest complete: %d bars, Sharpe=%.2f, FinalNAV=%.0f",
            len(nav_series),
            metrics.get("sharpe", 0.0),
            nav_series[-1] if nav_series else 0.0,
        )

        # Flush Langfuse to ensure all trace events are sent before process ends.
        if not self.use_stub:
            try:
                from mcp_quant_agent.observability.langfuse_setup import flush

                flush()
                logger.info("Langfuse traces flushed.")
            except Exception as exc:
                logger.debug("Langfuse flush skipped: %s", exc)

        return {
            "nav_series": nav_series,
            "metrics": metrics,
            "sharpe_ci": sharpe_ci,
            "n_bars": len(nav_series),
            "n_decisions": n_decisions,
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "backbone": backbone_label,
            "decisions_log": decisions_log[-20:],  # last 20 for display
            "order_history": portfolio.get_order_history(),
        }
