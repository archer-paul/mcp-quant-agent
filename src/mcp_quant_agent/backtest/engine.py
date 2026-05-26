"""Event-driven backtest engine — drip-feed agent loop.

The engine advances the simulation clock bar by bar, calls the orchestrator,
routes orders to the in-memory portfolio, and records the equity curve.

Anti-look-ahead guarantee
--------------------------
The clock is advanced to the bar date *before* the orchestrator is invoked.
The orchestrator's data calls (``get_price_history``, ``get_news_items``) all
go through the clock filter, so the agent can only observe data ≤ the current
bar date.  This is the structural guarantee.

Output artefacts
----------------
Every real (non-stub) run writes two files:

- ``runs/<run_id>/decisions.jsonl`` — one JSON line per decision, complete schema
  (action, quantity, rationale, tool_outputs, regime, indicators, fill, latency).
  This is the raw data needed to compute reasoning metrics without re-paying the API.
- ``results/<run_id>/summary.csv`` — human-readable per-ticker performance table.

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

import asyncio
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, cast

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
        run_id: str | None = None,
        concurrency: int = 15,
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.model = model
        self.use_stub = use_stub
        self.use_llm_cache = use_llm_cache
        self.concurrency = concurrency
        # Run ID: used for output file paths.  Auto-generated if not provided.
        if run_id is None:
            backbone_label = "stub" if use_stub else model.replace(".", "-")
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{backbone_label}_{ts}"
        self.run_id = run_id

    def run(self) -> dict[str, Any]:
        """Run the full backtest and return performance metrics + equity curve.

        Internally calls ``asyncio.run(_run_async_body())`` so that LLM calls
        within each bar-date are dispatched concurrently (bounded by
        ``self.concurrency``).  The public interface is unchanged — callers
        see a synchronous method returning a dict.

        Determinism guarantee
        ---------------------
        All tickers within a date receive an identical **start-of-date portfolio
        snapshot** (computed before any orders are placed).  Orders are then
        executed in a fixed ticker order (``self.tickers`` order) after all
        decisions arrive.  The final ``decisions.jsonl`` is sorted by
        ``(date, ticker)`` so it is byte-for-byte reproducible across runs.

        Returns
        -------
        dict with keys:
            ``nav_series``, ``metrics``, ``sharpe_ci``, ``n_bars``,
            ``n_decisions``, ``tickers``, ``start_date``, ``end_date``,
            ``backbone``, ``decisions_log`` (last 20 entries).
        """
        return asyncio.run(self._run_async_body())

    async def _run_async_body(self) -> dict[str, Any]:
        """Async implementation of the backtest loop."""
        from mcp_quant_agent.agents.orchestrator import (
            OpenAIBackbone,
            StubBackbone,
            perceive_ticker,
        )
        from mcp_quant_agent.clock import SimulationClock, set_clock
        from mcp_quant_agent.eval.financial import (
            bootstrap_sharpe_ci,
            compute_all_metrics,
        )
        from mcp_quant_agent.mcp_servers.data.yfinance_source import _fetch_raw_bars
        from mcp_quant_agent.mcp_servers.execution.paper_trading import Portfolio
        from mcp_quant_agent.observability.langfuse_setup import (
            _is_langfuse_configured,
            log_decision,
        )

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

        # ── 4. Build backbone ─────────────────────────────────────────────────
        # We use the backbone directly (not through LangGraph) so that the
        # perceive and decide steps can be decoupled and parallelised.
        backbone: StubBackbone | OpenAIBackbone
        if self.use_stub:
            backbone = StubBackbone()
        else:
            import os

            from mcp_quant_agent.config import settings as _settings

            # pydantic-settings doesn't inject into os.environ; forward manually
            # so AsyncOpenAI and langfuse.openai can find OPENAI_API_KEY.
            if not os.environ.get("OPENAI_API_KEY") and _settings.openai_api_key:
                os.environ["OPENAI_API_KEY"] = _settings.openai_api_key
            _is_langfuse_configured()
            backbone = OpenAIBackbone(model=self.model, use_cache=self.use_llm_cache)

        semaphore = asyncio.Semaphore(self.concurrency)

        # ── 5. Collect trading dates ──────────────────────────────────────────
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

        # ── 5b. Set up output paths ───────────────────────────────────────────
        runs_dir = Path("runs") / self.run_id
        results_dir = Path("results") / self.run_id
        if not self.use_stub:
            runs_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

        # ── 6. Async bar-by-bar loop ──────────────────────────────────────────
        nav_series: list[float] = []
        all_entries: list[dict[str, Any]] = []  # collected across all dates; sorted at end
        n_decisions = 0

        logger.info(
            "Backtest: %d bars × %d tickers (backbone=%s, concurrency=%d).",
            len(all_date_strs),
            len(self.tickers),
            "stub" if self.use_stub else self.model,
            self.concurrency,
        )

        for date_str in all_date_strs:
            date = dt.date.fromisoformat(date_str)
            clock.advance_to(date)

            # Tickers with a bar on this date (in deterministic order).
            active_tickers = [
                t for t in self.tickers
                if t in price_lookup and date_str in price_lookup[t]
            ]
            if not active_tickers:
                current_prices_empty: dict[str, float] = {}
                nav_series.append(portfolio.record_nav(current_prices_empty, timestamp=date_str))
                continue

            # Start-of-date portfolio snapshot (read once, shared by all tickers).
            # All tickers on this date see the same portfolio context, regardless
            # of async completion order — this is the determinism guarantee.
            current_prices_snap = {
                t: float(price_lookup[t][date_str]["close"]) for t in active_tickers
            }
            portfolio_snap = portfolio.get_portfolio_summary(current_prices_snap)

            # ── Concurrent phase: perceive + LLM decide ───────────────────────
            async def _one_ticker(
                tkr: str,
                snap: dict[str, Any] = portfolio_snap,
            ) -> tuple[str, dict[str, Any], dict[str, Any]]:
                """Perceive (thread) then decide (async), return (ticker, perceived, decision)."""
                perceived = await asyncio.to_thread(perceive_ticker, tkr, snap)
                # Inject ticker so backbone._build_prompt can access state["ticker"].
                perceived["ticker"] = tkr
                # perceive_ticker returns dict[str, Any]; cast to AgentState so
                # decide_async type-checks correctly (fields are structurally compatible).
                from mcp_quant_agent.agents.orchestrator import AgentState as _AS

                t_start = asyncio.get_event_loop().time()
                decision = await backbone.decide_async(cast(_AS, perceived), semaphore)
                decision["_latency_ms"] = round(
                    (asyncio.get_event_loop().time() - t_start) * 1000.0, 1
                )
                return tkr, perceived, decision

            ticker_results = await asyncio.gather(
                *[_one_ticker(t) for t in active_tickers],
                return_exceptions=False,
            )

            # ── Serial phase: execute orders in deterministic ticker order ─────
            # ticker_results arrives in the same order as active_tickers (gather
            # preserves order), but sort explicitly for clarity + safety.
            ticker_results_sorted = sorted(ticker_results, key=lambda x: x[0])

            for tkr, perceived, decision in ticker_results_sorted:
                action = decision.get("action", "hold").lower()
                quantity = int(decision.get("quantity", 0))
                rationale = str(decision.get("rationale", ""))
                latency_ms = float(decision.get("_latency_ms", 0.0))
                errors = list(perceived.get("errors", []))

                bars_for_ticker = perceived.get("bars", [])
                fill: dict[str, Any] | None = None
                if action in ("buy", "sell") and quantity > 0 and bars_for_ticker:
                    price = float(bars_for_ticker[-1]["close"])
                    try:
                        fill_obj = portfolio.place_order(tkr, action, quantity, price)
                        fill = {
                            "ticker": tkr,
                            "side": action,
                            "quantity": quantity,
                            "price": price,
                            "timestamp": fill_obj.timestamp,
                            "notional": fill_obj.notional,
                        }
                    except ValueError as exc:
                        errors.append(f"order_rejected: {exc}")
                        logger.warning("Order rejected for %s: %s", tkr, exc)

                indicators = perceived.get("indicators", {})
                nav = float(portfolio_snap.get("nav", 0.0)) or 1.0
                positions_with_pct = [
                    {
                        "ticker": p.get("ticker"),
                        "quantity": p.get("quantity"),
                        "market_value": round(float(p.get("market_value", 0.0)), 2),
                        "pct_of_nav": round(float(p.get("market_value", 0.0)) / nav, 4),
                    }
                    for p in portfolio_snap.get("positions", [])
                ]
                news = perceived.get("news", [])

                tool_outputs: list[dict[str, Any]] = [
                    {
                        "tool": "get_price_history",
                        "bars_count": len(bars_for_ticker),
                        "bars_recent": [
                            {
                                "date": b["date"],
                                "open": round(float(b.get("open", 0)), 4),
                                "high": round(float(b.get("high", 0)), 4),
                                "low": round(float(b.get("low", 0)), 4),
                                "close": round(float(b.get("close", 0)), 4),
                                "volume": int(b.get("volume", 0)),
                            }
                            for b in bars_for_ticker[-5:]
                        ],
                    },
                    {
                        "tool": "get_news_items",
                        "items_count": len(news),
                        "items_recent": [
                            {
                                "date": n.get("datetime", n.get("date", "")),
                                "headline": str(n.get("headline", n.get("title", "")))[:120],
                            }
                            for n in news[:3]
                        ],
                    },
                    {
                        "tool": "compute_indicators",
                        "values": {
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in indicators.items()
                            if v is not None and not str(k).startswith("_")
                        },
                    },
                    {
                        "tool": "get_current_regime",
                        "regime": perceived.get("regime"),
                    },
                    {
                        "tool": "get_portfolio",
                        "cash": round(float(portfolio_snap.get("cash", 0.0)), 2),
                        "nav": round(float(portfolio_snap.get("nav", 0.0)), 2),
                        "positions": positions_with_pct,
                    },
                ]

                entry: dict[str, Any] = {
                    "date": date_str,
                    "ticker": tkr,
                    "action": action,
                    "quantity": quantity,
                    "rationale": rationale,
                    "fill": fill,
                    "regime": perceived.get("regime"),
                    "indicators": {
                        k: v for k, v in indicators.items()
                        if not str(k).startswith("_")
                    },
                    "tool_outputs": tool_outputs,
                    "latency_ms": latency_ms,
                    "errors": errors,
                }
                all_entries.append(entry)
                n_decisions += 1

                # Best-effort Langfuse trace
                if not self.use_stub:
                    try:
                        log_decision(
                            trace_id=f"{tkr}-{date_str}",
                            t_now=date_str,
                            ticker=tkr,
                            inputs={
                                "bars_count": len(bars_for_ticker),
                                "indicators": {
                                    k: v for k, v in indicators.items()
                                    if not str(k).startswith("_")
                                },
                                "regime": perceived.get("regime"),
                                "portfolio": portfolio_snap,
                                "errors": errors,
                            },
                            tool_outputs=tool_outputs,
                            reasoning=rationale,
                            decision=decision,
                            fill=fill,
                            latency_ms=latency_ms,
                        )
                    except Exception as exc:
                        logger.debug("Langfuse logging skipped: %s", exc)

            # Record NAV after all orders for this date
            nav_val = portfolio.record_nav(current_prices_snap, timestamp=date_str)
            nav_series.append(nav_val)

            if len(nav_series) % 50 == 0:
                logger.info(
                    "  bar %d/%d  NAV=%.0f",
                    len(nav_series), len(all_date_strs), nav_val,
                )

        # ── 7. Write decisions.jsonl sorted by (date, ticker) ─────────────────
        # Sorting makes the file deterministic and diffable across runs.
        all_entries.sort(key=lambda e: (e["date"], e["ticker"]))
        decisions_log = all_entries  # full list alias

        if not self.use_stub:
            jsonl_path = runs_dir / "decisions.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as jsonl_fh:  # noqa: SIM115
                for entry in all_entries:
                    jsonl_fh.write(json.dumps(entry, default=str) + "\n")
            logger.info("Decisions written to %s (%d entries)", jsonl_path, len(all_entries))

        # ── 8. Compute metrics ────────────────────────────────────────────────
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

        # ── 9. Write summary.csv ──────────────────────────────────────────────
        if not self.use_stub:
            self._write_summary_csv(
                results_dir=results_dir,
                metrics=metrics,
                sharpe_ci=sharpe_ci,
                decisions_log=decisions_log,
                backbone_label=backbone_label,
                nav_series=nav_series,
            )

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
            "run_id": self.run_id,
            "decisions_log": decisions_log[-20:],  # last 20 for display
            "decisions_all": decisions_log,  # full list for reasoning eval
            "order_history": portfolio.get_order_history(),
        }

    def _write_summary_csv(
        self,
        results_dir: Path,
        metrics: dict[str, Any],
        sharpe_ci: dict[str, Any],
        decisions_log: list[dict[str, Any]],
        backbone_label: str,
        nav_series: list[float],
    ) -> None:
        """Write a human-readable summary CSV to results/<run_id>/summary.csv."""
        import csv

        # Per-ticker metrics from decisions_log
        regime_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {"buy": 0, "sell": 0, "hold": 0, "error": 0}
        for d in decisions_log:
            regime = str(d.get("regime") or "unknown")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            action = str(d.get("action", "hold"))
            action_counts[action] = action_counts.get(action, 0) + 1

        rows: list[list[Any]] = [
            ["run_id", self.run_id],
            ["backbone", backbone_label],
            ["tickers", ", ".join(self.tickers)],
            ["start_date", self.start_date],
            ["end_date", self.end_date],
            ["n_bars", len(nav_series)],
            ["n_decisions", len(decisions_log)],
            ["final_nav", round(nav_series[-1], 2) if nav_series else 0.0],
            ["initial_cash", self.initial_cash],
            ["annualised_return", round(float(metrics.get("annualised_return", 0.0)), 4)],
            ["sharpe", round(float(metrics.get("sharpe", 0.0)), 4)],
            ["sharpe_ci_low", round(float(sharpe_ci.get("ci_lower", 0.0)), 4)],
            ["sharpe_ci_high", round(float(sharpe_ci.get("ci_upper", 0.0)), 4)],
            ["sortino", round(float(metrics.get("sortino", 0.0)), 4)],
            ["calmar", round(float(metrics.get("calmar", 0.0)), 4)],
            ["max_drawdown", round(float(metrics.get("max_drawdown", 0.0)), 4)],
            ["hit_rate", round(float(metrics.get("hit_rate", 0.0)), 4)],
            ["actions_buy", action_counts.get("buy", 0)],
            ["actions_sell", action_counts.get("sell", 0)],
            ["actions_hold", action_counts.get("hold", 0)],
            ["actions_error", action_counts.get("error", 0)],
        ]
        # Regime distribution
        for regime, count in sorted(regime_counts.items()):
            rows.append([f"regime_{regime}_days", count])

        csv_path = results_dir / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(rows)
        logger.info("Summary written to %s", csv_path)
