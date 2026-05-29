"""Backtest engine for the global Portfolio Manager multi-agent path."""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PMBacktestEngine:
    """Daily global-PM backtest engine.

    The normal path is deterministic/stubbed and validates the multi-agent
    plumbing, target-weight schema, rebalancer, anti-lookahead data flow, and
    decisions.jsonl shape.  The real PM API path is only available when a
    validated smoke guard is injected by ``scripts/run_pm_api_smoke.py``.
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100_000.0,
        run_id: str | None = None,
        use_stub: bool = True,
        price_data: dict[str, list[dict[str, Any]]] | None = None,
        news_data: dict[str, list[dict[str, Any]]] | None = None,
        news_offline: bool = True,
        allow_empty_news: bool = False,
        min_trade_notional: float = 100.0,
        write_artifacts: bool = True,
        output_root: Path | str = Path("."),
        model: str | None = None,
        use_llm_cache: bool = True,
        smoke_guard: Any | None = None,
        pm_backbone: Any | None = None,
        transaction_cost_bps: float | None = None,
    ) -> None:
        self.tickers = [ticker.upper() for ticker in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.use_stub = use_stub
        self.price_data = price_data
        self.news_data = news_data
        self.news_offline = news_offline
        self.allow_empty_news = allow_empty_news
        self.min_trade_notional = min_trade_notional
        self.write_artifacts = write_artifacts
        self.output_root = Path(output_root)
        self.model = model
        self.use_llm_cache = use_llm_cache
        self.smoke_guard = smoke_guard
        self.pm_backbone = pm_backbone
        self._price_sources: dict[str, str] = {}
        if transaction_cost_bps is None:
            from mcp_quant_agent.config import settings
            transaction_cost_bps = settings.transaction_cost_bps
        self.transaction_cost_bps = transaction_cost_bps

        if run_id is None:
            label = "pm_stub" if use_stub else "pm_api_smoke"
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{label}_{ts}"
        self.run_id = run_id

    def run(self) -> dict[str, Any]:
        """Run the PM backtest and return metrics, decisions, and orders."""
        from mcp_quant_agent.agents.multi_agent import (
            parse_pm_stub_output,
            run_pm_stub,
        )
        from mcp_quant_agent.agents.pm_backbone import (
            OpenAIDiscussionBackbone,
            PMDiscussionResult,
        )
        from mcp_quant_agent.agents.pm_memory import PMDecisionLog
        from mcp_quant_agent.agents.pm_schemas import (
            DailyPMDecision,
            PMTargetWeights,
        )
        from mcp_quant_agent.backtest.rebalancer import rebalance_to_targets
        from mcp_quant_agent.clock import SimulationClock, set_clock
        from mcp_quant_agent.config import settings
        from mcp_quant_agent.eval.financial import (
            bootstrap_sharpe_ci,
            compute_all_metrics,
        )
        from mcp_quant_agent.mcp_servers.analytics.indicators import compute_indicators
        from mcp_quant_agent.mcp_servers.analytics.regime import get_current_regime
        from mcp_quant_agent.mcp_servers.execution.paper_trading import Portfolio

        model = self.model or settings.agent_model_dev
        if not self.use_stub:
            self._validate_api_smoke_authorized(model)

        clock = SimulationClock(self.start_date)
        set_clock(clock)
        portfolio = Portfolio(
            cash=self.initial_cash,
            initial_cash=self.initial_cash,
            commission_bps=self.transaction_cost_bps,
        )
        price_data = self._load_price_data()
        pm_backbone = None
        if not self.use_stub:
            pm_backbone = self.pm_backbone or OpenAIDiscussionBackbone(
                model=model,
                use_cache=self.use_llm_cache,
            )

        # Decision memory log — causal cross-date context injection.
        # Persisted to disk only when write_artifacts or API mode is active.
        _persist_log = self.write_artifacts or not self.use_stub
        _log_path = (
            self.output_root / "runs" / self.run_id / "pm_decision_log.md"
            if _persist_log
            else None
        )
        decision_log = PMDecisionLog(log_path=_log_path)
        _prev_prices: dict[str, float] = {}
        _prev_decision_date: str | None = None

        all_date_strs = sorted(
            {
                str(bar["date"])
                for bars in price_data.values()
                for bar in bars
                if self.start_date <= str(bar["date"]) <= self.end_date
            }
        )
        if not all_date_strs:
            return {
                "nav_series": [],
                "metrics": {},
                "sharpe_ci": {},
                "n_bars": 0,
                "n_decisions": 0,
                "tickers": self.tickers,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "backbone": "pm_stub" if self.use_stub else model,
                "run_id": self.run_id,
                "decisions_log": [],
                "decisions_all": [],
                "order_history": [],
                "error": "No price data available.",
            }

        nav_series: list[float] = []
        all_entries: list[dict[str, Any]] = []

        for date_str in all_date_strs:
            clock.advance_to(dt.date.fromisoformat(date_str))
            bars_by_ticker = {
                ticker: [
                    bar for bar in price_data.get(ticker, [])
                    if str(bar["date"]) <= date_str
                ]
                for ticker in self.tickers
            }
            active_tickers = [
                ticker for ticker in self.tickers
                if bars_by_ticker.get(ticker)
            ]
            if not active_tickers:
                nav_series.append(portfolio.record_nav({}, timestamp=date_str))
                continue

            current_prices = {
                ticker: float(bars_by_ticker[ticker][-1]["close"])
                for ticker in active_tickers
            }

            # Update previous decision with one-period realized returns (causal:
            # we now know the move from the previous close to the current close).
            if _prev_decision_date is not None and _prev_prices:
                realized_returns = {
                    ticker: (current_prices[ticker] - _prev_prices[ticker])
                    / _prev_prices[ticker]
                    for ticker in active_tickers
                    if ticker in _prev_prices and _prev_prices[ticker] > 0.0
                }
                decision_log.update_with_returns(
                    date=_prev_decision_date,
                    realized_returns=realized_returns,
                )

            past_context = decision_log.get_past_context(
                t_now=date_str,
                n_recent=5,
            )

            portfolio_before = portfolio.get_portfolio_summary(current_prices)
            market_data: dict[str, dict[str, Any]] = {}
            tool_outputs: list[dict[str, Any]] = []
            mcp_calls: list[dict[str, Any]] = []
            errors: list[str] = []
            indicators_flat: dict[str, Any] = {}
            regimes: dict[str, str | None] = {}

            for ticker in active_tickers:
                bars_full = bars_by_ticker[ticker]
                bars_recent = bars_full[-120:]
                indicators: dict[str, Any] = {}
                regime: str | None = None

                if len(bars_full) >= 26:
                    try:
                        indicators = compute_indicators(bars_full)
                    except Exception as exc:
                        errors.append(f"{ticker} indicators: {exc}")

                try:
                    regime = get_current_regime(bars_full)
                except Exception as exc:
                    errors.append(f"{ticker} regime: {exc}")

                try:
                    news = self._news_for_ticker(ticker, date_str)
                except RuntimeError as exc:
                    if not self.allow_empty_news:
                        raise
                    errors.append(f"{ticker} news: {exc}")
                    news = []

                market_data[ticker] = {
                    "bars": bars_recent,
                    "news": news,
                    "indicators": indicators,
                    "regime": regime,
                }
                regimes[ticker] = regime
                for key, value in indicators.items():
                    if value is not None:
                        indicators_flat[f"{ticker}.{key}"] = value
                tool_outputs.extend(
                    self._tool_outputs_for_ticker(
                        ticker=ticker,
                        bars=bars_full,
                        news=news,
                        indicators=indicators,
                        regime=regime,
                    )
                )
                mcp_calls.extend(
                    self._mcp_calls_for_ticker(
                        ticker=ticker,
                        bars=bars_full,
                        news=news,
                        indicators=indicators,
                        regime=regime,
                    )
                )

            nav_before = float(portfolio_before.get("nav", 0.0))
            positions_with_pct = self._positions_with_pct(portfolio_before)
            tool_outputs.append(
                {
                    "tool": "get_portfolio",
                    "cash": round(float(portfolio_before.get("cash", 0.0)), 2),
                    "nav": round(nav_before, 2),
                    "positions": positions_with_pct,
                }
            )
            mcp_calls.append(
                {
                    "tool": "get_portfolio",
                    "source": "paper_portfolio",
                    "row_count": len(portfolio_before.get("positions", [])),
                    "max_timestamp": date_str,
                    "phase": "before_rebalance",
                }
            )

            t_start = time.perf_counter()
            _pm_is_error = False
            discussion: list[dict[str, Any]] = []
            if self.use_stub:
                state = run_pm_stub(
                    date=date_str,
                    tickers=active_tickers,
                    market_data=market_data,
                    portfolio=portfolio_before,
                )
                reports, targets = parse_pm_stub_output(state)
            else:
                reports = []
                try:
                    if pm_backbone is None:
                        raise RuntimeError("PM API backbone was not initialised")
                    outcome = pm_backbone.decide(
                        date=date_str,
                        tickers=active_tickers,
                        market_data=market_data,
                        tool_outputs=tool_outputs,
                        portfolio=portfolio_before,
                        current_prices=current_prices,
                        regimes=regimes,
                        mcp_calls=mcp_calls,
                        past_context=past_context,
                    )
                    if isinstance(outcome, PMDiscussionResult):
                        reports = outcome.reports
                        discussion = outcome.discussion
                        targets = outcome.targets
                    elif isinstance(outcome, PMTargetWeights):
                        targets = outcome
                    else:
                        raise TypeError(
                            "PM backbone must return PMDiscussionResult or PMTargetWeights"
                        )
                except Exception as exc:  # noqa: BLE001
                    raw_snippet = getattr(exc, "_raw_completion", "")[:300]
                    error = (
                        f"pm_api_error: {type(exc).__name__}: {exc}"
                        + (f" [raw={raw_snippet!r}]" if raw_snippet else "")
                    )
                    errors.append(error)
                    logger.error(
                        "PM API ERROR on %s: %s", date_str, error
                    )
                    targets = PMTargetWeights(
                        date=date_str,
                        weights={},
                        cash_weight=1.0,
                        rationale=f"{error}; explicit all-cash fallback",
                    )
                    _pm_is_error = True
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

            # Store decision in the memory log (pending until J+1 returns known).
            decision_log.store_decision(
                date=date_str,
                weights=dict(targets.weights),
                cash_weight=float(targets.cash_weight),
                rationale=str(targets.rationale),
                regimes={ticker: regimes.get(ticker) for ticker in active_tickers},
            )
            _prev_prices = dict(current_prices)
            _prev_decision_date = date_str

            rebalance = rebalance_to_targets(
                date=date_str,
                portfolio_snapshot=portfolio_before,
                current_prices=current_prices,
                target_weights=targets,
                min_trade_notional=self.min_trade_notional,
            )

            fills: list[dict[str, Any]] = []
            for order in rebalance.orders:
                if order.side not in ("buy", "sell") or order.quantity <= 0:
                    continue
                try:
                    fill_obj = portfolio.place_order(
                        order.ticker,
                        order.side,
                        order.quantity,
                        order.price,
                        simulation_time=date_str,
                    )
                    fills.append(
                        {
                            "ticker": fill_obj.ticker,
                            "side": fill_obj.side,
                            "quantity": fill_obj.quantity,
                            "price": fill_obj.price,
                            "timestamp": fill_obj.timestamp,
                            "notional": fill_obj.notional,
                        }
                    )
                except ValueError as exc:
                    errors.append(f"{order.ticker} order_rejected: {exc}")

            portfolio_after = portfolio.get_portfolio_summary(current_prices)
            nav_after = portfolio.record_nav(current_prices, timestamp=date_str)

            decision = DailyPMDecision(
                date=date_str,
                tickers=active_tickers,
                reports=reports,
                discussion=discussion,
                targets=rebalance.targets,
                rationale=rebalance.targets.rationale,
                orders=rebalance.orders,
                fills=fills,
                tool_outputs=tool_outputs,
                mcp_calls=mcp_calls,
                indicators=indicators_flat,
                regimes=regimes,
                portfolio_before=portfolio_before,
                portfolio_after=portfolio_after,
                nav_before=nav_before,
                nav_after=nav_after,
                latency_ms=latency_ms,
                errors=errors,
                is_error=_pm_is_error,
            )
            entry = decision.model_dump()
            all_entries.append(entry)
            nav_series.append(nav_after)

            if not self.use_stub:
                self._log_pm_langfuse(
                    date_str=date_str,
                    model=model,
                    decision=entry,
                    reports=[report.model_dump() for report in reports],
                    discussion=discussion,
                    portfolio=portfolio_before,
                    tool_outputs=tool_outputs,
                    mcp_calls=mcp_calls,
                    orders=[order.model_dump() for order in rebalance.orders],
                    fills=fills,
                    past_context=past_context,
                    cache_hit=bool(getattr(pm_backbone, "last_cache_hit", False)),
                    cache_events=list(getattr(pm_backbone, "cache_events", [])),
                )

        all_entries.sort(key=lambda entry: (entry["date"], entry["ticker"]))

        # Fail-loud error rate: any pm_api_error is a data-quality issue.
        n_errors = sum(1 for e in all_entries if e.get("is_error", False))
        if n_errors > 0:
            logger.error(
                "PM run %s: %d/%d decisions are pm_api_error (excluded from eval). "
                "Do NOT cite eval metrics from this run until errors are investigated.",
                self.run_id, n_errors, len(all_entries),
            )
        else:
            logger.info("PM run %s: 0 pm_api_error decisions.", self.run_id)

        metrics = (
            compute_all_metrics(
                nav_series,
                total_commission=portfolio.total_commission,
                total_turnover=portfolio.total_turnover,
            )
            if len(nav_series) > 1 else {}
        )
        sharpe_ci = bootstrap_sharpe_ci(nav_series) if len(nav_series) > 10 else {}

        backbone_label = "pm_stub" if self.use_stub else model
        if self.write_artifacts or not self.use_stub:
            self._write_artifacts(
                all_entries,
                metrics,
                sharpe_ci,
                nav_series,
                backbone_label,
            )

        if not self.use_stub:
            try:
                from mcp_quant_agent.observability.langfuse_setup import flush

                flush()
            except Exception as exc:
                logger.debug("Langfuse flush skipped: %s", exc)

        return {
            "nav_series": nav_series,
            "metrics": metrics,
            "sharpe_ci": sharpe_ci,
            "n_bars": len(nav_series),
            "n_decisions": len(all_entries),
            "n_pm_errors": n_errors,
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "backbone": backbone_label,
            "run_id": self.run_id,
            "decisions_log": all_entries[-20:],
            "decisions_all": all_entries,
            "order_history": portfolio.get_order_history(),
        }

    def _validate_api_smoke_authorized(self, model: str) -> None:
        """Require the exact smoke guard result before PM API mode can run."""
        if self.smoke_guard is None:
            raise RuntimeError(
                "PM API mode requires a validated PMSmokeGuardResult or "
                "PMMultiDayGuardResult.  Use scripts/run_pm_api_smoke.py (1 date) "
                "or scripts/run_pm_multiday.py (multi-date bounded)."
            )

        from mcp_quant_agent.backtest.pm_smoke_guard import PMMultiDayGuardResult

        guard = self.smoke_guard
        guard_tickers = list(getattr(guard, "tickers", []))
        if sorted(guard_tickers) != sorted(self.tickers):
            raise RuntimeError(
                f"PM guard tickers {guard_tickers} do not match engine tickers {self.tickers}."
            )
        if getattr(guard, "start_date", None) != self.start_date or getattr(guard, "end_date", None) != self.end_date:
            raise RuntimeError("PM guard dates do not match engine dates.")
        # 1-date guard: enforce single date.  Multi-day guard: allow range.
        is_multiday = isinstance(guard, PMMultiDayGuardResult)
        if not is_multiday and self.start_date != self.end_date:
            raise RuntimeError(
                "PMSmokeGuardResult (1-date guard) requires start_date==end_date. "
                "Use PMMultiDayGuardResult for multi-date runs."
            )
        if getattr(guard, "model", None) != model:
            raise RuntimeError("PM guard model does not match engine model.")
        if not bool(getattr(guard, "use_llm_cache", False)) or not self.use_llm_cache:
            raise RuntimeError("PM run requires LLM cache enabled.")
        if not is_multiday and int(getattr(guard, "estimated_pm_calls", 0)) != 1:
            raise RuntimeError("PMSmokeGuardResult must estimate exactly one PM call.")

    def _load_price_data(self) -> dict[str, list[dict[str, Any]]]:
        if self.price_data is not None:
            loaded_from_input: dict[str, list[dict[str, Any]]] = {}
            for ticker in self.tickers:
                rows = (
                    self.price_data.get(ticker)
                    or self.price_data.get(ticker.lower())
                    or self.price_data.get(ticker.upper())
                    or []
                )
                loaded_from_input[ticker] = sorted(
                    [dict(row) for row in rows],
                    key=lambda row: str(row["date"]),
                )
                self._price_sources[ticker] = "in_memory"
            return loaded_from_input

        from mcp_quant_agent.mcp_servers.analytics.regime import (
            REGIME_WARMUP_CALENDAR_DAYS,
        )
        from mcp_quant_agent.mcp_servers.data.bar_validation import validate_bar
        from mcp_quant_agent.mcp_servers.data.yfinance_source import (
            _check_price_continuity,
            _fetch_raw_bars,
            _get_cache,
        )
        warmup_start = (
            dt.date.fromisoformat(self.start_date)
            - dt.timedelta(days=REGIME_WARMUP_CALENDAR_DAYS)
        ).isoformat()
        loaded: dict[str, list[dict[str, Any]]] = {}
        cache = _get_cache()
        for ticker in self.tickers:
            cached = self._cached_price_window(cache, ticker, warmup_start)
            source = "cache"

            if self._price_cache_needs_refresh(cached):
                source = "api" if not cached else "cache+api"
                bars = _fetch_raw_bars(ticker, warmup_start, self.end_date, "1d")
                if bars:
                    cache.merge_and_write(ticker, "1d", bars)
                cached = self._cached_price_window(cache, ticker, warmup_start)

            bars = sorted(cached, key=lambda row: str(row["date"]))
            for bar in bars:
                validate_bar(bar, ticker)
            _check_price_continuity(bars, ticker)
            loaded[ticker] = sorted(bars, key=lambda row: str(row["date"]))
            self._price_sources[ticker] = source
        return loaded

    def _cached_price_window(
        self,
        cache: Any,
        ticker: str,
        warmup_start: str,
    ) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in cache.read_all(ticker, "1d")
            if warmup_start <= str(row.get("date", "")) <= self.end_date
        ]

    def _price_cache_needs_refresh(
        self,
        cached: list[dict[str, Any]],
    ) -> bool:
        if not cached:
            return True
        dates = sorted(str(row["date"]) for row in cached)
        has_backtest_data = any(self.start_date <= date <= self.end_date for date in dates)
        has_warmup = dates[0] <= self.start_date and len(dates) >= 26
        reaches_end = dates[-1] >= self.end_date
        return not (has_backtest_data and has_warmup and reaches_end)

    def _news_for_ticker(self, ticker: str, date_str: str) -> list[dict[str, Any]]:
        if self.news_data is not None:
            from mcp_quant_agent.clock import get_clock

            clock = get_clock()
            start = (dt.date.fromisoformat(date_str) - dt.timedelta(days=30)).isoformat()
            source_items = (
                self.news_data.get(ticker)
                or self.news_data.get(ticker.lower())
                or self.news_data.get(ticker.upper())
                or []
            )
            items = [
                dict(item) for item in source_items
                if start <= str(item.get("datetime", ""))[:10] <= date_str
            ]
            filtered = clock.filter_rows(items, date_key="datetime") if items else []
            return sorted(filtered, key=lambda item: str(item["datetime"]), reverse=True)

        from mcp_quant_agent.mcp_servers.data.finnhub_source import (
            get_news_items_cache_first,
        )

        start_date = (
            dt.date.fromisoformat(date_str) - dt.timedelta(days=30)
        ).isoformat()
        return get_news_items_cache_first(
            ticker,
            start_date,
            date_str,
            offline=self.news_offline,
        )

    @staticmethod
    def _positions_with_pct(portfolio_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        nav = float(portfolio_snapshot.get("nav", 0.0)) or 1.0
        return [
            {
                "ticker": pos.get("ticker"),
                "quantity": pos.get("quantity"),
                "market_value": round(float(pos.get("market_value", 0.0)), 2),
                "pct_of_nav": round(float(pos.get("market_value", 0.0)) / nav, 4),
            }
            for pos in portfolio_snapshot.get("positions", [])
            if isinstance(pos, dict)
        ]

    @staticmethod
    def _max_timestamp(rows: list[dict[str, Any]], *keys: str) -> str | None:
        values: list[str] = []
        for row in rows:
            for key in keys:
                raw = row.get(key)
                if raw:
                    values.append(str(raw))
                    break
        return max(values) if values else None

    def _mcp_calls_for_ticker(
        self,
        *,
        ticker: str,
        bars: list[dict[str, Any]],
        news: list[dict[str, Any]],
        indicators: dict[str, Any],
        regime: str | None,
    ) -> list[dict[str, Any]]:
        price_source = self._price_sources.get(
            ticker,
            "in_memory" if self.price_data is not None else "cache_or_api",
        )
        news_source = (
            "in_memory"
            if self.news_data is not None
            else "cache"
            if self.news_offline
            else "cache_or_api"
        )
        bars_ts = self._max_timestamp(bars, "date")
        news_ts = self._max_timestamp(news, "datetime", "date")
        return [
            {
                "ticker": ticker,
                "tool": "get_price_history",
                "source": price_source,
                "row_count": len(bars),
                "max_timestamp": bars_ts,
            },
            {
                "ticker": ticker,
                "tool": "get_news_items_cache_first",
                "source": news_source,
                "row_count": len(news),
                "max_timestamp": news_ts,
            },
            {
                "ticker": ticker,
                "tool": "compute_indicators",
                "source": "local",
                "row_count": 1 if indicators else 0,
                "max_timestamp": bars_ts if indicators else None,
            },
            {
                "ticker": ticker,
                "tool": "get_current_regime",
                "source": "local",
                "row_count": 1 if regime else 0,
                "max_timestamp": bars_ts if regime else None,
            },
        ]

    def _log_pm_langfuse(
        self,
        *,
        date_str: str,
        model: str,
        decision: dict[str, Any],
        reports: list[dict[str, Any]],
        discussion: list[dict[str, Any]],
        portfolio: dict[str, Any],
        tool_outputs: list[dict[str, Any]],
        mcp_calls: list[dict[str, Any]],
        orders: list[dict[str, Any]],
        fills: list[dict[str, Any]],
        past_context: str = "",
        cache_hit: bool = False,
        cache_events: list[dict[str, Any]] | None = None,
    ) -> None:
        try:
            from mcp_quant_agent.observability.langfuse_setup import log_pm_decision

            log_pm_decision(
                run_id=self.run_id,
                t_now=date_str,
                model=model,
                tickers=decision.get("tickers", []),
                reports=reports,
                discussion=discussion,
                portfolio=portfolio,
                tool_outputs=tool_outputs,
                mcp_calls=mcp_calls,
                rationale=str(decision.get("rationale", "")),
                target_weights=decision.get("targets", {}),
                orders=orders,
                fills=fills,
                past_context=past_context,
                metadata={"llm_cache_hit": cache_hit, "llm_cache_events": cache_events or []},
            )
        except Exception as exc:
            logger.debug("PM Langfuse logging skipped: %s", exc)

    @staticmethod
    def _tool_outputs_for_ticker(
        *,
        ticker: str,
        bars: list[dict[str, Any]],
        news: list[dict[str, Any]],
        indicators: dict[str, Any],
        regime: str | None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "ticker": ticker,
                "tool": "get_price_history",
                "bars_count": len(bars),
                "max_timestamp": PMBacktestEngine._max_timestamp(bars, "date"),
                "bars_recent": [
                    {
                        "date": bar["date"],
                        "open": round(float(bar.get("open", 0.0)), 4),
                        "high": round(float(bar.get("high", 0.0)), 4),
                        "low": round(float(bar.get("low", 0.0)), 4),
                        "close": round(float(bar.get("close", 0.0)), 4),
                        "volume": int(bar.get("volume", 0)),
                    }
                    for bar in bars[-5:]
                ],
            },
            {
                "ticker": ticker,
                "tool": "get_news_items_cache_first",
                "items_count": len(news),
                "max_timestamp": PMBacktestEngine._max_timestamp(news, "datetime", "date"),
                "items_recent": [
                    {
                        "datetime": item.get("datetime", ""),
                        "headline": str(item.get("headline", ""))[:120],
                    }
                    for item in news[:3]
                ],
            },
            {
                "ticker": ticker,
                "tool": "compute_indicators",
                "values": {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in indicators.items()
                    if value is not None
                },
            },
            {
                "ticker": ticker,
                "tool": "get_current_regime",
                "regime": regime,
            },
        ]

    def _write_artifacts(
        self,
        decisions: list[dict[str, Any]],
        metrics: dict[str, Any],
        sharpe_ci: dict[str, Any],
        nav_series: list[float],
        backbone_label: str,
    ) -> None:
        import csv

        runs_dir = self.output_root / "runs" / self.run_id
        results_dir = self.output_root / "results" / self.run_id
        runs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = runs_dir / "decisions.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:  # noqa: SIM115
            for entry in decisions:
                fh.write(json.dumps(entry, default=str) + "\n")

        rows: list[list[Any]] = [
            ["run_id", self.run_id],
            ["backbone", backbone_label],
            ["tickers", ", ".join(self.tickers)],
            ["start_date", self.start_date],
            ["end_date", self.end_date],
            ["n_bars", len(nav_series)],
            ["n_decisions", len(decisions)],
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
        ]
        csv_path = results_dir / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value"])
            writer.writerows(rows)

        logger.info("PM decisions written to %s", jsonl_path)
        logger.info("PM summary written to %s", csv_path)
