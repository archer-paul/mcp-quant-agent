"""Deterministic stub graph for the PM multi-agent path.

The graph is intentionally compact:
technical analyst -> news analyst -> risk analyst -> global PM.

It does not call OpenAI. It exists to test PM plumbing, state shape,
anti-lookahead data flow, target-weight validation, and logging before any
paid API smoke test is considered.
"""

from __future__ import annotations

from typing import Any, TypedDict

from mcp_quant_agent.agents.pm_schemas import AnalystReport, PMTargetWeights
from mcp_quant_agent.backtest.rebalancer import build_target_weights


class PMGraphState(TypedDict, total=False):
    """Compact state passed through the PM stub graph."""

    date: str
    tickers: list[str]
    market_data: dict[str, dict[str, Any]]
    portfolio: dict[str, Any]
    reports: list[dict[str, Any]]
    targets: dict[str, Any]
    errors: list[str]
    pm_received_tickers: list[str]


_SIGNAL_VALUE = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
_PM_ROLE_WEIGHT = {"technical": 0.55, "news": 0.25, "risk": 0.20}


def _latest_close(market: dict[str, Any]) -> float | None:
    indicators = market.get("indicators", {})
    if indicators.get("close") is not None:
        return float(indicators["close"])
    bars = market.get("bars", [])
    if bars:
        return float(bars[-1]["close"])
    return None


def _make_report(
    *,
    date: str,
    ticker: str,
    analyst: str,
    signal: str,
    confidence: float,
    summary: str,
    evidence: list[str],
) -> dict[str, Any]:
    report = AnalystReport(
        date=date,
        ticker=ticker,
        analyst=analyst,  # type: ignore[arg-type]
        signal=signal,  # type: ignore[arg-type]
        confidence=max(0.0, min(1.0, confidence)),
        summary=summary[:500],
        evidence=evidence[:5],
    )
    return report.model_dump()


def _technical_node(state: PMGraphState) -> dict[str, Any]:
    date = state["date"]
    reports = list(state.get("reports", []))
    for ticker in state["tickers"]:
        market = state["market_data"].get(ticker, {})
        indicators = market.get("indicators", {})
        bars = market.get("bars", [])
        close = _latest_close(market)
        sma20 = indicators.get("sma_20")
        rsi = indicators.get("rsi_14")

        if close is None or sma20 is None or len(bars) < 20:
            reports.append(
                _make_report(
                    date=date,
                    ticker=ticker,
                    analyst="technical",
                    signal="neutral",
                    confidence=0.25,
                    summary="insufficient technical history",
                    evidence=[f"bars={len(bars)}"],
                )
            )
            continue

        sma = float(sma20)
        ratio = close / sma if sma > 0 else 1.0
        if ratio > 1.02:
            signal = "bullish"
            confidence = min(0.9, 0.45 + (ratio - 1.0) * 6.0)
            summary = "close is above SMA20 with positive trend"
        elif ratio < 0.98:
            signal = "bearish"
            confidence = min(0.9, 0.45 + (1.0 - ratio) * 6.0)
            summary = "close is below SMA20 with negative trend"
        else:
            signal = "neutral"
            confidence = 0.4
            summary = "close is near SMA20"

        reports.append(
            _make_report(
                date=date,
                ticker=ticker,
                analyst="technical",
                signal=signal,
                confidence=confidence,
                summary=summary,
                evidence=[
                    f"close={close:.4f}",
                    f"sma20={sma:.4f}",
                    f"rsi14={rsi}" if rsi is not None else "rsi14=None",
                ],
            )
        )
    return {"reports": reports}


def _news_node(state: PMGraphState) -> dict[str, Any]:
    date = state["date"]
    reports = list(state.get("reports", []))
    positive = ("beat", "upgrade", "growth", "record", "raises", "strong")
    negative = ("miss", "downgrade", "lawsuit", "warning", "cuts", "weak")

    for ticker in state["tickers"]:
        news = state["market_data"].get(ticker, {}).get("news", [])
        joined = " ".join(
            str(item.get("headline") or item.get("title") or "").lower()
            for item in news[:5]
        )
        pos_hits = sum(word in joined for word in positive)
        neg_hits = sum(word in joined for word in negative)

        if not news or pos_hits == neg_hits:
            signal = "neutral"
            confidence = 0.3 if not news else 0.45
            summary = "no directional news signal"
        elif pos_hits > neg_hits:
            signal = "bullish"
            confidence = min(0.8, 0.45 + 0.1 * pos_hits)
            summary = "recent headlines skew positive"
        else:
            signal = "bearish"
            confidence = min(0.8, 0.45 + 0.1 * neg_hits)
            summary = "recent headlines skew negative"

        reports.append(
            _make_report(
                date=date,
                ticker=ticker,
                analyst="news",
                signal=signal,
                confidence=confidence,
                summary=summary,
                evidence=[
                    str(item.get("headline") or item.get("title") or "")[:120]
                    for item in news[:3]
                    if item.get("headline") or item.get("title")
                ],
            )
        )
    return {"reports": reports}


def _position_weight(portfolio: dict[str, Any], ticker: str) -> float:
    nav = float(portfolio.get("nav", 0.0)) or 1.0
    for pos in portfolio.get("positions", []):
        if isinstance(pos, dict) and str(pos.get("ticker", "")).upper() == ticker:
            return float(pos.get("market_value", 0.0)) / nav
    return 0.0


def _risk_node(state: PMGraphState) -> dict[str, Any]:
    date = state["date"]
    reports = list(state.get("reports", []))
    portfolio = state.get("portfolio", {})

    for ticker in state["tickers"]:
        market = state["market_data"].get(ticker, {})
        regime = market.get("regime") or "unknown"
        pos_weight = _position_weight(portfolio, ticker)

        if pos_weight > 0.70:
            signal = "bearish"
            confidence = 0.7
            summary = "position concentration is high"
        elif regime == "high_vol":
            signal = "bearish"
            confidence = 0.65
            summary = "high-volatility regime reduces risk budget"
        elif regime == "bear":
            signal = "bearish"
            confidence = 0.55
            summary = "bear regime reduces risk budget"
        else:
            signal = "neutral"
            confidence = 0.4
            summary = "risk regime does not force de-risking"

        reports.append(
            _make_report(
                date=date,
                ticker=ticker,
                analyst="risk",
                signal=signal,
                confidence=confidence,
                summary=summary,
                evidence=[f"regime={regime}", f"position_weight={pos_weight:.4f}"],
            )
        )
    return {"reports": reports}


def _pm_node(state: PMGraphState) -> dict[str, Any]:
    date = state["date"]
    tickers = [str(t).upper() for t in state["tickers"]]
    reports = [AnalystReport.model_validate(r) for r in state.get("reports", [])]
    reports_by_ticker: dict[str, list[AnalystReport]] = {ticker: [] for ticker in tickers}
    for report in reports:
        reports_by_ticker.setdefault(report.ticker, []).append(report)

    raw_scores: dict[str, float] = {}
    defensive = False
    for ticker in tickers:
        score = 0.20
        for report in reports_by_ticker.get(ticker, []):
            role_weight = _PM_ROLE_WEIGHT[report.analyst]
            score += role_weight * _SIGNAL_VALUE[report.signal] * report.confidence
            defensive = defensive or (
                report.analyst == "risk"
                and report.signal == "bearish"
                and report.confidence >= 0.6
            )
        raw_scores[ticker] = max(0.0, score)

    active_scores = {ticker: score for ticker, score in raw_scores.items() if score > 0.05}
    gross_target = 0.65 if defensive else 0.95
    if not active_scores:
        weights: dict[str, float] = {}
        cash_weight = 1.0
    else:
        total_score = sum(active_scores.values())
        weights = {
            ticker: gross_target * score / total_score
            for ticker, score in active_scores.items()
        }
        cash_weight = 1.0 - sum(weights.values())

    leaders = sorted(weights, key=lambda ticker: weights[ticker], reverse=True)[:3]
    targets = build_target_weights(
        date=date,
        weights=weights,
        cash_weight=max(0.0, cash_weight),
        rationale=(
            f"stub PM: gross={sum(weights.values()):.2f}; "
            f"leaders={','.join(leaders) if leaders else 'cash'}"
        ),
        normalize=True,
    )
    return {
        "targets": targets.model_dump(),
        "pm_received_tickers": tickers,
    }


def build_pm_stub_graph() -> Any:
    """Build the deterministic LangGraph PM stub graph."""
    from langgraph.graph import END, START, StateGraph

    graph_builder: StateGraph = StateGraph(PMGraphState)  # type: ignore[type-arg]
    graph_builder.add_node("technical_analyst", _technical_node)
    graph_builder.add_node("news_analyst", _news_node)
    graph_builder.add_node("risk_analyst", _risk_node)
    graph_builder.add_node("portfolio_manager", _pm_node)
    graph_builder.add_edge(START, "technical_analyst")
    graph_builder.add_edge("technical_analyst", "news_analyst")
    graph_builder.add_edge("news_analyst", "risk_analyst")
    graph_builder.add_edge("risk_analyst", "portfolio_manager")
    graph_builder.add_edge("portfolio_manager", END)
    return graph_builder.compile()


def run_pm_stub(
    *,
    date: str,
    tickers: list[str],
    market_data: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
) -> dict[str, Any]:
    """Run one deterministic PM stub decision and return the final graph state."""
    graph = build_pm_stub_graph()
    initial: PMGraphState = {
        "date": date,
        "tickers": [ticker.upper() for ticker in tickers],
        "market_data": market_data,
        "portfolio": portfolio,
        "reports": [],
        "errors": [],
    }
    return graph.invoke(initial)  # type: ignore[no-any-return]


def run_pm_analysts(
    *,
    date: str,
    tickers: list[str],
    market_data: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
) -> list[AnalystReport]:
    """Run deterministic analyst nodes only, without the stub PM allocator."""
    state: PMGraphState = {
        "date": date,
        "tickers": [ticker.upper() for ticker in tickers],
        "market_data": market_data,
        "portfolio": portfolio,
        "reports": [],
        "errors": [],
    }
    for node in (_technical_node, _news_node, _risk_node):
        update = node(state)
        state["reports"] = list(update.get("reports", state.get("reports", [])))
    return [AnalystReport.model_validate(report) for report in state.get("reports", [])]


def parse_pm_stub_output(state: dict[str, Any]) -> tuple[list[AnalystReport], PMTargetWeights]:
    """Parse final graph state into strict PM schemas."""
    reports = [AnalystReport.model_validate(report) for report in state.get("reports", [])]
    targets = PMTargetWeights.model_validate(state["targets"])
    return reports, targets
