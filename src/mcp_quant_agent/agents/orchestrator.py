"""Single-agent orchestrator — perceive → reason/act → observe LangGraph graph.

Architecture
------------
Minimal LangGraph ``StateGraph`` with 3 nodes:

1. **perceive** — fetches price bars, news, indicators, regime, and portfolio
   state by calling the MCP data/analytics functions directly.  All data is
   filtered through the simulation clock (``t_now`` enforced by the data layer).

2. **reason_and_act** — passes the perceived state to the LLM backbone (OpenAI
   or stub), parses the JSON decision, executes the order via ``Portfolio``.

3. **observe** — logs the complete decision trace to Langfuse, returns state.

LLM backbones
-------------
- ``StubBackbone``: deterministic trend-following rule (SMA20).  Used for
  smoke-testing the plumbing without an API key.
  **NEVER use stub results for thesis claims** — the decision rule is not
  the agent; it is a plumbing test fixture.
- ``OpenAIBackbone``: calls OpenAI via Langfuse drop-in (traces every call).
  Responses are disk-cached by prompt hash in dev (``settings.llm_cache``).

Fail-loud design
----------------
If ``OPENAI_API_KEY`` is absent and ``use_stub=False``, ``build_graph`` raises
``RuntimeError`` immediately with a clear message.  No silent fallback.

See ``docs/DECISIONS.md``: stub-backbone and orchestration entries.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mcp_quant_agent.mcp_servers.execution.paper_trading import Portfolio

# ---------------------------------------------------------------------------
# Agent state (TypedDict — required by LangGraph StateGraph)
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """State object passed between LangGraph nodes.

    Each node receives the full state and returns a *partial* dict of the
    fields it modifies.  LangGraph merges the partial update into the state.
    """

    ticker: str
    t_now_str: str
    bars: list[dict[str, Any]]
    news: list[dict[str, Any]]
    indicators: dict[str, Any]
    portfolio: dict[str, Any]
    reasoning: str
    decision: dict[str, Any]
    fill: dict[str, Any] | None
    regime: str | None
    errors: list[str]


# ---------------------------------------------------------------------------
# LLM response cache (disk-based, keyed by MD5 of model + prompt)
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("runs/.llm_cache")


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()


def _load_cache(key: str) -> str | None:
    f = _CACHE_DIR / f"{key}.json"
    if f.exists():
        with contextlib.suppress(Exception):
            return str(json.loads(f.read_text(encoding="utf-8"))["response"])
    return None


def _save_cache(key: str, response: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (_CACHE_DIR / f"{key}.json").write_text(
        json.dumps({"response": response}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# LLM backbones
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quantitative trading agent managing a paper portfolio.
Today's simulation date is given in the user message as T_NOW.  You can only
act on data available at or before T_NOW.  Do not speculate about future events.

Your task: analyse the provided market data for the given ticker and make ONE
trading decision.  Work through this chain-of-thought:

1. TREND: What is the price trend over the last 20 bars? Compare close to SMA20.
2. MOMENTUM: What do RSI-14 and MACD suggest about momentum direction?
3. VOLATILITY: Are Bollinger Bands wide / ATR elevated (high-vol regime)?
4. REGIME: What is the current market regime label? Is it consistent with your analysis?
5. NEWS SENTIMENT: Summarise the 3 most relevant news items and their likely direction.
6. PORTFOLIO: What is current exposure? Is position size within the 20%-NAV limit?
7. DECISION: State action, exact integer quantity, and 2-3 sentence rationale.

Output ONLY a JSON object -- no prose, no markdown code fences, no extra keys:
{"action": "buy"|"sell"|"hold", "quantity": <non-negative integer>, "rationale": "<2-3 sentences>"}

Rules:
- "hold" -> quantity MUST be 0.
- "buy" / "sell" -> quantity MUST be > 0.
- Single position MUST NOT exceed 20% of portfolio NAV.
- Insufficient data -> {"action": "hold", "quantity": 0, "rationale": "insufficient data"}.
"""


def _parse_decision(raw: str) -> dict[str, Any]:
    """Parse the LLM response into a decision dict, with fallback."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
    try:
        d: dict[str, Any] = json.loads(raw)
        action = str(d.get("action", "hold")).lower()
        if action not in ("buy", "sell", "hold"):
            action = "hold"
        quantity = max(0, int(d.get("quantity", 0)))
        if action == "hold":
            quantity = 0
        return {
            "action": action,
            "quantity": quantity,
            "rationale": str(d.get("rationale", "")),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Failed to parse LLM decision: %s | raw=%r", exc, raw[:200])
        return {"action": "hold", "quantity": 0, "rationale": f"parse_error: {exc}"}


class StubBackbone:
    """Deterministic rule-based stub for smoke-testing without an API key.

    Strategy: buy when close > SMA20 * 1.02, sell when close < SMA20 * 0.98.

    WARNING: This is a plumbing test fixture.  NEVER cite results from runs
    using this backbone in the thesis.  Always label as ``backbone=stub``.
    """

    def decide(self, state: AgentState) -> dict[str, Any]:
        bars = state["bars"]
        ticker = state["ticker"]
        if len(bars) < 5:
            return {"action": "hold", "quantity": 0, "rationale": "stub: insufficient data"}

        closes = [float(b["close"]) for b in bars[-20:]]
        sma = sum(closes) / len(closes)
        last_close = closes[-1]
        portfolio = state.get("portfolio", {})
        nav = float(portfolio.get("nav", 100_000.0))
        max_qty = max(1, int(nav * 0.10 / last_close)) if last_close > 0 else 1

        if last_close > sma * 1.02:
            return {
                "action": "buy",
                "quantity": max_qty,
                "rationale": f"stub: close {last_close:.2f} > SMA20 {sma:.2f} * 1.02",
            }
        if last_close < sma * 0.98:
            # Only sell if we actually hold a position -- no short selling.
            positions = portfolio.get("positions", [])
            held = 0.0
            for pos in positions:
                if isinstance(pos, dict) and pos.get("ticker") == ticker:
                    held = float(pos.get("quantity", 0.0))
                    break
            if held <= 0:
                return {
                    "action": "hold",
                    "quantity": 0,
                    "rationale": f"stub: close {last_close:.2f} < SMA20 {sma:.2f} * 0.98 but no position to sell",
                }
            sell_qty = min(max_qty, int(held))
            return {
                "action": "sell",
                "quantity": sell_qty,
                "rationale": f"stub: close {last_close:.2f} < SMA20 {sma:.2f} * 0.98",
            }
        return {
            "action": "hold",
            "quantity": 0,
            "rationale": f"stub: close {last_close:.2f} within SMA20 +/-2%",
        }


class OpenAIBackbone:
    """OpenAI-backed reasoning with Langfuse tracing and disk-cache.

    Parameters
    ----------
    model:
        OpenAI model ID (e.g. ``"gpt-4.1-mini"``).
    use_cache:
        If True (default), responses are cached on disk by prompt hash.
        Set to False for final thesis runs to ensure fresh responses.
    """

    def __init__(self, model: str, use_cache: bool = True) -> None:
        self.model = model
        self.use_cache = use_cache

    def decide(self, state: AgentState) -> dict[str, Any]:
        prompt = self._build_prompt(state)
        key = _cache_key(self.model, prompt)

        if self.use_cache:
            cached = _load_cache(key)
            if cached is not None:
                logger.debug("LLM cache HIT for %s/%s", state["ticker"], state["t_now_str"])
                return _parse_decision(cached)

        try:
            from langfuse.openai import openai  # type: ignore[attr-defined]
        except ImportError:
            import openai

        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            # Force JSON output so _parse_decision never falls back to hold.
            # Requires the system prompt to mention "JSON" (it does).
            response_format={"type": "json_object"},
        )
        content = str(response.choices[0].message.content or "")

        if self.use_cache:
            _save_cache(key, content)
            logger.debug("LLM cache MISS — saved for %s/%s", state["ticker"], state["t_now_str"])

        return _parse_decision(content)

    @staticmethod
    def _build_prompt(state: AgentState) -> str:
        """Build the user-turn prompt for the LLM.

        All data presented here is filtered to <= T_NOW by the data layer.
        The agent must make a decision based ONLY on this data.
        """
        bars = state["bars"][-30:]  # last 30 bars (for context)
        news = state["news"][:5]
        indicators = state.get("indicators", {})
        portfolio = state.get("portfolio", {})
        regime = state.get("regime") or "unknown"

        bar_rows = [
            f"  {b['date']} O={float(b['open']):.2f} H={float(b['high']):.2f} "
            f"L={float(b['low']):.2f} C={float(b['close']):.2f} V={int(b.get('volume', 0)):,}"
            for b in bars
        ]
        bar_text = "\n".join(bar_rows) if bar_rows else "  (no bars available)"

        news_rows = [
            f"  [{n.get('datetime', n.get('date', ''))}] {n.get('headline', n.get('title', ''))}"
            for n in news
        ]
        news_text = "\n".join(news_rows) if news_rows else "  (no recent news -- Finnhub key may not be set)"

        # Clean up indicators for display (round floats, drop None)
        ind_clean = {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in indicators.items() if v is not None}

        return (
            f"T_NOW (simulation date, do not use data after this): {state['t_now_str']}\n"
            f"Ticker: {state['ticker']}\n"
            f"Market regime (v2 detector): {regime}\n\n"
            f"Price bars, oldest to newest (last {len(bars)} bars up to T_NOW):\n"
            f"{bar_text}\n\n"
            f"Technical indicators (computed at T_NOW):\n"
            f"{json.dumps(ind_clean, indent=2)}\n\n"
            f"Recent news (up to T_NOW):\n{news_text}\n\n"
            f"Portfolio state:\n{json.dumps(portfolio, indent=2)}\n\n"
            f"Provide your chain-of-thought then output the JSON decision.\n"
        )


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    model: str = "gpt-4.1-mini",
    use_stub: bool = False,
    portfolio: Portfolio | None = None,
    use_cache: bool = True,
) -> Any:
    """Build and return a compiled LangGraph StateGraph.

    Parameters
    ----------
    model:
        OpenAI model ID (ignored when ``use_stub=True``).
    use_stub:
        If True, use ``StubBackbone`` instead of OpenAI.
    portfolio:
        ``Portfolio`` instance shared across all nodes.  If None, a fresh
        portfolio is created (useful for unit tests).
    use_cache:
        Forward to ``OpenAIBackbone`` — caches responses by prompt hash.

    Raises
    ------
    RuntimeError
        If ``use_stub=False`` and ``OPENAI_API_KEY`` is not configured.

    Returns
    -------
    Compiled LangGraph ``CompiledStateGraph``.
    """
    from langgraph.graph import END, START, StateGraph

    # ── Validate API key (fail loud) ─────────────────────────────────────────
    if not use_stub:
        import os

        from mcp_quant_agent.config import settings

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set in .env / environment.\n"
                "Options:\n"
                "  1. Set OPENAI_API_KEY=sk-... in .env and rerun.\n"
                "  2. Pass use_stub=True to use the deterministic stub backbone.\n"
                "WARNING: stub results are plumbing tests ONLY — never cite in thesis."
            )
        # pydantic-settings reads .env into Python objects but does NOT inject
        # into os.environ.  The OpenAI SDK (and langfuse.openai drop-in) reads
        # OPENAI_API_KEY from the process environment, so we forward it here.
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    # ── Inject Langfuse env vars (same reason: pydantic-settings doesn't set os.environ)
    # Must be called BEFORE importing langfuse.openai so the drop-in client can find
    # LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in the process environment.
    from mcp_quant_agent.observability.langfuse_setup import _is_langfuse_configured

    _is_langfuse_configured()

    # ── Build backbone ────────────────────────────────────────────────────────
    if portfolio is None:
        from mcp_quant_agent.mcp_servers.execution.paper_trading import Portfolio as _P

        portfolio = _P()

    _portfolio = portfolio  # capture for closures

    backbone: StubBackbone | OpenAIBackbone
    backbone = StubBackbone() if use_stub else OpenAIBackbone(model=model, use_cache=use_cache)

    # ── Node definitions ──────────────────────────────────────────────────────

    def perceive_node(state: AgentState) -> dict[str, Any]:
        """Fetch price bars, news, indicators, regime, and portfolio state."""
        import datetime as dt

        from mcp_quant_agent.clock import get_clock
        from mcp_quant_agent.mcp_servers.analytics.indicators import compute_indicators
        from mcp_quant_agent.mcp_servers.analytics.regime import get_current_regime
        from mcp_quant_agent.mcp_servers.data.yfinance_source import get_price_history

        ticker = state["ticker"]
        clock = get_clock()
        t_now_dt = clock.t_now
        end_str = t_now_dt.date().isoformat()
        start_str = (t_now_dt - dt.timedelta(days=120)).date().isoformat()

        errors: list[str] = []
        bars: list[dict[str, Any]] = []
        news: list[dict[str, Any]] = []
        indicators: dict[str, Any] = {}
        regime: str | None = None

        try:
            bars = get_price_history(ticker, start_str, end_str, use_cache=True)
        except Exception as exc:
            errors.append(f"bars: {exc}")
            logger.warning("perceive: bars fetch failed for %s: %s", ticker, exc)

        try:
            from mcp_quant_agent.mcp_servers.data.finnhub_source import get_news_items

            news_start = (t_now_dt - dt.timedelta(days=30)).date().isoformat()
            news = get_news_items(ticker, news_start, end_str)
        except Exception as exc:
            errors.append(f"news: {exc}")
            logger.debug("perceive: news fetch skipped for %s: %s", ticker, exc)

        if len(bars) >= 26:
            try:
                ind_result: dict[str, Any] = compute_indicators(bars)
                if ind_result:
                    indicators = {
                        k: v
                        for k, v in ind_result.items()
                        if k != "date" and v is not None
                    }
            except Exception as exc:
                errors.append(f"indicators: {exc}")

        try:
            regime = get_current_regime(bars)
        except Exception as exc:
            errors.append(f"regime: {exc}")

        current_prices: dict[str, float] = {}
        if bars:
            with contextlib.suppress(KeyError, TypeError, ValueError):
                current_prices[ticker] = float(bars[-1]["close"])
        portfolio_summary = _portfolio.get_portfolio_summary(current_prices)

        return {
            "t_now_str": t_now_dt.isoformat(),
            "bars": bars,
            "news": news,
            "indicators": indicators,
            "regime": regime,
            "portfolio": portfolio_summary,
            "errors": errors,
        }

    def reason_and_act_node(state: AgentState) -> dict[str, Any]:
        """Call LLM backbone, parse decision, execute order, record latency."""
        ticker = state["ticker"]
        bars = state["bars"]
        errors = list(state.get("errors", []))

        t_start = time.perf_counter()
        decision = backbone.decide(state)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        action = decision.get("action", "hold").lower()
        quantity = int(decision.get("quantity", 0))
        rationale = str(decision.get("rationale", ""))

        fill: dict[str, Any] | None = None
        if action in ("buy", "sell") and quantity > 0 and bars:
            price = float(bars[-1]["close"])
            try:
                fill_obj = _portfolio.place_order(ticker, action, quantity, price)
                fill = {
                    "ticker": ticker,
                    "side": action,
                    "quantity": quantity,
                    "price": price,
                    "timestamp": fill_obj.timestamp,
                    "notional": fill_obj.notional,
                }
            except ValueError as exc:
                errors.append(f"order_rejected: {exc}")
                logger.warning("Order rejected for %s: %s", ticker, exc)

        # Stash latency inside the decision dict so observe_node can read it.
        # Uses a leading-underscore key to distinguish from semantic fields.
        decision["_latency_ms"] = round(latency_ms, 1)

        return {
            "reasoning": rationale,
            "decision": decision,
            "fill": fill,
            "errors": errors,
        }

    def observe_node(state: AgentState) -> dict[str, Any]:
        """Log the full decision trace to Langfuse (best-effort, non-blocking).

        Captures everything needed for faithfulness and grounding evaluation:
        - inputs: recent bars, news, indicators, regime, portfolio
        - tool_outputs: actual data returned by each MCP function
        - reasoning: the model's CoT
        - decision: parsed action + quantity + rationale
        - fill: confirmed execution
        - latency: perceive-to-act duration
        """
        try:
            from mcp_quant_agent.observability.langfuse_setup import log_decision

            decision = state.get("decision", {})
            latency_ms = float(decision.get("_latency_ms", 0.0))

            # Capture the last 5 bars and first 3 news for grounding evaluation.
            # Truncating avoids huge payloads while preserving the key evidence.
            bars_recent = state["bars"][-5:] if state["bars"] else []
            news_recent = state["news"][:3] if state["news"] else []

            inputs = {
                "bars_count": len(state["bars"]),
                "bars_recent": bars_recent,
                "news_count": len(state["news"]),
                "news_recent": news_recent,
                "indicators": {
                    k: v for k, v in state.get("indicators", {}).items()
                    if not k.startswith("_")
                },
                "regime": state.get("regime"),
                "portfolio": state.get("portfolio", {}),
                "errors": state.get("errors", []),
            }

            tool_outputs: list[dict[str, Any]] = [
                {
                    "tool": "get_price_history",
                    "bars_returned": len(state["bars"]),
                    "date_range": (
                        f"{state['bars'][0]['date']} -- {state['bars'][-1]['date']}"
                        if state["bars"] else "no data"
                    ),
                },
                {
                    "tool": "get_news_items",
                    "items_returned": len(state["news"]),
                },
                {
                    "tool": "compute_indicators",
                    "keys": list(state.get("indicators", {}).keys()),
                },
                {
                    "tool": "get_current_regime",
                    "regime": state.get("regime"),
                },
            ]

            log_decision(
                trace_id=f"{state['ticker']}-{state['t_now_str']}",
                t_now=state["t_now_str"],
                ticker=state["ticker"],
                inputs=inputs,
                tool_outputs=tool_outputs,
                reasoning=state.get("reasoning", ""),
                decision=decision,
                fill=state.get("fill"),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.debug("Langfuse logging skipped: %s", exc)

        return {}

    # ── Assemble graph ────────────────────────────────────────────────────────
    graph_builder: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]
    graph_builder.add_node("perceive", perceive_node)
    graph_builder.add_node("reason_and_act", reason_and_act_node)
    graph_builder.add_node("observe", observe_node)
    graph_builder.add_edge(START, "perceive")
    graph_builder.add_edge("perceive", "reason_and_act")
    graph_builder.add_edge("reason_and_act", "observe")
    graph_builder.add_edge("observe", END)

    return graph_builder.compile()


def run_single_step(
    ticker: str,
    state: AgentState | None = None,
    graph: Any = None,
) -> dict[str, Any]:
    """Run one perceive → reason/act → observe cycle.

    Parameters
    ----------
    ticker:
        Equity ticker for this cycle.
    state:
        Initial state dict.  If None, a minimal state is constructed.
    graph:
        Pre-compiled LangGraph.  If None, ``build_graph(use_stub=True)``
        is called (useful for quick tests).

    Returns
    -------
    Final state dict after the full cycle.
    """
    if graph is None:
        graph = build_graph(use_stub=True)

    initial: AgentState = state or {
        "ticker": ticker,
        "t_now_str": "",
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
    return graph.invoke(initial)  # type: ignore[no-any-return]
