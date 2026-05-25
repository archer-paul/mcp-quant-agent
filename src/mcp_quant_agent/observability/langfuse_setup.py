"""Langfuse Cloud tracing wiring.

Policy (from CLAUDE.md)
-----------------------
- **Cloud only** — do NOT self-host.  YC credits cover $100/mo.
- **SDK v3+ API** — CLAUDE.md labels this "v4+" but the current PyPI package
  is langfuse ≥ 3.0.  The import paths match the v3 API.
  See docs/DECISIONS.md for the version clarification.
- **Do NOT** use ``from langfuse.decorators import ...`` — that module was
  removed in the v3+ SDK.

What to trace (per CLAUDE.md)
------------------------------
Every agent decision is logged as a structured Langfuse trace containing:
- Inputs seen (bars, news, indicators, portfolio state)
- Tool calls + their actual outputs (for grounding evaluation)
- The model's chain-of-thought
- The resulting order (for faithfulness evaluation)
- Latency, token count, cost

Scores attached to traces (for regime-segmented dashboards)
------------------------------------------------------------
After evaluation (J9 sprint), faithfulness / grounding / sophistication scores
are attached to the relevant traces via ``langfuse.score()``.  This gives
regime-segmented eval dashboards "for free" in the Langfuse UI.

Usage
-----
For LangGraph (pass the callback handler in the graph config)::

    from mcp_quant_agent.observability.langfuse_setup import get_langfuse_handler
    handler = get_langfuse_handler()
    graph.invoke(state, config={"callbacks": [handler]})

For drop-in OpenAI tracing (replaces ``openai`` import)::

    from langfuse.openai import openai  # NOT the standard openai import
    client = openai.OpenAI(api_key=settings.openai_api_key)
    # All calls to this client are automatically traced

For manual span creation::

    from langfuse import get_client, observe
    lf = get_client()
    with lf.start_as_current_span("decision") as span:
        span.update(input={"ticker": "AAPL", "t_now": "2022-06-15"})
        ...
        span.update(output=decision)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_langfuse_configured() -> bool:
    """Return True if Langfuse keys are set in settings."""
    from mcp_quant_agent.config import settings

    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def get_langfuse_client() -> Any:
    """Return an initialised Langfuse client.

    Uses settings from ``mcp_quant_agent.config``.  Returns ``None`` if
    Langfuse is not configured (missing keys) — callers must handle this
    gracefully so the agent runs even without tracing.

    Returns
    -------
    langfuse.Langfuse | None
        Initialised client, or None if keys are not configured.
    """
    if not _is_langfuse_configured():
        logger.warning(
            "Langfuse keys not configured — tracing disabled. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env to enable."
        )
        return None

    try:
        from langfuse import Langfuse

        from mcp_quant_agent.config import settings

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except ImportError as exc:
        logger.warning("langfuse package not installed: %s", exc)
        return None


def get_langfuse_handler() -> Any:
    """Return a Langfuse callback handler for LangGraph.

    Pass this to the LangGraph graph's ``config`` dict::

        graph.invoke(state, config={"callbacks": [get_langfuse_handler()]})

    This traces all LangGraph node calls, LLM calls, and tool calls
    automatically without manual instrumentation of every node.

    Returns ``None`` if Langfuse is not configured.
    """
    if not _is_langfuse_configured():
        return None

    try:
        from langfuse.callback import CallbackHandler

        from mcp_quant_agent.config import settings

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except ImportError as exc:
        logger.warning("Langfuse callback handler not available: %s", exc)
        return None


def score_trace(
    trace_id: str,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """Attach a numeric score to an existing Langfuse trace.

    Use this to attach reasoning-eval metrics (faithfulness, grounding,
    sophistication) after they are computed, so they appear in the Langfuse
    UI and enable regime-segmented dashboards.

    Parameters
    ----------
    trace_id:
        The Langfuse trace ID (from the callback handler or a manual trace).
    name:
        Score name (e.g. ``"faithfulness"``, ``"grounding"``, ``"sophistication"``).
    value:
        Numeric score value.
    comment:
        Optional human-readable note (e.g. regime label, decision summary).
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to score trace %s: %s", trace_id, exc)


def log_decision(
    trace_id: str,
    t_now: str,
    ticker: str,
    inputs: dict[str, Any],
    tool_outputs: list[dict[str, Any]],
    reasoning: str,
    decision: dict[str, Any],
    fill: dict[str, Any] | None,
    latency_ms: float,
    token_usage: dict[str, int] | None = None,
) -> None:
    """Log a full structured agent decision to Langfuse.

    This is called by the orchestrator's ``observe`` node after each decision.
    The trace schema captures everything needed for faithfulness and grounding
    evaluation:
    - Inputs (bars, news, indicators, portfolio) — the "perceived state"
    - Tool outputs — the actual data returned (for grounding)
    - Reasoning — the model's CoT
    - Decision + fill — for faithfulness evaluation
    - Latency + tokens — for the cost/latency axis

    Parameters
    ----------
    trace_id:
        Trace ID from the LangGraph callback handler.
    t_now, ticker:
        Decision context.
    inputs:
        Perceived state dict (bars, news, indicators, portfolio).
    tool_outputs:
        List of actual tool-call outputs (for grounding).
    reasoning:
        Raw CoT string from the LLM.
    decision:
        Parsed decision dict ``{action, ticker, quantity, rationale}``.
    fill:
        Fill confirmation from the execution server (None if "hold").
    latency_ms:
        Total perceive→act latency in milliseconds.
    token_usage:
        ``{prompt_tokens, completion_tokens, total_tokens}``.
    """
    client = get_langfuse_client()
    if client is None:
        return

    try:
        trace = client.trace(id=trace_id)
        trace.update(
            metadata={
                "t_now": t_now,
                "ticker": ticker,
                "action": decision.get("action"),
                "latency_ms": latency_ms,
                "token_usage": token_usage or {},
            },
            input=inputs,
            output={
                "reasoning": reasoning,
                "decision": decision,
                "fill": fill,
                "tool_outputs": tool_outputs,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log decision to Langfuse: %s", exc)


def flush() -> None:
    """Flush the Langfuse client to ensure all events are sent.

    Call at the end of a backtest run to prevent data loss.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to flush Langfuse: %s", exc)
