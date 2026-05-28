"""Langfuse Cloud tracing wiring -- SDK v4.x.

SDK version note (v4)
---------------------
The installed ``langfuse`` package is v4.x (verified: 4.6.1).  The API has
changed from v3:

- **No** ``from langfuse.callback import CallbackHandler``  -- module was
  removed.  LangGraph tracing in v4 goes through OpenTelemetry.
- **No** ``trace(id=...)``  -- the old trace-by-ID pattern is gone.
- ``from langfuse.openai import openai``  -- drop-in, works in v4.  All
  OpenAI calls are automatically traced with full token/cost data.
- ``get_client().start_as_current_observation(...)``  -- context-manager for
  creating custom spans.  Used in ``log_decision``.
- ``get_client().score_current_trace(...)``  -- attaches a score to the
  trace that is current in the OTEL context.

CLAUDE.md says "v4+" -- this matches the installed version.

What is traced
--------------
- Every OpenAI call (auto-traced by ``langfuse.openai`` drop-in) with full
  model, tokens, cost, latency.
- Every agent decision (``log_decision``) as an "agent" observation:
  inputs (bars, news, indicators, regime, portfolio), reasoning (CoT),
  decision, fill, latency.
- Reasoning-eval scores attached via ``score_trace`` after evaluation.

Usage
-----
For drop-in OpenAI tracing (in orchestrator)::

    from langfuse.openai import openai  # NOT standard openai
    response = openai.chat.completions.create(...)
    # Automatically traced with all metadata.

For manual observation (in observe_node)::

    from mcp_quant_agent.observability.langfuse_setup import log_decision
    log_decision(trace_id=..., t_now=..., ...)

For attaching eval scores later::

    from mcp_quant_agent.observability.langfuse_setup import score_trace
    score_trace(trace_id=..., name="faithfulness", value=0.85)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_langfuse_configured() -> bool:
    """Return True if Langfuse keys are present in settings.

    As a side-effect, injects the Langfuse keys into os.environ so that the
    ``langfuse.openai`` drop-in (which creates its own client from env vars)
    can find them.  pydantic-settings reads .env into Python objects but does
    NOT inject into os.environ automatically.
    """
    import os

    from mcp_quant_agent.config import settings

    configured = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
    if configured:
        if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        if not os.environ.get("LANGFUSE_SECRET_KEY"):
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        if not os.environ.get("LANGFUSE_HOST"):
            os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    return configured


def get_langfuse_client() -> Any:
    """Return an initialised Langfuse client, or None if not configured.

    Returns
    -------
    langfuse.Langfuse | None
    """
    if not _is_langfuse_configured():
        logger.warning(
            "Langfuse keys not configured -- tracing disabled. "
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


def get_langfuse_handler() -> None:
    """No-op in Langfuse v4 -- CallbackHandler is no longer available.

    In v4, LangGraph tracing uses the ``langfuse.openai`` drop-in (which
    auto-traces every OpenAI call) plus manual ``start_as_current_observation``
    spans.  There is no separate LangGraph callback handler.

    Kept for API compatibility; always returns None.
    """
    logger.debug(
        "Langfuse v4: no LangGraph callback handler -- use langfuse.openai drop-in."
    )
    return None


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
    """Log a full structured agent decision to Langfuse as an observation.

    Creates an "agent" observation containing the complete decision trace:
    - Inputs (bars, news, indicators, portfolio) for grounding evaluation
    - Tool call outputs (actual data seen by the agent)
    - Reasoning / CoT
    - Decision + fill for faithfulness evaluation
    - Latency and token usage

    Uses ``start_as_current_observation`` (Langfuse v4 API).

    Parameters
    ----------
    trace_id:
        Logical trace ID (used as observation name suffix for grouping).
        In Langfuse v4, traces are created automatically by the openai drop-in.
    t_now, ticker:
        Decision context.
    inputs:
        Perceived state dict (bars_recent, news_recent, indicators, portfolio).
    tool_outputs:
        List of tool-call result summaries (for grounding).
    reasoning:
        Raw CoT string from the LLM response.
    decision:
        Parsed decision dict ``{action, quantity, rationale}``.
    fill:
        Fill confirmation from the paper portfolio (None for "hold").
    latency_ms:
        Total perceive -> act latency in milliseconds.
    token_usage:
        ``{prompt_tokens, completion_tokens, total_tokens}``.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        with client.start_as_current_observation(
            name=f"agent-decision-{ticker}",
            as_type="agent",
            input={
                "t_now": t_now,
                "ticker": ticker,
                **inputs,
                "tool_outputs": tool_outputs,
            },
            output={
                "reasoning": reasoning,
                "decision": decision,
                "fill": fill,
            },
            metadata={
                "trace_id": trace_id,
                "action": decision.get("action"),
                "latency_ms": round(latency_ms, 1),
                "token_usage": token_usage or {},
            },
        ):
            pass  # span is created and closed; content captured above
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log decision to Langfuse: %s", exc)


def log_pm_decision(
    *,
    run_id: str,
    t_now: str,
    model: str,
    tickers: list[str],
    reports: list[dict[str, Any]],
    discussion: list[dict[str, Any]],
    portfolio: dict[str, Any],
    tool_outputs: list[dict[str, Any]],
    mcp_calls: list[dict[str, Any]],
    rationale: str,
    target_weights: dict[str, Any],
    orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    past_context: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log one PM decision to Langfuse with hierarchical analyst spans.

    Trace structure (Langfuse v4 nested observations):
    ::

        pm-decision-{t_now}   [agent, parent]
          ├── analyst-technical  [span]
          ├── analyst-news       [span]
          ├── analyst-risk       [span]
          ├── investment-debate  [span, one turn each]
          ├── research-manager   [span]
          ├── trader             [span]
          ├── risk-debate        [span, one turn each]
          └── portfolio-manager  [span, final allocation]

    The ``past_context`` injected into prompts is recorded in the parent's
    input so the memory layer is visible in the Langfuse UI.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        meta = {
            "mode": "pm_multi_agent",
            "run_id": run_id,
            "model": model,
            "tickers": tickers,
            "t_now": t_now,
            "has_past_context": bool(past_context),
        }
        if metadata:
            meta.update(metadata)

        # Build role → report list for analyst child spans
        reports_by_role: dict[str, list[dict[str, Any]]] = {}
        for rpt in reports:
            role = str(rpt.get("analyst", "unknown"))
            reports_by_role.setdefault(role, []).append(rpt)

        with client.start_as_current_observation(
            name=f"pm-decision-{t_now}",
            as_type="agent",
            input={
                "t_now": t_now,
                "tickers": tickers,
                "portfolio": portfolio,
                "mcp_calls": mcp_calls,
                "past_context_preview": past_context[:500] if past_context else "",
            },
            output={
                "rationale": rationale,
                "target_weights": target_weights,
                "orders": orders,
                "fills": fills,
            },
            metadata=meta,
        ):
            # Child spans for analyst reports
            for role in ("technical", "news", "risk"):
                role_reports = reports_by_role.get(role, [])
                role_tool_outputs = [
                    o for o in tool_outputs if str(o.get("tool", "")) in {
                        "get_price_history", "compute_indicators", "get_current_regime",
                        "get_news_items_cache_first", "get_portfolio",
                    }
                ]
                with client.start_as_current_observation(
                    name=f"analyst-{role}",
                    as_type="span",
                    input={
                        "role": role,
                        "tickers": tickers,
                        "tool_outputs_count": len(role_tool_outputs),
                    },
                    output={"reports": role_reports},
                ):
                    pass

            # Child spans for discussion turns (investment debate, RM, trader, risk)
            stage_order = [
                "investment_debate",
                "research_manager",
                "trader",
                "risk_debate",
            ]
            stage_turns: dict[str, list[dict[str, Any]]] = {}
            for turn in (discussion or []):
                stage = str(turn.get("stage", "discussion"))
                stage_turns.setdefault(stage, []).append(turn)

            for stage in stage_order:
                turns = stage_turns.get(stage, [])
                if not turns:
                    continue
                speakers = [str(t.get("speaker", "")) for t in turns]
                with client.start_as_current_observation(
                    name=stage,
                    as_type="span",
                    input={"speakers": speakers, "n_turns": len(turns)},
                    output={"turns": turns},
                ):
                    pass

            # Final PM allocation span
            with client.start_as_current_observation(
                name="portfolio-manager",
                as_type="span",
                input={"tickers": tickers, "portfolio_nav": portfolio.get("nav")},
                output={
                    "target_weights": target_weights,
                    "rationale": rationale[:500],
                },
            ):
                pass

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log PM decision to Langfuse: %s", exc)


def score_trace(
    trace_id: str,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """Attach a numeric score to the current Langfuse trace context.

    Use after evaluation (e.g. faithfulness, grounding, sophistication)
    so scores appear in the Langfuse UI and enable regime-segmented dashboards.

    Parameters
    ----------
    trace_id:
        Logical trace ID (for logging only; v4 scores attach to current context).
    name:
        Score name (e.g. ``"faithfulness"``, ``"grounding"``, ``"sophistication"``).
    value:
        Numeric score value.
    comment:
        Optional note (e.g. regime label, decision summary).
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.score_current_trace(name=name, value=value, comment=comment)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to score trace %s: %s", trace_id, exc)


def flush() -> None:
    """Flush the Langfuse client to ensure all queued events are sent.

    Call at the end of a backtest run to avoid data loss on process exit.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to flush Langfuse: %s", exc)
