"""Single-agent orchestrator — perceive → reason/act → observe LangGraph graph.

Status: STUB — implementation in the next sprint.

Architecture
------------
Minimal LangGraph StateGraph with 3 nodes:

1. **perceive** — fetches price history, recent news, indicators, and portfolio
   state by calling the MCP data and analytics tools.  All calls go through the
   same MCP tool interface used in live trading (t_now enforced).

2. **reason_and_act** — passes the perceived state to the LLM (OpenAI, keyed by
   ``settings.agent_model_dev`` in dev), structured to produce:
   - Chain-of-thought reasoning (logged to Langfuse)
   - A trading decision: ``{"action": "buy"|"sell"|"hold", "ticker": ..., "quantity": ..., "rationale": ...}``
   - The action is then executed via the execution MCP server.

3. **observe** — records the fill, updates NAV, and logs the full decision struct
   (inputs, CoT, action, fill) to Langfuse as a trace.

Langfuse integration: uses the native LangGraph callback handler (passed in
the graph config), not manual instrumentation of every node.

Why LangGraph (and not a hand-rolled loop)?
-------------------------------------------
LangGraph provides: explicit state schema, checkpoints (resume-after-crash),
cyclic graphs (needed for the week-2 debate pattern), and native Langfuse
callback integration.  The minimal 3-node graph is the tutorial shape —
low learning overhead, high payoff when extended to multi-agent in week 2.
See docs/DECISIONS.md.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# State schema (typed — LangGraph uses this for graph state management)
# ---------------------------------------------------------------------------


class AgentState:  # noqa: B903
    """Typed state object passed between LangGraph nodes.

    Attributes
    ----------
    ticker : str
        The equity ticker for this decision cycle.
    t_now_str : str
        Current simulation time (ISO-8601).
    bars : list[dict]
        Price history visible to the agent (≤ t_now).
    news : list[dict]
        Recent news headlines (≤ t_now).
    indicators : dict
        Latest indicator values.
    portfolio : dict
        Current portfolio snapshot.
    reasoning : str
        The LLM's chain-of-thought from the reason_and_act node.
    decision : dict
        Parsed trading decision: {action, ticker, quantity, rationale}.
    fill : dict | None
        Execution fill confirmation (None if action == "hold").
    regime : str | None
        Detected market regime.
    """

    def __init__(self) -> None:
        self.ticker: str = ""
        self.t_now_str: str = ""
        self.bars: list[dict[str, Any]] = []
        self.news: list[dict[str, Any]] = []
        self.indicators: dict[str, Any] = {}
        self.portfolio: dict[str, Any] = {}
        self.reasoning: str = ""
        self.decision: dict[str, Any] = {}
        self.fill: dict[str, Any] | None = None
        self.regime: str | None = None


# ---------------------------------------------------------------------------
# Placeholder — real implementation in the next sprint
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Build and return the LangGraph StateGraph.

    Not yet implemented — returns None as a placeholder.

    Raises
    ------
    NotImplementedError
        Until the implementation sprint.
    """
    raise NotImplementedError(
        "build_graph() is not yet implemented. "
        "See docs/PLAN.md (J4) for the implementation sprint."
    )


def run_single_step(
    ticker: str,
    state: AgentState | None = None,
) -> AgentState:
    """Run one perceive → reason/act → observe cycle.

    Not yet implemented.

    Raises
    ------
    NotImplementedError
        Until the implementation sprint.
    """
    raise NotImplementedError(
        "run_single_step() is not yet implemented. See docs/PLAN.md (J4)."
    )
