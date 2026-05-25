"""Reasoning quality metrics — faithfulness, grounding, sophistication.

This is the primary differentiator of this thesis vs. TradingAgents and
similar work: we score the *quality and faithfulness* of the LLM's
chain-of-thought reasoning against the actual tool outputs and executed orders.

Metrics (defined in docs/PLAN.md §3.3)
----------------------------------------
1. **Faithfulness** — does the executed order match the stated reasoning?
   Measured by comparing the ``decision.action`` field against the
   ``decision.rationale`` field.  A model that reasons "sell because RSI > 70"
   but places a buy order has low faithfulness.

2. **Grounding** — does the reasoning cite tool outputs that were actually
   returned, or hallucinate numbers?
   Measured by extracting numeric claims from the CoT and checking them against
   the actual tool outputs in the Langfuse trace.

3. **Sophistication** — a compact expert rubric (risk management, uncertainty
   acknowledgement, regime adaptation).  Scored automatically by an LLM judge
   on a sample, then validated against human annotation.

Reference: KellyBench (arXiv:2604.27865) §4 introduces the 52-point
sophistication rubric.  We adapt it to a smaller set of criteria suitable for
daily trading decisions.

Status: STUB — implementation in J9 sprint after traces are collected.
"""

from __future__ import annotations

from typing import Any


def compute_faithfulness(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute faithfulness score over a list of agent decisions.

    A decision is "faithful" if the executed action is consistent with the
    stated rationale.

    Parameters
    ----------
    decisions:
        List of decision dicts, each containing:
        - ``action``: "buy" | "sell" | "hold"
        - ``rationale``: free-text CoT string
        - ``fill``: executed fill dict (``None`` if "hold")
        - ``t_now``: simulation time of decision

    Returns
    -------
    dict[str, Any]
        ``{"faithfulness_rate": float, "n_decisions": int,
           "n_faithful": int, "examples_unfaithful": list}``.

    Raises
    ------
    NotImplementedError
        Until the implementation sprint (J9).
    """
    raise NotImplementedError(
        "compute_faithfulness() not yet implemented — see docs/PLAN.md J9."
    )


def compute_grounding(
    decisions: list[dict[str, Any]],
    tool_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute grounding score — fraction of numeric claims that match tool outputs.

    Parameters
    ----------
    decisions:
        Agent decision dicts (with CoT reasoning).
    tool_outputs:
        Actual tool output dicts from the Langfuse trace, aligned with decisions.

    Returns
    -------
    dict[str, Any]
        ``{"grounding_rate": float, "n_claims_checked": int,
           "n_grounded": int}``.

    Raises
    ------
    NotImplementedError
        Until the implementation sprint (J9).
    """
    raise NotImplementedError(
        "compute_grounding() not yet implemented — see docs/PLAN.md J9."
    )


def compute_sophistication_llm_judge(
    decisions: list[dict[str, Any]],
    judge_model: str = "gpt-4.1",
    sample_size: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Score reasoning sophistication using an LLM judge on a random sample.

    Criteria (adapted from KellyBench sophistication rubric):
    - Risk management acknowledgement (position sizing, stop-loss reasoning)
    - Uncertainty communication (confidence bounds, alternative scenarios)
    - Regime adaptation (does the reasoning mention current market conditions?)
    - Coherence (is the reasoning logically consistent?)

    Parameters
    ----------
    decisions:
        Full list of agent decisions.
    judge_model:
        OpenAI model used as judge (default ``"gpt-4.1"``).
    sample_size:
        Number of decisions to score (random sample for cost control).
    seed:
        Random seed for the sample selection.

    Returns
    -------
    dict[str, Any]
        ``{"mean_score": float, "std_score": float, "n_scored": int,
           "rubric_breakdown": dict}``.

    Raises
    ------
    NotImplementedError
        Until the implementation sprint (J9).
    """
    raise NotImplementedError(
        "compute_sophistication_llm_judge() not yet implemented — see docs/PLAN.md J9."
    )


def compute_all_reasoning_metrics(
    decisions: list[dict[str, Any]],
    tool_outputs: list[dict[str, Any]],
    regime_labels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute faithfulness, grounding, and sophistication, segmented by regime.

    Status: STUB.

    Raises
    ------
    NotImplementedError
        Until J9.
    """
    raise NotImplementedError(
        "compute_all_reasoning_metrics() not yet implemented — see docs/PLAN.md J9."
    )
