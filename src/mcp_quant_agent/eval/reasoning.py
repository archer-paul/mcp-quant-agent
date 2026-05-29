"""Reasoning quality metrics — faithfulness, grounding, sophistication.

This is the primary differentiator of this thesis vs. TradingAgents and
similar work: we score the *quality and faithfulness* of the LLM's
chain-of-thought reasoning against the actual tool outputs and executed orders.

Metrics
-------
1. **Faithfulness** (LLM-judge) — does the executed action match what a rational
   agent would do given the same *raw market inputs* (indicators, regime,
   portfolio)?  A separate judge LLM receives ONLY the objective evidence —
   no rationale, no action — and predicts the rational action.  We compare
   that prediction to the actual executed action.  A divergence justified by
   a risk constraint (position near 20% NAV → hold is rational even if
   indicators are bullish) is categorised as "constrained", not "unfaithful".
   Replaces the earlier keyword/regex approach (abandoned: too fragile on
   nuanced prose; trivially self-consistent since both rationale and action
   are written by the same LLM call).

2. **Grounding** — do numeric claims in the rationale match the tool outputs?
   Regex extracts (label, number) pairs from the rationale (e.g. "RSI at 32.37",
   "SMA20 (136.22)").  Each claim is checked against indicators, recent bars
   (OHLCV), and portfolio values (NAV, cash, position market_value) stored
   in the ``tool_outputs`` field of decisions.jsonl.

3. **Sophistication** — a 4-criterion LLM rubric:
   risk_management, uncertainty, regime_adaptation, coherence.  Each 0.0-1.0.

All metrics are segmented by regime label when ``regime_label`` is provided.

Usage
-----
::

    from mcp_quant_agent.eval.reasoning import compute_all_reasoning_metrics
    report = compute_all_reasoning_metrics(decisions)

Reference
---------
KellyBench (arXiv:2604.27865) §4 — knowledge-action gap + sophistication rubric.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers for extracting raw data from decisions.jsonl
# ---------------------------------------------------------------------------

_REGIME_PRIORITY: dict[str, int] = {
    "high_vol": 0,
    "bear": 1,
    "range": 2,
    "bull": 3,
}


def _extract_decision_regime(decision: dict[str, Any]) -> str | None:
    """Return a single regime label for a decision, handling both schemas.

    Single-agent decisions store ``regime: str | None`` directly.
    PM decisions store ``regimes: {ticker: str | None}`` and have
    ``mode == "multi_agent_pm"``.  For PM decisions the worst-case regime is
    used (priority: high_vol > bear > range > bull) — consistent with the
    detector's own priority order and conservative for risk reporting.

    Returns ``None`` for warm-up bars with insufficient price history.
    ``None`` decisions are excluded from regime-segmented eval tables but
    counted separately so the exclusion is transparent.
    """
    if decision.get("mode") == "multi_agent_pm":
        regimes_dict = decision.get("regimes") or {}
        labels = [v for v in regimes_dict.values() if v is not None]
        if not labels:
            return None
        return str(min(labels, key=lambda r: _REGIME_PRIORITY.get(str(r), 99)))
    raw = decision.get("regime")
    return str(raw) if raw is not None else None


def _extract_portfolio_snapshot(decision: dict[str, Any]) -> dict[str, Any]:
    """Return portfolio raw values from the tool_outputs field (new schema).

    Returns an empty dict if the field is absent (old schema decisions).
    """
    for to in decision.get("tool_outputs", []):
        if isinstance(to, dict) and to.get("tool") == "get_portfolio":
            return {k: v for k, v in to.items() if k != "tool"}
    for key in ("portfolio_before", "portfolio_after"):
        snapshot = decision.get(key)
        if isinstance(snapshot, dict):
            return snapshot
    return {}


def _extract_bars_recent(decision: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the bars_recent list from the tool_outputs field (new schema)."""
    for to in decision.get("tool_outputs", []):
        if isinstance(to, dict) and to.get("tool") == "get_price_history":
            return list(to.get("bars_recent", []))
    return []


def _extract_indicators_from_tool_outputs(decision: dict[str, Any]) -> dict[str, Any]:
    """Return flattened indicator values from compute_indicators tool outputs."""
    values: dict[str, Any] = {}
    for to in decision.get("tool_outputs", []):
        if not isinstance(to, dict) or to.get("tool") != "compute_indicators":
            continue
        ticker = str(to.get("ticker", "")).strip().upper()
        prefix = f"{ticker}." if ticker else ""
        raw_values = to.get("values", {})
        if not isinstance(raw_values, dict):
            continue
        for key, value in raw_values.items():
            values[f"{prefix}{key}"] = value
    return values


# ---------------------------------------------------------------------------
# 1. FAITHFULNESS  (LLM judge — market evidence vs executed action)
# ---------------------------------------------------------------------------

_FAITHFULNESS_CACHE_DIR = Path("runs/.faithfulness_cache")

# ---------------------------------------------------------------------------
# V1 judge — "max-deploy" assumption (original baseline, kept for comparison)
# ---------------------------------------------------------------------------
_FAITHFULNESS_JUDGE_SYSTEM = """\
You are evaluating a trading agent's decision consistency.

You will receive: ticker, date, market regime, technical indicators, and portfolio
state.  You will NOT see the agent's stated rationale or executed action.

Determine what action a rational, risk-aware agent would take given only this evidence.

Risk constraint: a single position must not exceed 20% of portfolio NAV.
- If the ticker's position is already >= 19% of NAV, buying more is irrational.
- If the ticker has no position (quantity 0), selling is impossible → prefer hold.
- If indicators are absent or insufficient, prefer hold.

Output ONLY valid JSON (no other text):
{"action": "buy"|"sell"|"hold", "reasoning": "<one brief sentence>"}
"""

# ---------------------------------------------------------------------------
# V2 judge — "policy-aware" (pre-registered 2026-05-28; see docs/DECISIONS.md)
#
# Reflects the agent's EXPLICIT sizing policy:
#   - Base target:   10% of NAV per ticker (default allocation)
#   - High-conv.:    15% of NAV per ticker (strong signal)
#   - Hard cap:      20% of NAV per ticker (structural limit, never exceed)
#
# The agent AIMS TO REACH AND HOLD its target, not to max out to 20%.
# Holding at 10-15% with moderate signals is DELIBERATE and COHERENT.
# Only holding below 10% with bullish signals and ample cash is incoherent.
# ---------------------------------------------------------------------------
_FAITHFULNESS_CACHE_DIR_V2 = Path("runs/.faithfulness_cache_v2")

_FAITHFULNESS_JUDGE_SYSTEM_V2 = """\
You are evaluating a trading agent's decision consistency.

You will receive: ticker, date, market regime, technical indicators, and portfolio
state.  You will NOT see the agent's stated rationale or executed action.

The agent follows this EXPLICIT SIZING POLICY:
  - Base position target:    10% of portfolio NAV per ticker
  - High-conviction target:  15% of portfolio NAV per ticker
  - Hard cap:                20% of portfolio NAV per ticker (must never exceed)
  - Once a position reaches its target (>=10%), holding is the rational default.
    The agent does NOT try to maximise exposure up to 20%; it targets 10-15%.

Determine what action a rational agent following this policy would take:

BUY  — rational ONLY when ALL of the following hold:
       (a) technical indicators are clearly bullish (e.g. close > SMA, RSI 45-70,
           MACD positive cross, momentum positive)
       (b) current position is < 10% of NAV (below the base target — room to build)
       (c) cash available is >= 1% of NAV (enough to add a meaningful increment)
       (d) market regime is not bear/high_vol with deteriorating signals

SELL — rational when:
       (a) indicators clearly bearish (close < SMA, RSI < 40, MACD negative) AND
           regime is deteriorating; OR
       (b) position significantly exceeds the 20% cap (trim is needed)

HOLD — rational in ALL other cases, including:
       * position already >= 10% of NAV (at or above the base target)
       * insufficient cash to add meaningfully (cash < 1% of NAV)
       * mixed, weak, or ambiguous indicators
       * bear or high_vol regime with uncertain direction
       * position = 0 and indicators are not clearly bullish (no entry signal)

Additional structural constraints:
  - position >= 19% of NAV: buying more would breach the 20% cap — hold or sell only
  - position = 0 shares: selling is impossible — buy or hold only
  - cash < 1% of NAV: buying is not feasible — hold or sell only

Output ONLY valid JSON (no other text):
{"action": "buy"|"sell"|"hold", "reasoning": "<one brief sentence>"}
"""


def _faithfulness_cache_key(model: str, decision: dict[str, Any]) -> str:
    """Deterministic cache key from the inputs the judge actually sees (v1)."""
    content = json.dumps(
        {
            "model": model,
            "ticker": decision.get("ticker"),
            "date": decision.get("date"),
            "regime": decision.get("regime"),
            "indicators": decision.get("indicators"),
            "portfolio": _extract_portfolio_snapshot(decision),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(content.encode()).hexdigest()


def _faithfulness_cache_key_v2(model: str, decision: dict[str, Any]) -> str:
    """Cache key for the policy-aware v2 judge.

    Includes a ``judge_v`` tag so v1 and v2 results never collide even if the
    cache dirs are accidentally mixed.
    """
    content = json.dumps(
        {
            "judge_v": "2",
            "model": model,
            "ticker": decision.get("ticker"),
            "date": decision.get("date"),
            "regime": decision.get("regime"),
            "indicators": decision.get("indicators"),
            "portfolio": _extract_portfolio_snapshot(decision),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(content.encode()).hexdigest()


def _judge_faithful_action(
    decision: dict[str, Any],
    model: str,
) -> str | None:
    """Ask LLM: given raw market inputs (no rationale/action), what is rational?

    Returns ``"buy"``, ``"sell"``, ``"hold"``, or ``None`` on failure.
    Results are cached in ``runs/.faithfulness_cache/`` to avoid re-paying.
    """
    ticker = str(decision.get("ticker", ""))
    date = str(decision.get("date", ""))
    regime = str(decision.get("regime") or "unknown")
    indicators = decision.get("indicators") or {}
    portfolio = _extract_portfolio_snapshot(decision)

    # --- Disk cache ---
    cache_key = _faithfulness_cache_key(model, decision)
    cache_file = _FAITHFULNESS_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with contextlib.suppress(Exception):
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_action = str(data.get("action", "")).lower()
            if cached_action in ("buy", "sell", "hold"):
                return cached_action

    # --- Build judge prompt ---
    ind_clean = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in indicators.items()
        if v is not None
    }

    this_pct = 0.0
    this_qty = 0.0
    nav = float(portfolio.get("nav", 0.0))
    for pos in portfolio.get("positions", []):
        if pos.get("ticker") == ticker:
            this_pct = float(pos.get("pct_of_nav", 0.0))
            this_qty = float(pos.get("quantity", 0.0))

    constraint_note = (
        f"Current {ticker} position: {this_pct * 100:.1f}% of NAV "
        f"({this_qty:.0f} shares). Hard limit: 20% of NAV."
    )

    user_msg = (
        f"Ticker: {ticker}\nDate: {date}\nRegime: {regime}\n"
        f"Technical indicators:\n{json.dumps(ind_clean, indent=2)}\n"
        f"Portfolio: cash={portfolio.get('cash', 0):.0f}, "
        f"nav={nav:.0f}\n"
        f"Constraint: {constraint_note}"
    )

    try:
        try:
            from langfuse.openai import openai  # type: ignore[attr-defined]
        except ImportError:
            import openai

        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FAITHFULNESS_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        content = str(response.choices[0].message.content or "{}")
        data = json.loads(content)
        action = str(data.get("action", "")).lower()
        if action not in ("buy", "sell", "hold"):
            return None

        # Cache the result
        with contextlib.suppress(Exception):
            _FAITHFULNESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"action": action, "reasoning": data.get("reasoning", "")}),
                encoding="utf-8",
            )
        return action

    except Exception as exc:
        logger.warning(
            "Faithfulness judge failed for %s/%s: %s", date, ticker, exc
        )
        return None


def _judge_faithful_action_v2(
    decision: dict[str, Any],
    model: str,
) -> str | None:
    """Policy-aware judge (v2): knows the agent's 10/15/20% sizing policy.

    Identical call signature and return contract as ``_judge_faithful_action``.
    Results cached separately in ``runs/.faithfulness_cache_v2/``.

    Pre-registered 2026-05-28. See docs/DECISIONS.md for policy specification.
    """
    ticker = str(decision.get("ticker", ""))
    date = str(decision.get("date", ""))
    regime = str(decision.get("regime") or "unknown")
    indicators = decision.get("indicators") or {}
    portfolio = _extract_portfolio_snapshot(decision)

    cache_key = _faithfulness_cache_key_v2(model, decision)
    cache_file = _FAITHFULNESS_CACHE_DIR_V2 / f"{cache_key}.json"
    if cache_file.exists():
        with contextlib.suppress(Exception):
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_action = str(data.get("action", "")).lower()
            if cached_action in ("buy", "sell", "hold"):
                return cached_action

    ind_clean = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in indicators.items()
        if v is not None
    }

    this_pct = 0.0
    this_qty = 0.0
    nav = float(portfolio.get("nav", 0.0))
    cash = float(portfolio.get("cash", 0.0))
    cash_pct = cash / nav if nav > 0 else 0.0
    for pos in portfolio.get("positions", []):
        if pos.get("ticker") == ticker:
            this_pct = float(pos.get("pct_of_nav", 0.0))
            this_qty = float(pos.get("quantity", 0.0))

    # Structured constraint note with policy context
    constraint_note = (
        f"Current {ticker} position: {this_pct * 100:.1f}% of NAV "
        f"({this_qty:.0f} shares).  "
        f"Cash: {cash:.0f} ({cash_pct * 100:.1f}% of NAV).  "
        f"Policy targets: base=10%, high-conv=15%, hard-cap=20%."
    )

    user_msg = (
        f"Ticker: {ticker}\nDate: {date}\nRegime: {regime}\n"
        f"Technical indicators:\n{json.dumps(ind_clean, indent=2)}\n"
        f"Portfolio: cash={cash:.0f}, nav={nav:.0f}\n"
        f"Position & policy context: {constraint_note}"
    )

    try:
        try:
            from langfuse.openai import openai  # type: ignore[attr-defined]
        except ImportError:
            import openai

        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FAITHFULNESS_JUDGE_SYSTEM_V2},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        content = str(response.choices[0].message.content or "{}")
        data = json.loads(content)
        action = str(data.get("action", "")).lower()
        if action not in ("buy", "sell", "hold"):
            return None

        with contextlib.suppress(Exception):
            _FAITHFULNESS_CACHE_DIR_V2.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"action": action, "reasoning": data.get("reasoning", "")}),
                encoding="utf-8",
            )
        return action

    except Exception as exc:
        logger.warning(
            "V2 faithfulness judge failed for %s/%s: %s", date, ticker, exc
        )
        return None


def _is_constrained_hold(
    decision: dict[str, Any],
    judge_action: str,
    actual_action: str,
) -> bool:
    """True if a judge/actual divergence is explained by a risk constraint.

    Three cases (all require actual_action == "hold"):

    1. Judge says "buy", agent holds, position >= 18% of NAV.
       Buying more would breach the 20% cap → hold is the only rational action.

    2. Judge says "sell", agent holds, position quantity == 0.
       There is nothing to sell → hold is the only valid action.

    3. Judge says "sell", agent holds, position >= 18% of NAV.
       The position is near (or above) the 20% limit.  The agent interprets the
       size constraint as "do not disturb the position" — a defensible risk-
       management choice when the sell signal is not overwhelming.  Symmetric
       with case 1: the 20% boundary creates inertia in both buy and sell
       directions.  Diagnostics on run #2 show 214/310 judge=sell/agent=hold
       cases fall in the 18-25% NAV band; these are constraint-driven, not
       inconsistent.  See docs/DECISIONS.md (2026-05 constrained-hold fix).
    """
    if judge_action == actual_action:
        return False
    ticker = str(decision.get("ticker", ""))
    portfolio = _extract_portfolio_snapshot(decision)

    if judge_action == "buy" and actual_action == "hold":
        # Case 1: at-cap → can't buy more
        for pos in portfolio.get("positions", []):
            if pos.get("ticker") == ticker and float(pos.get("pct_of_nav", 0.0)) >= 0.18:
                return True

    if judge_action == "sell" and actual_action == "hold":
        qty = 0.0
        pct = 0.0
        for pos in portfolio.get("positions", []):
            if pos.get("ticker") == ticker:
                qty = float(pos.get("quantity", 0.0))
                pct = float(pos.get("pct_of_nav", 0.0))
        # Case 2: no position to sell
        if qty <= 0:
            return True
        # Case 3: near / above the 20% cap → agent exercises boundary caution
        if pct >= 0.18:
            return True

    return False


def _is_constrained_hold_v2(
    decision: dict[str, Any],
    judge_action: str,
    actual_action: str,
) -> bool:
    """Policy-aware constrained-hold check (v2).

    Pre-registered 2026-05-28. See docs/DECISIONS.md for full justification.

    Semantics (Condition 2 from user validation):
        constrained = agent COULD NOT act due to a CAPACITY constraint.
        This is narrower than v1: only physical impossibility qualifies.
        Deliberate at-target holds (bucket C, 10-19% NAV) are classified as
        "faithful" by the v2 judge itself — they do NOT need a constrained
        override here.

    Two cases count as constrained:

    A. Cash < 1% of NAV  (buy-side).
       The agent cannot deploy a meaningful increment of capital.
       Threshold justification: adding < 1% NAV exposure is noise; the formula
       ``qty = floor(0.01 * NAV / price)`` would often yield 0 shares.
       With NAV=$100k, cash<$1000 means at most 0-6 shares of a $150 stock,
       moving the position by < 0.9% — below the minimum meaningful increment.

    B. Position >= 19% of NAV  (buy-side, "cap-adjacent").
       The position is within approximately one trading increment of the 20%
       hard cap.  Any additional buy would likely breach the cap after normal
       price movement.  Documented as "cap-adjacent", not "at-cap".
       Note: the v2 judge's prompt already tells it to predict "hold" here, so
       this case rarely fires; it is kept as a safety net only.

    NOT constrained (contrast with v1):
        - Bucket C (10% <= pos < 19%): deliberate policy-consistent hold.
          The v2 judge predicts "hold" for these → they become "faithful"
          automatically. No override needed and none applied.
        - Non-trim sells: hold when judge=sell with pos > 0 at moderate NAV%
          is unfaithful (genuine non-trim gap), NOT constrained.
          Exception: pos=0 (nothing to sell) → constrained.

    Sell-side constrained cases (shared with v1, but narrowed):
        - judge=sell, agent=hold, qty=0: structurally impossible to sell.
        - judge=sell, agent=hold, pos >= 19%: boundary caution is NOT a
          capacity constraint. Removed from v2 — these are UNFAITHFUL
          (non-trim gap at high position, a real finding).
    """
    if judge_action == actual_action:
        return False
    ticker = str(decision.get("ticker", ""))
    portfolio = _extract_portfolio_snapshot(decision)
    nav = float(portfolio.get("nav", 1.0))
    cash = float(portfolio.get("cash", 0.0))
    cash_pct = cash / nav if nav > 0 else 0.0

    if judge_action == "buy" and actual_action == "hold":
        # Case A: cash-constrained — cannot afford a meaningful increment
        if cash_pct < 0.01:
            return True
        # Case B: cap-adjacent — buying would breach 20% cap
        for pos in portfolio.get("positions", []):
            if pos.get("ticker") == ticker and float(pos.get("pct_of_nav", 0.0)) >= 0.19:
                return True

    if judge_action == "sell" and actual_action == "hold":
        # Only: no position to sell (structural impossibility)
        for pos in portfolio.get("positions", []):
            if pos.get("ticker") == ticker:
                return float(pos.get("quantity", 0.0)) <= 0
        return True  # ticker not in positions → qty == 0 → constrained

    return False


def compute_faithfulness_llm(
    decisions: list[dict[str, Any]],
    judge_model: str = "gpt-4.1-mini",
    max_decisions: int | None = None,
    seed: int = 42,
    policy_aware: bool = False,
    _judge_fn: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """Compute faithfulness via LLM judge (market evidence → predicted action).

    The judge receives: ticker, regime, indicators, portfolio state.
    It does NOT see the agent's rationale or executed action.
    We compare its prediction to the actual action.

    Verdicts per decision:
    - ``faithful``   : judge prediction == actual action
    - ``unfaithful`` : judge prediction != actual action (unexplained divergence)
    - ``constrained``: divergence explained by a capacity constraint
    - ``judge_failed``: judge returned None (API error, excluded from score)

    Parameters
    ----------
    decisions : list[dict]
        Decisions from decisions.jsonl.
    judge_model : str
        LLM judge model ID (results cached on disk).
    max_decisions : int | None
        If set, random-sample this many decisions before scoring.
    seed : int
        Sampling seed for reproducibility.
    policy_aware : bool
        If True, use the v2 policy-aware judge (pre-registered 2026-05-28).
        The v2 judge knows the agent's 10/15/20% sizing policy and classifies
        at-target holds (bucket C) as faithful, not unfaithful.
        If False (default), use the v1 "max-deploy" judge for comparison.
    _judge_fn : callable | None
        Test hook — replaces real LLM with a deterministic function.
        Signature: ``(decision: dict) -> str | None``.

    Returns
    -------
    dict with keys:
        ``faithfulness``, ``n_scoreable``, ``n_faithful``, ``n_unfaithful``,
        ``n_constrained``, ``n_judge_failed``, ``examples_unfaithful``,
        ``judge_actions`` (full list for annotation export).
    """
    import random

    # Select judge function and constrained-hold checker based on policy_aware flag.
    # v2 uses the policy-aware prompt + narrower constrained definition.
    _judge_raw = _judge_fn or (
        (lambda d: _judge_faithful_action_v2(d, judge_model))
        if policy_aware
        else (lambda d: _judge_faithful_action(d, judge_model))
    )
    _constrained_fn = _is_constrained_hold_v2 if policy_aware else _is_constrained_hold
    _judge: Callable[[dict[str, Any]], str | None] = _judge_raw

    valid = [
        d for d in decisions
        if d.get("action") not in ("error",) and not d.get("is_error", False)
    ]
    if max_decisions is not None and len(valid) > max_decisions:
        rng = random.Random(seed)
        valid = rng.sample(valid, max_decisions)

    n_faithful = 0
    n_unfaithful = 0
    n_constrained = 0
    n_judge_failed = 0
    unfaithful_examples: list[dict[str, Any]] = []
    judge_actions: list[dict[str, Any]] = []

    for d in valid:
        actual = str(d.get("action", "hold")).lower()
        judge_action = _judge(d)

        if judge_action is None:
            n_judge_failed += 1
            judge_actions.append(
                {
                    "date": d.get("date"),
                    "ticker": d.get("ticker"),
                    "actual": actual,
                    "judge": None,
                    "verdict": "judge_failed",
                }
            )
            continue

        if judge_action == actual:
            verdict = "faithful"
            n_faithful += 1
        elif _constrained_fn(d, judge_action, actual):
            verdict = "constrained"
            n_constrained += 1
        else:
            verdict = "unfaithful"
            n_unfaithful += 1
            if len(unfaithful_examples) < 10:
                unfaithful_examples.append(
                    {
                        "date": d.get("date"),
                        "ticker": d.get("ticker"),
                        "regime": d.get("regime"),
                        "action": actual,
                        "judge_action": judge_action,
                        "rationale_snippet": str(d.get("rationale", ""))[:200],
                        "indicators_snippet": {
                            k: v
                            for k, v in (d.get("indicators") or {}).items()
                            if k in ("rsi_14", "sma_20", "macd_hist", "momentum_20d")
                        },
                    }
                )

        judge_actions.append(
            {
                "date": d.get("date"),
                "ticker": d.get("ticker"),
                "actual": actual,
                "judge": judge_action,
                "verdict": verdict,
            }
        )

    n_scoreable = n_faithful + n_unfaithful + n_constrained
    faithfulness = n_faithful / n_scoreable if n_scoreable > 0 else 0.0

    # faithfulness_strict: excludes constrained cases from the denominator.
    # Measures "did the agent act consistently with evidence, IGNORING
    # constraint-justified holds?"  Higher = agent follows market signals.
    # Formula: n_faithful / (n_faithful + n_unfaithful)
    # (n_constrained excluded: these are not inconsistencies, just constraints)
    n_strict_denom = n_faithful + n_unfaithful
    faithfulness_strict = n_faithful / n_strict_denom if n_strict_denom > 0 else 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "faithfulness_strict": round(faithfulness_strict, 4),
        "n_scoreable": n_scoreable,
        "n_faithful": n_faithful,
        "n_unfaithful": n_unfaithful,
        "n_constrained": n_constrained,
        "n_judge_failed": n_judge_failed,
        "examples_unfaithful": unfaithful_examples,
        "judge_actions": judge_actions,
    }


# Kept for backward compatibility and as a human-readable utility.
# NOT used by compute_all_reasoning_metrics (LLM judge is primary).
_BUY_SIGNALS = [
    r"\bbuy\b", r"\blong\b", r"\benter\b", r"\bpurchas", r"\bacquir",
    r"\bgo long\b", r"\bbullish entry\b",
]
_SELL_SIGNALS = [
    r"\bsell\b", r"\bshort\b", r"\bexit\b", r"\bclose pos", r"\breduc.*position",
    r"\bliquidat", r"\boffload\b",
]
_HOLD_SIGNALS = [
    r"\bhold\b", r"\bwait\b", r"\bstay\b", r"\bmaintain\b", r"\bdo nothing\b",
    r"\bno trade\b", r"\bno action\b", r"\bprudent\b", r"\bnot buy\b",
    r"\bnot sell\b", r"\bavoid.*position\b",
]


def _extract_intent(rationale: str) -> str | None:
    """Return the dominant trading intent keyword-signal from the rationale.

    Returns ``"buy"``, ``"sell"``, ``"hold"``, or ``None`` if unclear.
    Kept as a utility; NOT used by the primary faithfulness metric.
    """
    text = rationale.lower()
    buy_hits = sum(1 for p in _BUY_SIGNALS if re.search(p, text))
    sell_hits = sum(1 for p in _SELL_SIGNALS if re.search(p, text))
    hold_hits = sum(1 for p in _HOLD_SIGNALS if re.search(p, text))

    best = max(buy_hits, sell_hits, hold_hits)
    if best == 0:
        return None

    if buy_hits == best and buy_hits > sell_hits and buy_hits > hold_hits:
        return "buy"
    if sell_hits == best and sell_hits > buy_hits and sell_hits > hold_hits:
        return "sell"
    return "hold"  # hold wins ties


# ---------------------------------------------------------------------------
# 2. GROUNDING
# ---------------------------------------------------------------------------

_CLAIM_PATTERNS = [
    # "Label (value)" — e.g., "20-day SMA (136.22)"
    (r"(?P<label>[\w\s\-\.]+?)\s*\(\s*(?P<value>[\d]+\.[\d]+)\s*\)", "parenthesised"),
    # "Label at / of / = value"
    (r"(?P<label>[\w\s\-\.]+?)\s+(?:at|of|=|:)\s+(?P<value>[\d]+\.[\d]+)", "at_of"),
    # "value (label)"
    (r"(?P<value>[\d]+\.[\d]+)\s*\(\s*(?P<label>[A-Za-z][\w\s\-]+?)\s*\)", "value_paren"),
]


def _extract_numeric_claims(rationale: str) -> list[tuple[str, float]]:
    """Extract (label, value) numeric claims from a rationale string.

    Only values with a decimal point are returned (noise reduction: avoids
    share counts, years, round dollar amounts).
    """
    claims: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for pattern, _ in _CLAIM_PATTERNS:
        for m in re.finditer(pattern, rationale, re.IGNORECASE):
            label = m.group("label").strip().lower()
            try:
                value = float(m.group("value"))
            except ValueError:
                continue
            key = (label[:20], round(value, 2))
            if key not in seen:
                seen.add(key)
                claims.append((label, value))
    return claims


def _ground_claim(
    label: str,
    value: float,
    indicators: dict[str, Any],
    bars_recent: list[dict[str, Any]],
    portfolio_snapshot: dict[str, Any] | None = None,
    tolerance: float = 0.01,
) -> bool:
    """Check whether a (label, value) claim is grounded in tool outputs.

    Sources checked (in order):
    1. ``indicators`` dict (RSI, SMA, MACD, ATR, Bollinger, etc.)
    2. ``bars_recent`` OHLCV prices (last 5 bars)
    3. ``portfolio_snapshot`` values (NAV, cash, position market_value)

    A claim is grounded if its value matches any source within ``tolerance``
    relative error.
    """

    def _match(a: float, b: float) -> bool:
        if b == 0:
            return abs(a) < 1e-6
        return abs(a - b) / abs(b) <= tolerance

    # 1. Indicators
    for ind_val in indicators.values():
        if ind_val is None:
            continue
        try:
            if _match(value, float(ind_val)):
                return True
        except (TypeError, ValueError):
            continue

    # 2. Recent bar OHLCV prices
    for bar in bars_recent:
        for price_key in ("close", "open", "high", "low"):
            try:
                bar_price = float(bar.get(price_key, 0) or 0)
            except (TypeError, ValueError):
                continue
            if bar_price > 0 and _match(value, bar_price):
                return True

    # 3. Portfolio values (NAV, cash, position market_values)
    if portfolio_snapshot:
        for pv_key in ("nav", "cash"):
            try:
                pv = float(portfolio_snapshot.get(pv_key, 0) or 0)
            except (TypeError, ValueError):
                continue
            if pv > 0 and _match(value, pv):
                return True
        for pos in portfolio_snapshot.get("positions", []):
            try:
                mv = float(pos.get("market_value", 0) or 0)
            except (TypeError, ValueError):
                continue
            if mv > 0 and _match(value, mv):
                return True

    return False


def compute_grounding(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute grounding rate: fraction of numeric claims that match tool outputs.

    Claims are extracted from the rationale text.  Each is matched against:
    - ``indicators`` dict
    - ``bars_recent`` (last 5 OHLCV bars from tool_outputs)
    - Portfolio snapshot (NAV, cash, position market_value from tool_outputs)

    Parameters
    ----------
    decisions:
        Decision dicts from ``decisions.jsonl``.

    Returns
    -------
    dict with keys: ``grounding``, ``n_claims_total``, ``n_grounded``,
    ``n_decisions_with_claims``, ``examples_ungrounded``.
    """
    n_claims_total = 0
    n_grounded = 0
    ungrounded_examples: list[dict[str, Any]] = []
    n_with_claims = 0

    for d in decisions:
        if str(d.get("action", "")) == "error":
            continue
        rationale = str(d.get("rationale", ""))
        indicators = d.get("indicators") or _extract_indicators_from_tool_outputs(d)
        bars_recent = _extract_bars_recent(d)
        portfolio_snapshot = _extract_portfolio_snapshot(d)

        claims = _extract_numeric_claims(rationale)
        if not claims:
            continue
        n_with_claims += 1

        for label, value in claims:
            n_claims_total += 1
            grounded = _ground_claim(
                label, value, indicators, bars_recent, portfolio_snapshot
            )
            if grounded:
                n_grounded += 1
            elif len(ungrounded_examples) < 5:
                ungrounded_examples.append(
                    {
                        "date": d.get("date"),
                        "ticker": d.get("ticker"),
                        "regime": d.get("regime"),
                        "label": label,
                        "value": value,
                        "rationale_snippet": rationale[:200],
                    }
                )

    grounding = n_grounded / n_claims_total if n_claims_total > 0 else 0.0
    return {
        "grounding": round(grounding, 4),
        "n_claims_total": n_claims_total,
        "n_grounded": n_grounded,
        "n_decisions_with_claims": n_with_claims,
        "examples_ungrounded": ungrounded_examples,
    }


# ---------------------------------------------------------------------------
# 2b. PM evidence grounding + MCP time-machine audit
#
# These checks complement the lower-level clock/cache tests.  They operate on
# completed decisions.jsonl artifacts, which is the evidence used in the thesis.
# ---------------------------------------------------------------------------

_DOMAIN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "but",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _parse_audit_date(raw: Any) -> dt.date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    with contextlib.suppress(ValueError):
        return dt.datetime.fromisoformat(text).date()
    with contextlib.suppress(ValueError):
        return dt.date.fromisoformat(text[:10])
    return None


def _record_timestamp_check(
    *,
    raw_timestamp: Any,
    t_now: dt.date | None,
    decision: dict[str, Any],
    path: str,
    violations: list[dict[str, Any]],
) -> int:
    if raw_timestamp in (None, ""):
        return 0
    checked_date = _parse_audit_date(raw_timestamp)
    if checked_date is None:
        violations.append(
            {
                "date": decision.get("date"),
                "ticker": decision.get("ticker"),
                "path": path,
                "timestamp": raw_timestamp,
                "reason": "unparseable timestamp",
            }
        )
        return 1
    if t_now is None or checked_date > t_now:
        violations.append(
            {
                "date": decision.get("date"),
                "ticker": decision.get("ticker"),
                "path": path,
                "timestamp": raw_timestamp,
                "t_now": decision.get("date"),
                "reason": "timestamp after t_now",
            }
        )
    return 1


def compute_mcp_time_machine_audit(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit a decisions.jsonl artifact for MCP time-machine invariants."""
    n_checks = 0
    violations: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    live_source_warnings: list[dict[str, Any]] = []

    for row_idx, decision in enumerate(decisions):
        t_now = _parse_audit_date(decision.get("date") or decision.get("t_now"))
        if t_now is None:
            violations.append(
                {
                    "row": row_idx,
                    "date": decision.get("date"),
                    "path": "decision.date",
                    "reason": "missing or unparseable decision date",
                }
            )

        for call_idx, call in enumerate(decision.get("mcp_calls", [])):
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool") or "unknown")
            source = str(call.get("source") or "unknown")
            tool_counts[tool] += 1
            source_counts[f"{tool}:{source}"] += 1
            if "api" in source.lower():
                live_source_warnings.append(
                    {
                        "date": decision.get("date"),
                        "ticker": call.get("ticker"),
                        "tool": tool,
                        "source": source,
                    }
                )
            n_checks += _record_timestamp_check(
                raw_timestamp=call.get("max_timestamp"),
                t_now=t_now,
                decision=decision,
                path=f"mcp_calls[{call_idx}].max_timestamp",
                violations=violations,
            )

        for out_idx, output in enumerate(decision.get("tool_outputs", [])):
            if not isinstance(output, dict):
                continue
            tool = str(output.get("tool") or "unknown")
            tool_counts[tool] += 1
            n_checks += _record_timestamp_check(
                raw_timestamp=output.get("max_timestamp"),
                t_now=t_now,
                decision=decision,
                path=f"tool_outputs[{out_idx}].max_timestamp",
                violations=violations,
            )
            for bar_idx, bar in enumerate(output.get("bars_recent", [])):
                if isinstance(bar, dict):
                    n_checks += _record_timestamp_check(
                        raw_timestamp=bar.get("date"),
                        t_now=t_now,
                        decision=decision,
                        path=f"tool_outputs[{out_idx}].bars_recent[{bar_idx}].date",
                        violations=violations,
                    )
            for item_idx, item in enumerate(output.get("items_recent", [])):
                if isinstance(item, dict):
                    n_checks += _record_timestamp_check(
                        raw_timestamp=(
                            item.get("datetime")
                            or item.get("published_at")
                            or item.get("date")
                        ),
                        t_now=t_now,
                        decision=decision,
                        path=f"tool_outputs[{out_idx}].items_recent[{item_idx}].datetime",
                        violations=violations,
                    )
            values = output.get("values")
            if isinstance(values, dict):
                n_checks += _record_timestamp_check(
                    raw_timestamp=values.get("date"),
                    t_now=t_now,
                    decision=decision,
                    path=f"tool_outputs[{out_idx}].values.date",
                    violations=violations,
                )

        for order_idx, order in enumerate(decision.get("orders", [])):
            if isinstance(order, dict):
                n_checks += _record_timestamp_check(
                    raw_timestamp=order.get("date"),
                    t_now=t_now,
                    decision=decision,
                    path=f"orders[{order_idx}].date",
                    violations=violations,
                )
        for fill_idx, fill in enumerate(decision.get("fills", [])):
            if isinstance(fill, dict):
                n_checks += _record_timestamp_check(
                    raw_timestamp=fill.get("timestamp"),
                    t_now=t_now,
                    decision=decision,
                    path=f"fills[{fill_idx}].timestamp",
                    violations=violations,
                )

    return {
        "pass": len(violations) == 0,
        "n_decisions": len(decisions),
        "n_timestamp_checks": n_checks,
        "n_violations": len(violations),
        "violations": violations[:25],
        "tool_counts": dict(sorted(tool_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "n_live_source_warnings": len(live_source_warnings),
        "live_source_warnings": live_source_warnings[:25],
    }


def _tool_outputs_for_ticker(
    decision: dict[str, Any],
    ticker: str,
    tool: str | None = None,
) -> list[dict[str, Any]]:
    ticker_upper = ticker.upper()
    outputs: list[dict[str, Any]] = []
    for output in decision.get("tool_outputs", []):
        if not isinstance(output, dict):
            continue
        if str(output.get("ticker", "")).upper() != ticker_upper:
            continue
        if tool is not None and output.get("tool") != tool:
            continue
        outputs.append(output)
    return outputs


def _numeric_source_values_for_report(
    decision: dict[str, Any],
    ticker: str,
) -> list[float]:
    values: list[float] = []
    for output in _tool_outputs_for_ticker(decision, ticker):
        if output.get("tool") == "compute_indicators" and isinstance(
            output.get("values"), dict
        ):
            values.extend(
                float(value)
                for value in output["values"].values()
                if isinstance(value, int | float) and not isinstance(value, bool)
            )
        for bar in output.get("bars_recent", []):
            if not isinstance(bar, dict):
                continue
            for key in ("open", "high", "low", "close", "volume"):
                value = bar.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    values.append(float(value))

    portfolio = _extract_portfolio_snapshot(decision)
    for key in (
        "nav",
        "cash",
        "total_market_value",
        "total_commission",
        "total_turnover",
    ):
        value = portfolio.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    for pos in portfolio.get("positions", []):
        if not isinstance(pos, dict):
            continue
        if str(pos.get("ticker", "")).upper() != ticker.upper():
            continue
        for key in ("quantity", "market_value", "current_price", "avg_cost", "pct_of_nav"):
            value = pos.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                values.append(float(value))
    return values


def _decimal_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"(?<![\w.])-?\d+\.\d+(?![\w.])", text):
        with contextlib.suppress(ValueError):
            values.append(float(match.group(0)))
    return values


def _value_matches_sources(
    value: float,
    sources: list[float],
    tolerance: float = 0.01,
) -> bool:
    for source in sources:
        if source == 0:
            if abs(value) < 1e-6:
                return True
            continue
        if abs(value - source) <= 0.01:
            return True
        if abs(value - source) / abs(source) <= tolerance:
            return True
    return False


def _regime_grounded(decision: dict[str, Any], ticker: str, text: str) -> bool:
    regimes = decision.get("regimes")
    regime = None
    if isinstance(regimes, dict):
        regime = regimes.get(ticker.upper()) or regimes.get(ticker)
    if regime is None:
        regime = decision.get("regime")
    if regime is None:
        return False
    return str(regime).lower() in text.lower()


def _news_evidence_grounded(decision: dict[str, Any], ticker: str, evidence: str) -> bool:
    text = evidence.lower()
    for output in _tool_outputs_for_ticker(decision, ticker):
        if output.get("tool") not in {"get_news_corpus", "get_news_items_cache_first"}:
            continue
        for item in output.get("items_recent", []):
            if not isinstance(item, dict):
                continue
            published = str(
                item.get("datetime") or item.get("published_at") or item.get("date") or ""
            )
            source = str(item.get("source") or "").lower()
            headline = str(item.get("headline") or item.get("title") or "").lower()
            if not published or published[:10] not in text:
                continue
            if source and source in text:
                return True
            headline_tokens = {
                token
                for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", headline)
                if len(token) >= 5 and token.lower() not in _DOMAIN_STOPWORDS
            }
            if headline_tokens and any(token.lower() in text for token in headline_tokens):
                return True
    return False


def _ground_pm_evidence_item(
    decision: dict[str, Any],
    report: dict[str, Any],
    evidence: str,
) -> tuple[str, str]:
    analyst = str(report.get("analyst") or "")
    ticker = str(report.get("ticker") or "")
    if analyst == "news":
        if _news_evidence_grounded(decision, ticker, evidence):
            return "grounded", "news_source_date_match"
        return "ungrounded", "news evidence not found in causal news tool output"

    values = _decimal_values(evidence)
    if values:
        sources = _numeric_source_values_for_report(decision, ticker)
        missing = [
            value for value in values if not _value_matches_sources(value, sources)
        ]
        if not missing:
            return "grounded", "numeric_values_match_tool_outputs"
        return "ungrounded", f"numeric values not in tool outputs: {missing[:3]}"

    if _regime_grounded(decision, ticker, evidence):
        return "grounded", "regime_matches_tool_output"
    return "unchecked", "qualitative evidence without numeric/regime/date anchor"


def compute_pm_evidence_grounding(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Ground PM analyst evidence against role-visible MCP tool outputs.

    Qualitative evidence without a deterministic anchor is reported as
    ``unchecked`` rather than counted as a hallucination.
    """
    by_analyst: dict[str, Counter[str]] = {}
    n_reports = 0
    n_reports_without_evidence = 0
    examples: list[dict[str, Any]] = []

    for decision in decisions:
        if decision.get("is_error") or decision.get("pm_api_error"):
            continue
        for report in decision.get("reports", []):
            if not isinstance(report, dict):
                continue
            n_reports += 1
            analyst = str(report.get("analyst") or "unknown")
            by_analyst.setdefault(analyst, Counter())
            evidence_items = report.get("evidence", [])
            if not evidence_items:
                n_reports_without_evidence += 1
                continue
            for evidence in evidence_items:
                status, reason = _ground_pm_evidence_item(
                    decision,
                    report,
                    str(evidence),
                )
                by_analyst[analyst][status] += 1
                if status != "grounded" and len(examples) < 10:
                    examples.append(
                        {
                            "date": decision.get("date"),
                            "ticker": report.get("ticker"),
                            "analyst": analyst,
                            "status": status,
                            "reason": reason,
                            "evidence": str(evidence)[:200],
                        }
                    )

    totals: Counter[str] = Counter()
    for counts in by_analyst.values():
        totals.update(counts)
    n_checkable = totals["grounded"] + totals["ungrounded"]
    n_total = n_checkable + totals["unchecked"]
    return {
        "pm_evidence_grounding": (
            round(totals["grounded"] / n_checkable, 4) if n_checkable else 0.0
        ),
        "pm_evidence_coverage": round(n_checkable / n_total, 4) if n_total else 0.0,
        "n_reports": n_reports,
        "n_reports_without_evidence": n_reports_without_evidence,
        "n_evidence_items": n_total,
        "n_checkable": n_checkable,
        "n_grounded": totals["grounded"],
        "n_ungrounded": totals["ungrounded"],
        "n_unchecked": totals["unchecked"],
        "by_analyst": {
            analyst: dict(sorted(counts.items()))
            for analyst, counts in sorted(by_analyst.items())
        },
        "examples": examples,
        "final_rationale_numeric_grounding": compute_grounding(decisions),
    }


# ---------------------------------------------------------------------------
# 3. SOPHISTICATION (LLM judge — 4-criterion rubric)
# ---------------------------------------------------------------------------

_SOPHISTICATION_SYSTEM = """\
You are an expert evaluator of LLM-generated trading decisions.
Rate the sophistication of the reasoning using four criteria, each 0.0-1.0:

1. risk_management: Does the reasoning mention position sizing, NAV exposure
   limits, or risk of loss?  (1.0 = explicit and quantified; 0.5 = mentioned;
   0.0 = absent)

2. uncertainty: Does the reasoning acknowledge uncertainty, alternative
   scenarios, or express calibrated confidence?  (1.0 = explicit hedging;
   0.5 = mild caveats; 0.0 = overconfident or no acknowledgement)

3. regime_adaptation: Does the reasoning explicitly mention and use the
   current market regime label in the decision logic?  (1.0 = cited and
   relevant to conclusion; 0.5 = mentioned but not integrated; 0.0 = absent)

4. coherence: Is the conclusion (action) logically supported by the stated
   reasoning?  (1.0 = fully consistent; 0.5 = minor inconsistency;
   0.0 = major contradiction)

Return ONLY valid JSON:
{"risk_management": <float>, "uncertainty": <float>,
 "regime_adaptation": <float>, "coherence": <float>}
"""


def _judge_one(decision: dict[str, Any], model: str) -> dict[str, float] | None:
    """Score a single decision's sophistication using an LLM judge."""
    action = str(decision.get("action", "hold"))
    rationale = str(decision.get("rationale", ""))
    regime = str(decision.get("regime") or "unknown")
    fill_info = ""
    if decision.get("fill"):
        fill = decision["fill"]
        fill_info = (
            f"EXECUTED: {fill.get('side')} {fill.get('quantity')} "
            f"@ {fill.get('price')}"
        )

    user_msg = (
        f"Regime: {regime}\n"
        f"Action: {action}\n"
        f"{fill_info}\n"
        f"Rationale: {rationale}"
    )

    try:
        try:
            from langfuse.openai import openai  # type: ignore[attr-defined]
        except ImportError:
            import openai

        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SOPHISTICATION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        content = str(response.choices[0].message.content or "{}")
        scores = json.loads(content)
        return {
            "risk_management": float(scores.get("risk_management", 0.0)),
            "uncertainty": float(scores.get("uncertainty", 0.0)),
            "regime_adaptation": float(scores.get("regime_adaptation", 0.0)),
            "coherence": float(scores.get("coherence", 0.0)),
        }
    except Exception as exc:
        logger.warning(
            "LLM judge failed for %s/%s: %s",
            decision.get("date"),
            decision.get("ticker"),
            exc,
        )
        return None


def compute_sophistication(
    decisions: list[dict[str, Any]],
    judge_model: str = "gpt-4.1-mini",
    max_decisions: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Score reasoning sophistication using an LLM judge (4-criterion rubric).

    Parameters
    ----------
    decisions:
        Decision dicts from decisions.jsonl.
    judge_model:
        LLM judge model ID.
    max_decisions:
        Sample cap per call (default: all).
    seed:
        Sampling seed.

    Returns
    -------
    dict with keys: ``sophistication``, ``criteria_means``, ``n_scored``,
    ``n_failed``.
    """
    import random

    valid = [
        d for d in decisions
        if d.get("action") not in ("error",) and d.get("rationale")
    ]
    if max_decisions is not None and len(valid) > max_decisions:
        rng = random.Random(seed)
        valid = rng.sample(valid, max_decisions)

    scores_list: list[dict[str, float]] = []
    n_failed = 0
    for d in valid:
        scores = _judge_one(d, judge_model)
        if scores is not None:
            scores_list.append(scores)
        else:
            n_failed += 1

    if not scores_list:
        return {
            "sophistication": 0.0,
            "criteria_means": {},
            "n_scored": 0,
            "n_failed": n_failed,
        }

    criteria = ["risk_management", "uncertainty", "regime_adaptation", "coherence"]
    criteria_means = {
        c: round(sum(s[c] for s in scores_list) / len(scores_list), 4)
        for c in criteria
    }
    overall = round(sum(criteria_means.values()) / len(criteria), 4)
    return {
        "sophistication": overall,
        "criteria_means": criteria_means,
        "n_scored": len(scores_list),
        "n_failed": n_failed,
    }


# ---------------------------------------------------------------------------
# 3b. PM multi-agent faithfulness
# ---------------------------------------------------------------------------

_SIGNAL_DIRECTION: dict[str, int] = {"bullish": 1, "neutral": 0, "bearish": -1}
_ROLE_WEIGHT_PM: dict[str, float] = {"technical": 0.55, "news": 0.25, "risk": 0.20}


def _pm_analyst_consensus(
    reports: list[dict[str, Any]],
    ticker: str,
) -> float:
    """Weighted directional consensus for one ticker from PM analyst reports.

    Returns a float in [-1, 1]:
      +1 = strongly bullish consensus
      -1 = strongly bearish consensus
       0 = neutral
    """
    score = 0.0
    for report in reports:
        if str(report.get("ticker", "")).upper() != ticker.upper():
            continue
        role = str(report.get("analyst", "technical"))
        signal = str(report.get("signal", "neutral"))
        confidence = float(report.get("confidence", 0.5))
        role_weight = _ROLE_WEIGHT_PM.get(role, 0.33)
        score += role_weight * _SIGNAL_DIRECTION.get(signal, 0) * confidence
    return max(-1.0, min(1.0, score))


def _pm_weight_direction(
    target_weight: float,
    previous_weight: float,
    tolerance: float = 0.01,
) -> int:
    """Classify PM weight change direction: +1=increase, 0=hold, -1=decrease."""
    delta = target_weight - previous_weight
    if delta > tolerance:
        return 1
    if delta < -tolerance:
        return -1
    return 0


def _pm_constrained_hold(
    *,
    ticker: str,
    target_weight: float,
    previous_weight: float,
    gross_target: float,
    portfolio_snapshot: dict[str, Any],
    consensus: float = 0.0,
    tolerance: float = 0.01,
) -> bool:
    """True if the PM's failure to adjust is explained by a portfolio constraint.

    Three PM-level constrained cases:
    A. Gross target exposure >= 95%: cash is insufficient to add more — 'constrained-full'.
    B. Previous weight for this ticker >= 40%: already highly concentrated.
    C. Previous weight = 0 and consensus is bearish: can't reduce below zero
       (structural impossibility, symmetric to single-agent's 'qty=0 sell blocked').
    """
    if _pm_weight_direction(target_weight, previous_weight, tolerance) != 0:
        return False  # PM did adjust — not a constrained hold
    nav = float(portfolio_snapshot.get("nav", 1.0)) or 1.0
    cash = float(portfolio_snapshot.get("cash", 0.0))
    cash_pct = cash / nav if nav > 0 else 0.0
    if cash_pct < 0.05:
        return True  # Case A: less than 5% cash left — can't rebalance
    if previous_weight >= 0.40:
        return True  # Case B: already highly concentrated
    # Case C: no position and bearish consensus — can't short
    return previous_weight <= tolerance and consensus < -0.10


def compute_pm_faithfulness(
    decisions: list[dict[str, Any]],
    tolerance: float = 0.01,
    _analyst_consensus_fn: Any = None,
) -> dict[str, Any]:
    """Faithfulness metric for PM multi-agent decisions.

    Measures whether each PM target-weight direction (increase / hold / decrease)
    per ticker is consistent with the analyst consensus signal for that ticker.

    This replaces the per-decision ``action=buy/sell/hold`` framework of
    single-agent faithfulness with a weight-direction framing appropriate for
    the portfolio-manager architecture.

    Faithfulness criterion (per ticker per date):
    - consensus > +0.10 AND target weight increased → faithful
    - consensus < -0.10 AND target weight decreased → faithful
    - consensus in [-0.10, +0.10] AND weight unchanged → faithful (neutral hold)
    - direction and consensus misalign → unfaithful
    - constrained hold (near-full, concentration) → constrained

    Consecutive dates are required to compute weight changes. The first
    decision is skipped (no previous weight to compare against).

    Parameters
    ----------
    decisions:
        PM JSONL decisions (must all be ``mode == "multi_agent_pm"``).
    tolerance:
        Minimum absolute weight change to count as a direction (default 1%).
    _analyst_consensus_fn:
        Test hook — callable(reports, ticker) → float.

    Returns
    -------
    dict with keys: ``pm_faithfulness``, ``n_scoreable``, ``n_faithful``,
    ``n_unfaithful``, ``n_constrained``, ``by_ticker``, ``examples_unfaithful``.
    """
    _consensus_fn = _analyst_consensus_fn or _pm_analyst_consensus

    # Exclude error fallback decisions: they are not real PM allocations.
    pm_only = [
        d for d in decisions
        if d.get("mode") == "multi_agent_pm" and not d.get("is_error", False)
    ]
    pm_only = sorted(pm_only, key=lambda d: str(d.get("date", "")))

    n_faithful = 0
    n_unfaithful = 0
    n_constrained = 0
    by_ticker: dict[str, dict[str, int]] = {}
    examples_unfaithful: list[dict[str, Any]] = []

    prev_weights: dict[str, float] = {}  # ticker → weight from previous date

    for dec in pm_only:
        date = str(dec.get("date", ""))
        targets = dec.get("targets") or {}
        weights: dict[str, float] = {
            str(t).upper(): float(w)
            for t, w in (targets.get("weights") or {}).items()
        }
        # Full ticker universe from the decision, not just tickers with non-zero weight.
        # This is critical: if the PM is all-cash, weights={} but we still need to
        # score the hold decision for each ticker vs. the analyst signals.
        universe: set[str] = {str(t).upper() for t in (dec.get("tickers") or [])}
        if not universe:
            universe = set(weights) | set(prev_weights)
        gross = sum(weights.values())
        reports = dec.get("reports") or []
        portfolio_before = dec.get("portfolio_before") or {}

        if not prev_weights:
            # First decision — no comparison possible; set baseline from universe.
            prev_weights = {ticker: weights.get(ticker, 0.0) for ticker in universe}
            continue

        for ticker in universe | set(prev_weights):
            target_weight = weights.get(ticker, 0.0)
            previous_weight = prev_weights.get(ticker, 0.0)
            direction = _pm_weight_direction(target_weight, previous_weight, tolerance)
            consensus = float(_consensus_fn(reports, ticker))

            # Direction classification
            bullish = consensus > 0.10
            bearish = consensus < -0.10

            if direction == 1 and bullish or direction == -1 and bearish:
                verdict = "faithful"
            elif direction == 0 and not bullish and not bearish:
                verdict = "faithful"  # neutral consensus → hold is rational
            elif direction == 0 and _pm_constrained_hold(
                ticker=ticker,
                target_weight=target_weight,
                previous_weight=previous_weight,
                gross_target=gross,
                portfolio_snapshot=portfolio_before,
                consensus=consensus,
                tolerance=tolerance,
            ):
                verdict = "constrained"
            else:
                verdict = "unfaithful"

            ticker_stats = by_ticker.setdefault(
                ticker, {"n_faithful": 0, "n_unfaithful": 0, "n_constrained": 0}
            )
            ticker_stats[f"n_{verdict}"] = ticker_stats.get(f"n_{verdict}", 0) + 1

            if verdict == "faithful":
                n_faithful += 1
            elif verdict == "constrained":
                n_constrained += 1
            else:
                n_unfaithful += 1
                if len(examples_unfaithful) < 10:
                    examples_unfaithful.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "consensus": round(consensus, 3),
                            "direction": direction,
                            "target_weight": round(target_weight, 4),
                            "previous_weight": round(previous_weight, 4),
                        }
                    )

        prev_weights = {ticker: weights.get(ticker, 0.0) for ticker in universe | set(prev_weights)}

    n_scoreable = n_faithful + n_unfaithful + n_constrained
    pm_faith = n_faithful / n_scoreable if n_scoreable > 0 else 0.0
    n_strict_denom = n_faithful + n_unfaithful
    pm_faith_strict = n_faithful / n_strict_denom if n_strict_denom > 0 else 0.0
    return {
        "pm_faithfulness": round(pm_faith, 4),
        "pm_faithfulness_strict": round(pm_faith_strict, 4),
        "n_scoreable": n_scoreable,
        "n_faithful": n_faithful,
        "n_unfaithful": n_unfaithful,
        "n_constrained": n_constrained,
        "by_ticker": by_ticker,
        "examples_unfaithful": examples_unfaithful,
    }


# ---------------------------------------------------------------------------
# 4. Regime segmentation
# ---------------------------------------------------------------------------


def _segment_by_regime(
    decisions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group decisions by regime label.

    Decisions with no regime (warm-up bars, insufficient price history) are
    collected under the key ``"__warmup__"`` so they are visible in the output
    but excluded from the primary regime-segmented eval tables.  This prevents
    a spurious ``"unknown"`` row that inflates faithfulness for warm-up decisions.

    Both single-agent (``decision["regime"]``) and PM (``decision["regimes"]``
    per ticker, worst-case aggregated) schemas are handled via
    ``_extract_decision_regime``.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for d in decisions:
        regime = _extract_decision_regime(d)
        key = regime if regime is not None else "__warmup__"
        groups.setdefault(key, []).append(d)
    return groups


# ---------------------------------------------------------------------------
# 5. Compute all metrics (public entry-point)
# ---------------------------------------------------------------------------


def compute_all_reasoning_metrics(
    decisions: list[dict[str, Any]],
    judge_model: str = "gpt-4.1-mini",
    max_sophistication_per_regime: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute faithfulness (LLM judge), grounding, and sophistication by regime.

    Parameters
    ----------
    decisions:
        All decision records from decisions.jsonl.
    judge_model:
        Model for both faithfulness and sophistication judges.
    max_sophistication_per_regime:
        Cap on sophistication judge calls per regime.
    seed:
        Sampling seed.

    Returns
    -------
    dict with keys:
        ``overall``, ``by_regime``, ``faithfulness_detail``,
        ``grounding_detail``, ``sophistication_detail``, ``summary_table``.
    """
    all_segments = _segment_by_regime(decisions)
    # Warm-up decisions (no regime label) are excluded from regime tables but
    # their count is surfaced in the summary for transparency.
    warmup_decisions = all_segments.pop("__warmup__", [])
    regimes = all_segments

    # Overall metrics (ALL decisions including warmup; consistent with thesis run)
    faith_overall = compute_faithfulness_llm(decisions, judge_model=judge_model, seed=seed)
    ground_overall = compute_grounding(decisions)
    soph_overall = compute_sophistication(
        decisions, judge_model=judge_model,
        max_decisions=max_sophistication_per_regime, seed=seed,
    )

    # Per-regime metrics (warm-up excluded from table)
    by_regime: dict[str, dict[str, Any]] = {}
    for regime, rdecs in sorted(regimes.items()):
        faith = compute_faithfulness_llm(rdecs, judge_model=judge_model, seed=seed)
        ground = compute_grounding(rdecs)
        soph = compute_sophistication(
            rdecs, judge_model=judge_model,
            max_decisions=max_sophistication_per_regime, seed=seed,
        )
        by_regime[regime] = {
            "n_decisions": len(rdecs),
            "faithfulness": faith["faithfulness"],
            "n_faithful": faith["n_faithful"],
            "n_unfaithful": faith["n_unfaithful"],
            "n_constrained": faith["n_constrained"],
            "grounding": ground["grounding"],
            "n_grounded": ground["n_grounded"],
            "n_claims": ground["n_claims_total"],
            "sophistication": soph["sophistication"],
            "sophistication_criteria": soph.get("criteria_means", {}),
        }

    header = [
        "regime", "n_decisions",
        "faithfulness", "n_unfaithful", "n_constrained",
        "grounding",
        "sophistication", "soph_risk", "soph_uncertainty",
        "soph_regime_adapt", "soph_coherence",
    ]
    rows = []
    for regime, m in sorted(by_regime.items()):
        crit = m.get("sophistication_criteria", {})
        rows.append([
            regime, m["n_decisions"],
            f"{m['faithfulness']:.3f}",
            m["n_unfaithful"], m["n_constrained"],
            f"{m['grounding']:.3f}",
            f"{m['sophistication']:.3f}",
            f"{crit.get('risk_management', 0.0):.3f}",
            f"{crit.get('uncertainty', 0.0):.3f}",
            f"{crit.get('regime_adaptation', 0.0):.3f}",
            f"{crit.get('coherence', 0.0):.3f}",
        ])
    crit_overall = soph_overall.get("criteria_means", {})
    rows.append([
        "OVERALL", len(decisions),
        f"{faith_overall['faithfulness']:.3f}",
        faith_overall["n_unfaithful"], faith_overall["n_constrained"],
        f"{ground_overall['grounding']:.3f}",
        f"{soph_overall['sophistication']:.3f}",
        f"{crit_overall.get('risk_management', 0.0):.3f}",
        f"{crit_overall.get('uncertainty', 0.0):.3f}",
        f"{crit_overall.get('regime_adaptation', 0.0):.3f}",
        f"{crit_overall.get('coherence', 0.0):.3f}",
    ])

    return {
        "overall": {
            "faithfulness": faith_overall["faithfulness"],
            "grounding": ground_overall["grounding"],
            "sophistication": soph_overall["sophistication"],
        },
        "by_regime": by_regime,
        "faithfulness_detail": faith_overall,
        "grounding_detail": ground_overall,
        "sophistication_detail": soph_overall,
        "summary_table": {"header": header, "rows": rows},
        "n_warmup_excluded": len(warmup_decisions),
    }
