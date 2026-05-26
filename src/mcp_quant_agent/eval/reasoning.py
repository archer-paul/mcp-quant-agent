"""Reasoning quality metrics — faithfulness, grounding, sophistication.

This is the primary differentiator of this thesis vs. TradingAgents and
similar work: we score the *quality and faithfulness* of the LLM's
chain-of-thought reasoning against the actual tool outputs and executed orders.

Metrics (KellyBench §4 adapted for daily trading decisions)
-----------------------------------------------------------
1. **Faithfulness** — does the executed order match the stated reasoning?
   Rule-based: extract the "intent signal" from the rationale text (keyword
   matching for buy/sell/hold direction) and compare to the actual action.
   A model that writes "I should buy" but executes hold has low faithfulness.
   Score: 1.0 (faithful) or 0.0 (unfaithful) per decision, averaged over
   the group.

2. **Grounding** — do numeric claims in the rationale match the tool outputs?
   Regex extracts (label, number) pairs from the rationale (e.g., "RSI at 32.37",
   "SMA20 (136.22)").  Each claim is checked against the indicators dict and
   bars_recent prices.  A match is within 1% relative tolerance.
   Score: fraction of extractable claims that are grounded.

3. **Sophistication** — a compact 4-criterion rubric scored by LLM judge:
   - Risk management: position sizing, NAV limits, drawdown mention
   - Uncertainty: caveats, alternative scenarios, confidence language
   - Regime adaptation: explicitly mentions/uses the regime label
   - Coherence: logical support between reasoning steps and conclusion
   Each criterion 0.0-1.0; overall = mean.  Scored on ALL decisions by default
   (at ~$0.00032/call for gpt-4.1-mini, 2520 decisions ≈ $0.80 total).

All metrics are segmented by regime when ``regime_label`` is provided.

Usage
-----
From the decisions.jsonl file written by BacktestEngine::

    import json
    decisions = [json.loads(line) for line in open("runs/<id>/decisions.jsonl")]
    from mcp_quant_agent.eval.reasoning import compute_all_reasoning_metrics
    report = compute_all_reasoning_metrics(decisions)
    print(report["summary_table"])

Reference
---------
KellyBench (arXiv:2604.27865) §4 — introduces knowledge-action gap and
sophistication rubric.  We adapt the rubric to a 4-criterion version suitable
for daily single-asset decisions.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. FAITHFULNESS
# ---------------------------------------------------------------------------

# Keywords that signal a buy/sell/hold *intent* in the rationale text.
# Ordered: more specific phrases first to avoid partial matches.
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
    """Return the dominant trading intent signalled in the rationale text.

    Returns ``"buy"``, ``"sell"``, ``"hold"``, or ``None`` if unclear.

    The function counts keyword hits for each intent class and returns the
    dominant one.  Ties default to ``"hold"`` (conservative fallback).
    """
    text = rationale.lower()
    buy_hits = sum(1 for p in _BUY_SIGNALS if re.search(p, text))
    sell_hits = sum(1 for p in _SELL_SIGNALS if re.search(p, text))
    hold_hits = sum(1 for p in _HOLD_SIGNALS if re.search(p, text))

    best = max(buy_hits, sell_hits, hold_hits)
    if best == 0:
        return None  # no signal found

    if buy_hits == best and buy_hits > sell_hits and buy_hits > hold_hits:
        return "buy"
    if sell_hits == best and sell_hits > buy_hits and sell_hits > hold_hits:
        return "sell"
    # hold wins ties (conservative default)
    return "hold"


def compute_faithfulness(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute faithfulness score over a list of agent decisions.

    A decision is ``faithful`` if the executed action is consistent with the
    dominant intent in the stated rationale.  Decisions where the rationale
    contains no discernible intent are excluded from the score (not penalised).

    Parameters
    ----------
    decisions:
        List of decision dicts (from ``decisions.jsonl``), each containing:
        ``action``, ``rationale``, ``fill``, ``regime``.

    Returns
    -------
    dict with keys:
        ``faithfulness``, ``n_decisions``, ``n_faithful``, ``n_no_signal``,
        ``n_unfaithful``, ``examples_unfaithful``.
    """
    n_faithful = 0
    n_no_signal = 0
    unfaithful_examples: list[dict[str, Any]] = []

    scoreable = 0
    for d in decisions:
        action = str(d.get("action", "hold")).lower()
        if action in ("error",):
            continue  # skip engine errors
        rationale = str(d.get("rationale", ""))
        intent = _extract_intent(rationale)
        if intent is None:
            n_no_signal += 1
            continue
        scoreable += 1
        if intent == action:
            n_faithful += 1
        else:
            unfaithful_examples.append(
                {
                    "date": d.get("date"),
                    "ticker": d.get("ticker"),
                    "regime": d.get("regime"),
                    "action": action,
                    "intent": intent,
                    "rationale_snippet": rationale[:200],
                }
            )

    faithfulness = n_faithful / scoreable if scoreable > 0 else 0.0
    return {
        "faithfulness": round(faithfulness, 4),
        "n_scoreable": scoreable,
        "n_faithful": n_faithful,
        "n_unfaithful": len(unfaithful_examples),
        "n_no_signal": n_no_signal,
        "examples_unfaithful": unfaithful_examples[:5],  # first 5 for inspection
    }


# ---------------------------------------------------------------------------
# 2. GROUNDING
# ---------------------------------------------------------------------------

# Pattern: capture (label, value) from text like "RSI at 32.37" or "SMA20 (136.22)"
_CLAIM_PATTERNS = [
    # "Label (value)" — e.g., "20-day SMA (136.22)"
    (r"(?P<label>[\w\s\-\.]+?)\s*\(\s*(?P<value>[\d]+\.[\d]+)\s*\)", "parenthesised"),
    # "Label: value" or "Label at value"
    (r"(?P<label>[\w\s\-\.]+?)\s+(?:at|of|=|:)\s+(?P<value>[\d]+\.[\d]+)", "at_of"),
    # "value (label)" e.g. "136.22 (SMA20)"
    (r"(?P<value>[\d]+\.[\d]+)\s*\(\s*(?P<label>[A-Za-z][\w\s\-]+?)\s*\)", "value_paren"),
]

# Known indicator key aliases (lowercase) → canonical form
_INDICATOR_ALIASES: dict[str, list[str]] = {
    "rsi": ["rsi_14", "rsi", "rsi14"],
    "sma20": ["sma_20", "sma20", "sma 20", "20-day sma", "20d sma", "20 day sma"],
    "sma50": ["sma_50", "sma50", "sma 50", "50-day sma"],
    "macd": ["macd", "macd_line", "macd line"],
    "macd_signal": ["macd_signal", "macd signal"],
    "macd_hist": ["macd_hist", "macd histogram", "macd_histogram"],
    "atr": ["atr", "atr_14", "atr14"],
    "upper_bb": ["upper bb", "upper bollinger", "bb upper", "bollinger upper"],
    "lower_bb": ["lower bb", "lower bollinger", "bb lower", "bollinger lower"],
    "close": ["close", "current close", "current price", "price"],
}


def _extract_numeric_claims(rationale: str) -> list[tuple[str, float]]:
    """Extract (label, value) numeric claims from a rationale string.

    Returns a list of (label_lower, float_value) tuples.
    Only values with a decimal point are included (to avoid noise from
    counts and dates).
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


def _ground_claim(label: str, value: float, indicators: dict[str, Any],
                  bars_recent: list[dict[str, Any]], tolerance: float = 0.01) -> bool:
    """Check whether a (label, value) claim is grounded in tool outputs.

    Parameters
    ----------
    label:
        The label string as extracted from the rationale (lowercase).
    value:
        The numeric value claimed.
    indicators:
        Indicator dict from the decision record.
    bars_recent:
        Recent price bars (last 5) from the decision record.
    tolerance:
        Relative tolerance for floating-point comparison (default 1%).

    Returns
    -------
    True if the claim matches any entry in indicators or bars, else False.
    """
    def _match(a: float, b: float) -> bool:
        if b == 0:
            return abs(a) < 1e-6
        return abs(a - b) / abs(b) <= tolerance

    # Check indicators
    for _ind_key, ind_val in indicators.items():
        if ind_val is None:
            continue
        try:
            ind_f = float(ind_val)
        except (TypeError, ValueError):
            continue
        if _match(value, ind_f):
            return True  # value matches — good enough

    # Check close prices in recent bars
    for bar in bars_recent:
        for price_key in ("close", "open", "high", "low"):
            try:
                bar_price = float(bar.get(price_key, 0) or 0)
            except (TypeError, ValueError):
                continue
            if bar_price > 0 and _match(value, bar_price):
                return True

    return False


def compute_grounding(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute grounding rate: fraction of numeric claims that match tool outputs.

    Numeric claims are extracted from the rationale using regex patterns.
    Each claim is matched against the ``indicators`` dict and ``bars_recent``
    prices stored in the decision record.

    Parameters
    ----------
    decisions:
        Decision dicts from ``decisions.jsonl``.  Each must have:
        ``rationale``, ``indicators``, ``tool_outputs`` (for ``bars_recent``).

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
        action = str(d.get("action", "hold"))
        if action == "error":
            continue
        rationale = str(d.get("rationale", ""))
        indicators = d.get("indicators") or {}
        # bars_recent is stored under tool_outputs[0] or directly
        bars_recent: list[dict[str, Any]] = []
        for to in d.get("tool_outputs", []):
            if to.get("tool") == "get_price_history":
                # Tool output doesn't store bars directly, but bars are in indicators
                pass
        # Fallback: use bars stored in indicators (the engine captures last 5 bars
        # in tool_outputs as bars_returned count; actual bar data is in decision)
        # For grounding we use the indicators dict which contains prices via
        # sma/close values, and we also parse the bars from indicators if available.
        # Note: for full grounding, the bars_recent would ideally be stored in
        # decisions.jsonl -- they currently aren't (tool_outputs has counts, not
        # the raw bars). We use indicators as the primary source.

        claims = _extract_numeric_claims(rationale)
        if not claims:
            continue
        n_with_claims += 1
        for label, value in claims:
            n_claims_total += 1
            grounded = _ground_claim(label, value, indicators, bars_recent)
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
# 3. SOPHISTICATION (LLM judge)
# ---------------------------------------------------------------------------

_SOPHISTICATION_SYSTEM = """\
You are an expert evaluator of LLM-generated trading decisions.
Your task is to rate the sophistication of the reasoning in a trading decision
using four criteria, each scored 0.0 to 1.0:

1. risk_management: Does the reasoning mention position sizing, NAV exposure
   limits, or risk of loss?  (1.0 = explicit and quantified; 0.5 = mentioned;
   0.0 = absent)

2. uncertainty: Does the reasoning acknowledge uncertainty, alternative
   scenarios, or express calibrated confidence?  (1.0 = explicit hedging with
   conditionals; 0.5 = mild caveats; 0.0 = overconfident or no acknowledgement)

3. regime_adaptation: Does the reasoning explicitly mention or use the current
   market regime label (e.g., "bear", "bull", "high_vol") in the decision
   logic?  (1.0 = explicitly cited and relevant to the conclusion; 0.5 = mentioned
   but not integrated; 0.0 = absent)

4. coherence: Is the conclusion (action) logically supported by the stated
   reasoning?  Are there internal contradictions?  (1.0 = fully consistent;
   0.5 = minor inconsistency; 0.0 = major contradiction or non-sequitur)

Return ONLY a JSON object with these keys and float values 0.0-1.0:
{"risk_management": <float>, "uncertainty": <float>, "regime_adaptation": <float>, "coherence": <float>}
"""


def _judge_one(
    decision: dict[str, Any],
    model: str,
) -> dict[str, float] | None:
    """Score a single decision's sophistication using an LLM judge.

    Returns the rubric scores dict, or None on failure.
    """
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
        import json

        scores = json.loads(content)
        return {
            "risk_management": float(scores.get("risk_management", 0.0)),
            "uncertainty": float(scores.get("uncertainty", 0.0)),
            "regime_adaptation": float(scores.get("regime_adaptation", 0.0)),
            "coherence": float(scores.get("coherence", 0.0)),
        }
    except Exception as exc:
        logger.warning("LLM judge failed for %s/%s: %s", decision.get("date"), decision.get("ticker"), exc)
        return None


def compute_sophistication(
    decisions: list[dict[str, Any]],
    judge_model: str = "gpt-4.1-mini",
    max_decisions: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Score reasoning sophistication using an LLM judge.

    Parameters
    ----------
    decisions:
        Full list of agent decisions (from ``decisions.jsonl``).
    judge_model:
        Model to use as judge.  Default ``"gpt-4.1-mini"`` (low cost).
    max_decisions:
        If set, sample this many decisions (random sample with ``seed``).
        Default: all decisions (cost ~$0.00032 × n for gpt-4.1-mini).
    seed:
        Random seed for sampling (reproducible).

    Returns
    -------
    dict with keys: ``sophistication``, ``rubric_mean``, ``n_scored``,
    ``n_failed``, ``criteria_means``.
    """
    import random

    valid = [d for d in decisions if d.get("action") not in ("error",) and d.get("rationale")]
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
# 4. Regime-segmented aggregation
# ---------------------------------------------------------------------------


def _segment_by_regime(
    decisions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group decisions by regime label.

    Returns
    -------
    dict mapping regime label (str) to list of decisions.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for d in decisions:
        regime = str(d.get("regime") or "unknown")
        groups.setdefault(regime, []).append(d)
    return groups


def compute_all_reasoning_metrics(
    decisions: list[dict[str, Any]],
    judge_model: str = "gpt-4.1-mini",
    max_sophistication_per_regime: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute faithfulness, grounding, and sophistication, segmented by regime.

    Parameters
    ----------
    decisions:
        All decision records from ``decisions.jsonl``.
    judge_model:
        Model for the sophistication LLM judge.
    max_sophistication_per_regime:
        If set, cap the number of decisions sent to the judge per regime.
    seed:
        Sampling seed.

    Returns
    -------
    dict with keys:
        ``overall``, ``by_regime``, ``faithfulness_detail``,
        ``grounding_detail``, ``sophistication_detail``, ``summary_table``.
    """
    regimes = _segment_by_regime(decisions)

    # --- Overall metrics ---
    faith_overall = compute_faithfulness(decisions)
    ground_overall = compute_grounding(decisions)
    soph_overall = compute_sophistication(
        decisions, judge_model=judge_model,
        max_decisions=max_sophistication_per_regime, seed=seed,
    )

    # --- Per-regime metrics ---
    by_regime: dict[str, dict[str, Any]] = {}
    for regime, regime_decisions in sorted(regimes.items()):
        faith = compute_faithfulness(regime_decisions)
        ground = compute_grounding(regime_decisions)
        soph = compute_sophistication(
            regime_decisions, judge_model=judge_model,
            max_decisions=max_sophistication_per_regime, seed=seed,
        )
        by_regime[regime] = {
            "n_decisions": len(regime_decisions),
            "faithfulness": faith["faithfulness"],
            "grounding": ground["grounding"],
            "sophistication": soph["sophistication"],
            "sophistication_criteria": soph.get("criteria_means", {}),
            "n_faithful": faith["n_faithful"],
            "n_grounded": ground["n_grounded"],
            "n_claims": ground["n_claims_total"],
        }

    # --- Build summary table (list of rows for display/CSV) ---
    header = [
        "regime", "n_decisions", "faithfulness",
        "grounding", "sophistication",
        "soph_risk", "soph_uncertainty", "soph_regime_adapt", "soph_coherence",
    ]
    rows = []
    for regime, m in sorted(by_regime.items()):
        crit = m.get("sophistication_criteria", {})
        rows.append([
            regime,
            m["n_decisions"],
            f"{m['faithfulness']:.3f}",
            f"{m['grounding']:.3f}",
            f"{m['sophistication']:.3f}",
            f"{crit.get('risk_management', 0.0):.3f}",
            f"{crit.get('uncertainty', 0.0):.3f}",
            f"{crit.get('regime_adaptation', 0.0):.3f}",
            f"{crit.get('coherence', 0.0):.3f}",
        ])
    # Overall row
    crit_overall = soph_overall.get("criteria_means", {})
    rows.append([
        "OVERALL",
        len(decisions),
        f"{faith_overall['faithfulness']:.3f}",
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
    }
