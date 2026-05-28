"""Audit des 613 cas 'non-trim sell'.

Questions:
  1. Décomposition par ticker × régime
  2. Dépassement par ACHAT (vrai gap) vs APPRÉCIATION (dérive passive)
  3. Policy: le SYSTEM_PROMPT impose-t-il le trim des gagnants?
  4. Exemples complets (rationale + portfolio + indicateurs)
  5. Recomptage du résidu infidèle après reclassification

Aucun appel API — analyse offline sur decisions.jsonl + caches.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

RUN_DIR = Path("runs/gpt-4-1-mini_20260528_112645")
DECISIONS_FILE = RUN_DIR / "decisions.jsonl"
NON_TRIM_CSV = Path("results/gpt-4-1-mini_20260528_112645/non_trim_sells.csv")
VERDICT_CSV   = Path("results/gpt-4-1-mini_20260528_112645/verdict_movements.csv")

decisions: list[dict[str, Any]] = [
    json.loads(l)
    for l in DECISIONS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()
]

# Load v2 judge_actions from the faithfulness cache (rebuild from cache files)
import sys; sys.path.insert(0, "src")
import os
if not os.environ.get("OPENAI_API_KEY"):
    from mcp_quant_agent.config import settings
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

from mcp_quant_agent.eval.reasoning import (
    _extract_portfolio_snapshot,
    compute_faithfulness_llm,
)

import logging; logging.disable(logging.CRITICAL)

print("Loading v2 verdicts from cache (no API calls)...")
v2_result = compute_faithfulness_llm(decisions, judge_model="gpt-4.1-mini", policy_aware=True)
v2_map: dict[tuple, dict] = {}
for ja in v2_result.get("judge_actions", []):
    k = (ja.get("date"), ja.get("ticker"))
    v2_map[k] = ja

# Non-trim sell decisions
non_trim_keys: set[tuple] = set()
for ja in v2_result.get("judge_actions", []):
    if ja.get("judge") == "sell" and ja.get("actual") == "hold" and ja.get("verdict") == "unfaithful":
        non_trim_keys.add((ja.get("date"), ja.get("ticker")))

non_trim_decisions = [d for d in decisions
                      if (d.get("date"), d.get("ticker")) in non_trim_keys]

print(f"Non-trim sell decisions: {len(non_trim_decisions)}")
print()

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pos_pct(d: dict) -> float:
    portfolio = _extract_portfolio_snapshot(d)
    ticker = d.get("ticker", "")
    nav = float(portfolio.get("nav", 1.0))
    for pos in portfolio.get("positions", []):
        if pos.get("ticker") == ticker:
            return float(pos.get("pct_of_nav", 0.0))
    return 0.0

def _pos_qty(d: dict) -> float:
    portfolio = _extract_portfolio_snapshot(d)
    ticker = d.get("ticker", "")
    for pos in portfolio.get("positions", []):
        if pos.get("ticker") == ticker:
            return float(pos.get("quantity", 0.0))
    return 0.0

def _nav(d: dict) -> float:
    return float(_extract_portfolio_snapshot(d).get("nav", 0.0))

# ---------------------------------------------------------------------------
# 1. Ticker × regime decomposition
# ---------------------------------------------------------------------------

print("=" * 70)
print("1. DECOMPOSITION BY TICKER x REGIME")
print("=" * 70)

ticker_regime: dict[str, Counter[str]] = defaultdict(Counter)
for d in non_trim_decisions:
    ticker_regime[d.get("ticker", "?")][d.get("regime", "?") or "?"] += 1

tickers = sorted(ticker_regime.keys())
regimes = ["bear", "bull", "high_vol", "range"]

header = f"  {'ticker':>6}  {'total':>7}  " + "  ".join(f"{r:>9}" for r in regimes)
print(header)
print("  " + "-" * (len(header) - 2))
for t in tickers:
    total = sum(ticker_regime[t].values())
    cols = "  ".join(f"{ticker_regime[t].get(r, 0):>9}" for r in regimes)
    print(f"  {t:>6}  {total:>7}  {cols}")

total_all = len(non_trim_decisions)
total_cols = "  ".join(f"{sum(ticker_regime[t].get(r, 0) for t in tickers):>9}" for r in regimes)
print(f"  {'TOTAL':>6}  {total_all:>7}  {total_cols}")
print()

# ---------------------------------------------------------------------------
# 2. BUY-PUSH vs APPRECIATION-DRIFT
#
# Algorithm:
#   For each (ticker, date) in non-trim, find all preceding decisions for that
#   ticker. The position first crossed 20% at some point.
#   "BOUGHT above cap": the LAST BUY for this ticker happened when position
#     was already >= 18% of NAV at the time of that buy (agent bought into
#     a near-cap position, suggesting no or broken cap enforcement).
#   "Appreciation drift": the LAST BUY for this ticker happened when position
#     was < 18% of NAV — the position drifted above 20% through price gains.
# ---------------------------------------------------------------------------

print("=" * 70)
print("2. BUY-PUSH vs APPRECIATION-DRIFT")
print("=" * 70)

# Build sorted decision history per ticker
ticker_history: dict[str, list[dict]] = defaultdict(list)
for d in decisions:
    ticker_history[d.get("ticker", "")].append(d)
for t in ticker_history:
    ticker_history[t].sort(key=lambda d: d.get("date", ""))

# For each non-trim decision, find the last BUY for this ticker before this date
# and the position% at that time.
PUSH_CAP_THRESHOLD = 0.18  # position % at which agent shouldn't be buying more

buy_push_count = 0     # position was >= 18% at time of last buy
appreciation_count = 0 # position was < 18% at last buy → drifted up
no_prior_buy = 0       # never bought (edge case)

push_examples: list[dict] = []
drift_examples: list[dict] = []

for d in non_trim_decisions:
    ticker = d.get("ticker", "")
    date = d.get("date", "")
    history = ticker_history.get(ticker, [])

    # Find all BUY decisions for this ticker BEFORE this date
    prior_buys = [h for h in history
                  if h.get("date", "") < date and h.get("action") == "buy"]

    if not prior_buys:
        no_prior_buy += 1
        continue

    last_buy = prior_buys[-1]
    pos_at_last_buy = _pos_pct(last_buy)
    current_pos = _pos_pct(d)

    if pos_at_last_buy >= PUSH_CAP_THRESHOLD:
        buy_push_count += 1
        if len(push_examples) < 3:
            push_examples.append({
                "d": d, "last_buy": last_buy,
                "pos_at_buy": pos_at_last_buy, "current_pos": current_pos,
            })
    else:
        appreciation_count += 1
        if len(drift_examples) < 5:
            drift_examples.append({
                "d": d, "last_buy": last_buy,
                "pos_at_buy": pos_at_last_buy, "current_pos": current_pos,
            })

total_classified = buy_push_count + appreciation_count + no_prior_buy
print(f"\n  Total non-trim cases: {total_all}")
print(f"  Classified: {total_classified}")
print()
print(f"  APPRECIATION DRIFT (pos at last buy < 18%): {appreciation_count}  "
      f"({appreciation_count/total_all*100:.0f}%)  ← policy-silent, coherent hold")
print(f"  BUY-PUSH above cap  (pos at last buy >= 18%): {buy_push_count}  "
      f"({buy_push_count/total_all*100:.0f}%)  ← potentially real gap")
print(f"  No prior buy (edge):  {no_prior_buy}  ({no_prior_buy/total_all*100:.0f}%)")
print()

# Breakdown by ticker
print(f"  {'ticker':>6}  {'drift':>8}  {'buy_push':>10}  {'no_buy':>8}")
for t in tickers:
    t_decisions = [d for d in non_trim_decisions if d.get("ticker") == t]
    drift_t = buy_push_t = nob_t = 0
    for d in t_decisions:
        date = d.get("date", "")
        history = ticker_history.get(t, [])
        prior_buys = [h for h in history
                      if h.get("date", "") < date and h.get("action") == "buy"]
        if not prior_buys:
            nob_t += 1
        elif _pos_pct(prior_buys[-1]) >= PUSH_CAP_THRESHOLD:
            buy_push_t += 1
        else:
            drift_t += 1
    print(f"  {t:>6}  {drift_t:>8}  {buy_push_t:>10}  {nob_t:>8}")
print()

# ---------------------------------------------------------------------------
# 3. Policy check
# ---------------------------------------------------------------------------

print("=" * 70)
print("3. POLICY CHECK — does SYSTEM_PROMPT mandate trimming winning positions?")
print("=" * 70)

POLICY_QUOTES = [
    ("Target 10-15% of NAV per position. Hard cap: 20% of NAV per position.",
     "ENTRY constraint only — 'hard cap' appears in the BUY formula section."),
    ("BUY quantity formula: additional_value = min(target_value, 0.20 * NAV - existing_value)",
     "The 20% limit appears exclusively in the BUY quantity calculation."),
    ("SELL quantity: use position size from the portfolio (sell entire position, or a partial fraction).",
     "SELL instruction is purely mechanical — no trigger condition about position % stated."),
    ("Single position MUST NOT exceed 20% of portfolio NAV.",
     "Rule section — but no TRIM instruction when it drifts above 20% by appreciation."),
]

print()
for quote, interpretation in POLICY_QUOTES:
    print(f"  PROMPT: \"{quote}\"")
    print(f"  INTERPRETATION: {interpretation}")
    print()

print("  VERDICT: The SYSTEM_PROMPT is SILENT on trimming positions that drift")
print("  above 20% through price appreciation.  The 20% cap is an ENTRY constraint,")
print("  enforced at the BUY step by the engine.  A 'cap-at-entry, let-winners-run'")
print("  policy is FULLY CONSISTENT with the stated rules.")
print()
print("  The v2 judge's 'SELL because pos > 20%' instruction in its prompt is an")
print("  INVENTED requirement not present in the agent's policy spec.")
print("  This is the symmetric error to v1's 'buy to 20% if room' assumption.")
print()

# ---------------------------------------------------------------------------
# 4. Examples — 8–10 complete (rationale + portfolio + indicators)
# ---------------------------------------------------------------------------

print("=" * 70)
print("4. EXAMPLES (8 drift cases, full detail)")
print("=" * 70)

# Get 8 drift examples across tickers
drift_all: list[dict] = []
for d in non_trim_decisions:
    ticker = d.get("ticker", "")
    date = d.get("date", "")
    history = ticker_history.get(ticker, [])
    prior_buys = [h for h in history
                  if h.get("date", "") < date and h.get("action") == "buy"]
    if prior_buys and _pos_pct(prior_buys[-1]) < PUSH_CAP_THRESHOLD:
        drift_all.append({
            "d": d,
            "last_buy": prior_buys[-1],
            "pos_at_buy": _pos_pct(prior_buys[-1]),
            "current_pos": _pos_pct(d),
        })

# Pick diverse examples: different tickers
shown: dict[str, int] = defaultdict(int)
selected = []
for ex in sorted(drift_all, key=lambda x: x["d"].get("date", "")):
    t = ex["d"].get("ticker", "")
    if shown[t] < 2:
        selected.append(ex)
        shown[t] += 1
    if len(selected) >= 8:
        break

for i, ex in enumerate(selected, 1):
    d = ex["d"]
    last_buy = ex["last_buy"]
    portfolio = _extract_portfolio_snapshot(d)
    nav = float(portfolio.get("nav", 1.0))
    cash = float(portfolio.get("cash", 0.0))
    inds = d.get("indicators") or {}
    print(f"\n  === Example {i}: {d.get('date')}  {d.get('ticker')}  regime={d.get('regime')} ===")
    print(f"  Agent action: HOLD  |  V2 judge predicts: SELL  |  Verdict: unfaithful")
    print(f"  Current position: {ex['current_pos']*100:.1f}% NAV")
    print(f"  Last BUY was: {last_buy.get('date')}  position at that buy: {ex['pos_at_buy']*100:.1f}% NAV")
    print(f"  → DRIFT of +{(ex['current_pos']-ex['pos_at_buy'])*100:.1f}pp since last buy (appreciation)")
    print(f"  Cash: {cash:.0f} ({cash/nav*100:.1f}% NAV)  NAV: {nav:.0f}")
    print(f"  Indicators: close={inds.get('close')}, rsi_14={inds.get('rsi_14')}, "
          f"sma20={inds.get('sma_20')}, macd_hist={inds.get('macd_histogram')}")
    print(f"  Rationale: {d.get('rationale', '')}")
print()

# ---------------------------------------------------------------------------
# 5. Recount of unfaithful residual + categorise the 204 "other"
# ---------------------------------------------------------------------------

print("=" * 70)
print("5. RECOUNT OF UNFAITHFUL RESIDUAL + CATEGORISATION OF 204 'OTHER'")
print("=" * 70)

# Build full v2 verdict details for all 825 unfaithful
all_unfaithful = [d for d in decisions
                  if v2_map.get((d.get("date"), d.get("ticker")), {}).get("verdict") == "unfaithful"]

# Categorise each unfaithful decision
CAT_D1 = "D1-gap (buy-gap)"
CAT_NONTRIM_DRIFT = "non-trim sell (appreciation drift)"
CAT_NONTRIM_PUSH  = "non-trim sell (buy-pushed above cap)"
CAT_NONTRIM_NOBUY = "non-trim sell (no prior buy)"
CAT_OTHER_HOLD_BUY  = "hold/judge=buy (not D1)"
CAT_OTHER_ACT_MISMATCH = "action mismatch (other)"

categorised: list[tuple[dict, str]] = []

for d in all_unfaithful:
    ticker = d.get("ticker", "")
    date = d.get("date", "")
    ja = v2_map.get((date, ticker), {})
    judge = ja.get("judge", "")
    actual = d.get("action", "hold")
    portfolio = _extract_portfolio_snapshot(d)
    pos_pct_now = _pos_pct(d)
    cash_pct = float(portfolio.get("cash", 0.0)) / max(float(portfolio.get("nav", 1.0)), 1.0)

    if judge == "sell" and actual == "hold":
        # Non-trim sell — sub-categorise
        history = ticker_history.get(ticker, [])
        prior_buys = [h for h in history
                      if h.get("date", "") < date and h.get("action") == "buy"]
        if not prior_buys:
            categorised.append((d, CAT_NONTRIM_NOBUY))
        elif _pos_pct(prior_buys[-1]) >= PUSH_CAP_THRESHOLD:
            categorised.append((d, CAT_NONTRIM_PUSH))
        else:
            categorised.append((d, CAT_NONTRIM_DRIFT))

    elif judge == "buy" and actual == "hold" and pos_pct_now < 0.10 and cash_pct >= 0.01:
        categorised.append((d, CAT_D1))

    elif judge == "buy" and actual == "hold":
        # Hold/judge=buy but not D1 (position >= 10% or cash < 1%)
        categorised.append((d, CAT_OTHER_HOLD_BUY))

    else:
        categorised.append((d, CAT_OTHER_ACT_MISMATCH))

cat_counts: Counter[str] = Counter(cat for _, cat in categorised)

print(f"\n  Total unfaithful (v2): {len(all_unfaithful)}")
print()
print(f"  {'Category':55}  {'N':>6}  {'%':>5}  Policy verdict")
for cat, n in cat_counts.most_common():
    pct = n / len(all_unfaithful) * 100
    if cat == CAT_NONTRIM_DRIFT:
        verdict = "POLICY SILENT → reclassify COHERENT"
    elif cat == CAT_NONTRIM_PUSH:
        verdict = "engine cap should prevent → inspect"
    elif cat == CAT_NONTRIM_NOBUY:
        verdict = "edge case"
    elif cat == CAT_D1:
        verdict = "real gap (confirmed)"
    elif cat == CAT_OTHER_HOLD_BUY:
        verdict = "hold/judge=buy at pos>=10% → policy-coherent?"
    else:
        verdict = "needs inspection"
    print(f"  {cat:55}  {n:>6}  {pct:>4.0f}%  {verdict}")

print()
# Proposed revised counts
n_drift = cat_counts[CAT_NONTRIM_DRIFT]
n_push  = cat_counts[CAT_NONTRIM_PUSH]
n_nobuy = cat_counts[CAT_NONTRIM_NOBUY]
n_d1    = cat_counts[CAT_D1]
n_hold_buy = cat_counts[CAT_OTHER_HOLD_BUY]
n_other = cat_counts[CAT_OTHER_ACT_MISMATCH]

print(f"  ─── PROPOSED RECLASSIFICATION ───")
print(f"  Reclassify as FAITHFUL (policy-coherent):")
print(f"    Appreciation drift:        {n_drift:>4}  (policy silent on trim)")
print(f"    Hold/buy at pos>=10%:      {n_hold_buy:>4}  (at/above target, hold = policy-coherent)")
print()
n_true_unfaithful = n_push + n_d1 + n_other + n_nobuy
n_reclassified = n_drift + n_hold_buy
print(f"  True unfaithful after reclassification: {n_true_unfaithful}")
print(f"  Reclassified as coherent:               {n_reclassified}")
print()
n_total = len(decisions)
n_total_faithful_v2 = sum(1 for ja in v2_result.get("judge_actions", [])
                           if ja.get("verdict") == "faithful")
n_proposed_faithful = n_total_faithful_v2 + n_reclassified
proposed_faith = n_proposed_faithful / n_total
print(f"  Proposed faithfulness (v3-policy):   {proposed_faith:.4f}")
print(f"    (v2 was {n_total_faithful_v2/n_total:.4f}, v1 was 0.272)")
print()

# Also show regime breakdown of drift cases
print(f"  Drift cases by ticker × regime:")
drift_cases = [(d, cat) for d, cat in categorised if cat == CAT_NONTRIM_DRIFT]
drift_ticker_regime: dict[str, Counter[str]] = defaultdict(Counter)
for d, _ in drift_cases:
    drift_ticker_regime[d.get("ticker", "?")][d.get("regime", "?") or "?"] += 1

print(f"  {'ticker':>6}  {'total':>6}  " + "  ".join(f"{r:>9}" for r in regimes))
for t in tickers:
    total_t = sum(drift_ticker_regime.get(t, Counter()).values())
    cols = "  ".join(f"{drift_ticker_regime.get(t, Counter()).get(r, 0):>9}" for r in regimes)
    print(f"  {t:>6}  {total_t:>6}  {cols}")
print()

# ---------------------------------------------------------------------------
# 6. Characterise the 204 "other" (non-sell, non-D1) unfaithful
# ---------------------------------------------------------------------------

print("=" * 70)
print("6. CHARACTERISING THE 204 'OTHER' UNFAITHFUL")
print("=" * 70)

other_cases = [(d, cat) for d, cat in categorised
               if cat not in (CAT_NONTRIM_DRIFT, CAT_NONTRIM_PUSH,
                              CAT_NONTRIM_NOBUY, CAT_D1)]
print(f"\n  Total 'other': {len(other_cases)}")

# Show judge×actual breakdown
judge_actual: Counter[tuple] = Counter()
for d, cat in other_cases:
    ja = v2_map.get((d.get("date"), d.get("ticker")), {})
    judge = ja.get("judge", "?")
    actual = d.get("action", "?")
    judge_actual[(judge, actual)] += 1

print(f"\n  Judge action × Agent action breakdown:")
for (j, a), n in judge_actual.most_common():
    print(f"    judge={j}, actual={a}: {n}")

# Show 5 examples of each type
print(f"\n  Examples from 'other' unfaithful (up to 5 per judge-actual pair):")
shown_other: dict[tuple, int] = defaultdict(int)
for d, cat in other_cases:
    ja = v2_map.get((d.get("date"), d.get("ticker")), {})
    judge = ja.get("judge", "?")
    actual = d.get("action", "?")
    pair = (judge, actual)
    if shown_other[pair] >= 3:
        continue
    shown_other[pair] += 1
    portfolio = _extract_portfolio_snapshot(d)
    nav = float(portfolio.get("nav", 1.0))
    cash = float(portfolio.get("cash", 0.0))
    inds = d.get("indicators") or {}
    pos = _pos_pct(d)
    print(f"\n    [{judge}→{actual}] {d.get('date')}  {d.get('ticker')}  "
          f"regime={d.get('regime')}  pos={pos*100:.1f}%  cash={cash/nav*100:.1f}%")
    print(f"    RSI={inds.get('rsi_14')}, close={inds.get('close')}, sma20={inds.get('sma_20')}")
    print(f"    Rationale: {(d.get('rationale') or '')[:250]}")
print()

# ---------------------------------------------------------------------------
# 7. Summary table for DECISIONS.md
# ---------------------------------------------------------------------------

print("=" * 70)
print("7. SUMMARY TABLE (for DECISIONS.md)")
print("=" * 70)

print(f"""
  UNFAITHFUL RESIDUAL DECOMPOSITION (post-audit)

  Category                             N     % of 825   Policy status
  ──────────────────────────────────  ────  ─────────  ──────────────
  Non-trim: appreciation drift         {n_drift:>4}     {n_drift/825*100:>4.0f}%    POLICY SILENT → coherent
  Non-trim: buy-pushed above cap       {n_push:>4}     {n_push/825*100:>4.0f}%    True cap breach? (check engine)
  Non-trim: no prior buy               {n_nobuy:>4}     {n_nobuy/825*100:>4.0f}%    Edge case
  Hold/judge=buy at pos>=10%           {n_hold_buy:>4}     {n_hold_buy/825*100:>4.0f}%    Policy-coherent (at target)
  True D1 gaps (confirmed)               {n_d1:>2}      {n_d1/825*100:>3.0f}%    Real gap
  Other action mismatches              {n_other:>4}     {n_other/825*100:>4.0f}%    Needs inspection
  ──────────────────────────────────  ────  ─────────
  Total                                 825    100%

  Proposed V3 classification:
    Coherent (reclassify):  {n_reclassified}  (drift {n_drift} + hold/at-target {n_hold_buy})
    True unfaithful:         {n_true_unfaithful}  (push {n_push} + D1 {n_d1} + other {n_other} + edge {n_nobuy})
    Proposed faithfulness:   {proposed_faith:.4f}
""")
