"""Step 0 pre-registration: extract revealed sizing policy from decisions.jsonl.

Reads run gpt-4-1-mini_20260528_112645/decisions.jsonl.
Does NOT call any LLM, does NOT modify any results file.
Outputs a policy spec table + hold-decision diagnostic for user validation.

Usage:
    python scripts/step0_policy_preregistration.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_DIR = Path("runs/gpt-4-1-mini_20260528_112645")
DECISIONS_FILE = RUN_DIR / "decisions.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_portfolio(decision: dict) -> dict:
    """Return the get_portfolio dict from tool_outputs list."""
    for item in decision.get("tool_outputs", []):
        if isinstance(item, dict) and item.get("tool") == "get_portfolio":
            return item
    return {}


def _position_pct(portfolio: dict, ticker: str) -> float:
    """Return this ticker's share of NAV (0.0–1.0)."""
    nav = float(portfolio.get("nav", 0.0))
    if nav <= 0:
        return 0.0
    for pos in portfolio.get("positions", []):
        if pos.get("ticker") == ticker:
            return float(pos.get("pct_of_nav", 0.0))
    return 0.0


def _position_qty(portfolio: dict, ticker: str) -> float:
    for pos in portfolio.get("positions", []):
        if pos.get("ticker") == ticker:
            return float(pos.get("quantity", 0.0))
    return 0.0


def _cash_pct(portfolio: dict) -> float:
    nav = float(portfolio.get("nav", 1.0))
    return float(portfolio.get("cash", 0.0)) / nav if nav > 0 else 0.0


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

lines = DECISIONS_FILE.read_text(encoding="utf-8").splitlines()
decisions = [json.loads(l) for l in lines if l.strip()]
print(f"Loaded {len(decisions)} decisions\n")

# ---------------------------------------------------------------------------
# Part A — percentage mentions in rationales (sizing language)
# ---------------------------------------------------------------------------

pct_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*%")
pct_counter: Counter[str] = Counter()
has_sizing_lang = 0

for d in decisions:
    rationale = d.get("rationale", "") or ""
    matches = pct_pattern.findall(rationale)
    if matches:
        has_sizing_lang += 1
        for m in matches:
            v = float(m)
            if 5.0 <= v <= 25.0:          # only position-sizing %
                pct_counter[f"{v:.0f}%"] += 1

print("=" * 60)
print("A. PERCENTAGE MENTIONS IN RATIONALES (sizing language, 5-25%)")
print("=" * 60)
print(f"  Decisions with any sizing %%: {has_sizing_lang}/{len(decisions)}")
print()
print(f"  {'%':>6}  {'mentions':>10}")
for label, count in sorted(pct_counter.items(), key=lambda x: -x[1])[:15]:
    bar = "#" * (count // 40)
    print(f"  {label:>6}  {count:>10}  {bar}")
print()

# ---------------------------------------------------------------------------
# Part B — bucket hold decisions
# Buckets:
#   A = cash_pct < 2%       → cash-forced hold
#   B = pos_pct >= 19%      → cap-forced hold
#   C = 10% <= pos_pct < 19% → at/near-target disciplined hold
#   D = pos_pct < 10% AND cash > 2%  → genuine signal gap (potentially unfaithful)
# ---------------------------------------------------------------------------

holds = [d for d in decisions if d.get("action") == "hold"]
print("=" * 60)
print(f"B. HOLD-DECISION BUCKETING  ({len(holds)} holds)")
print("=" * 60)

bucket_counts: Counter[str] = Counter()
bucket_examples: dict[str, list[dict]] = defaultdict(list)

for d in holds:
    portfolio = _extract_portfolio(d)
    ticker = d.get("ticker", "")
    pos_pct = _position_pct(portfolio, ticker)
    cash_p  = _cash_pct(portfolio)

    if cash_p < 0.02:
        bucket = "A"   # cash-forced
    elif pos_pct >= 0.19:
        bucket = "B"   # cap-forced
    elif pos_pct >= 0.10:
        bucket = "C"   # at/near-target disciplined
    else:
        bucket = "D"   # pos < 10% AND cash > 2%  → signal gap?

    bucket_counts[bucket] += 1
    if len(bucket_examples[bucket]) < 3:
        bucket_examples[bucket].append({
            "date": d.get("date"),
            "ticker": ticker,
            "regime": d.get("regime"),
            "pos_pct": round(pos_pct * 100, 1),
            "cash_pct": round(cash_p * 100, 1),
            "rationale_snippet": (d.get("rationale") or "")[:200],
        })

bucket_labels = {
    "A": "cash-forced    (cash < 2% NAV)",
    "B": "cap-forced     (pos >= 19% NAV)",
    "C": "at-target      (10% <= pos < 19%)",
    "D": "signal gap?    (pos < 10%, cash >= 2%)",
}

print(f"\n  {'Bucket':>6}  {'N':>6}  {'%':>6}  Description")
for b in "ABCD":
    n = bucket_counts[b]
    pct_of_holds = n / len(holds) * 100
    print(f"  {b:>6}  {n:>6}  {pct_of_holds:>5.1f}%  {bucket_labels[b]}")
print()

# ---------------------------------------------------------------------------
# Part C — show 3 examples per bucket
# ---------------------------------------------------------------------------

print("=" * 60)
print("C. EXAMPLES PER BUCKET")
print("=" * 60)
for b in "ABCD":
    print(f"\n  ── Bucket {b}: {bucket_labels[b]} ──")
    for i, ex in enumerate(bucket_examples[b], 1):
        print(f"\n  [{i}] {ex['date']}  {ex['ticker']}  regime={ex['regime']}")
        print(f"      pos={ex['pos_pct']}% NAV,  cash={ex['cash_pct']}% NAV")
        print(f"      rationale: {ex['rationale_snippet']!r}")

print()

# ---------------------------------------------------------------------------
# Part D — Bucket D sub-analysis (are these really signal gaps?)
# ---------------------------------------------------------------------------

print("=" * 60)
print("D. BUCKET D DEEP-DIVE (pos < 10%, cash ≥ 2%  — potential real gaps)")
print("=" * 60)

bucket_d = []
for d in holds:
    portfolio = _extract_portfolio(d)
    ticker = d.get("ticker", "")
    pos_pct = _position_pct(portfolio, ticker)
    cash_p  = _cash_pct(portfolio)
    if pos_pct < 0.10 and cash_p >= 0.02:
        bucket_d.append({
            "d": d,
            "pos_pct": pos_pct,
            "cash_pct": cash_p,
        })

# Among bucket D, check if rationale mentions bearish/risk signals
# that could justify a hold even with low position
caution_pattern = re.compile(
    r"\b(bear|bearish|risk|volatile|uncertain|declining|downtrend|"
    r"overbought|divergen|negative|caution|weak|below|breakdown|"
    r"resistance|momentum.{0,10}negative)\b",
    re.IGNORECASE,
)

has_caution = sum(
    1 for item in bucket_d
    if caution_pattern.search(item["d"].get("rationale", "") or "")
)

print(f"\n  Bucket D total: {len(bucket_d)}")
print(f"  With bearish/caution language: {has_caution} ({has_caution/max(len(bucket_d),1)*100:.0f}%)")
print(f"  Without bearish language:      {len(bucket_d) - has_caution} — these are the clearest 'real gap' candidates")
print()

# Show 5 bucket D examples WITHOUT caution language
print("  5 Bucket D examples WITHOUT caution language:")
shown = 0
for item in bucket_d:
    d = item["d"]
    rationale = d.get("rationale", "") or ""
    if not caution_pattern.search(rationale):
        print(f"\n  [{shown+1}] {d.get('date')}  {d.get('ticker')}  regime={d.get('regime')}")
        print(f"      pos={item['pos_pct']*100:.1f}% NAV,  cash={item['cash_pct']*100:.1f}% NAV")
        print(f"      rationale: {rationale[:250]!r}")
        shown += 1
        if shown >= 5:
            break

print()

# ---------------------------------------------------------------------------
# Part E — regime breakdown of all hold decisions
# ---------------------------------------------------------------------------

print("=" * 60)
print("E. HOLD DECISIONS BY REGIME + BUCKET")
print("=" * 60)

# regime × bucket
regime_bucket: dict[str, Counter[str]] = defaultdict(Counter)
for d in holds:
    portfolio = _extract_portfolio(d)
    ticker = d.get("ticker", "")
    pos_pct = _position_pct(portfolio, ticker)
    cash_p  = _cash_pct(portfolio)
    regime  = d.get("regime", "unknown") or "unknown"

    if cash_p < 0.02:
        b = "A"
    elif pos_pct >= 0.19:
        b = "B"
    elif pos_pct >= 0.10:
        b = "C"
    else:
        b = "D"
    regime_bucket[regime][b] += 1

regimes = sorted(regime_bucket.keys())
print(f"\n  {'regime':>10}  {'A':>5}  {'B':>5}  {'C':>5}  {'D':>5}  {'total':>7}")
for r in regimes:
    cb = regime_bucket[r]
    total = sum(cb.values())
    print(f"  {r:>10}  {cb['A']:>5}  {cb['B']:>5}  {cb['C']:>5}  {cb['D']:>5}  {total:>7}")

print()

# ---------------------------------------------------------------------------
# Part F — PROPOSED POLICY SPEC (for user validation)
# ---------------------------------------------------------------------------

print("=" * 60)
print("F. PROPOSED POLICY SPEC  (pre-registration — pending user validation)")
print("=" * 60)
print("""
The agent SYSTEM_PROMPT specifies:
  • Base target:      10% of NAV  (formula: target_value = 0.10 * NAV)
  • High-conviction:  15% of NAV  (target_value = 0.15 * NAV)
  • Hard cap:         20% of NAV  (never exceed)

Coherent-hold conditions (agent is RATIONAL to hold even with bullish signals):
  [C1]  Position ≥ 10% NAV AND indicators are moderately bullish (not extreme)
  [C2]  Position ≥ 10% NAV AND cash < 2% NAV  (can't buy anyway)
  [C3]  Position ≥ 19% NAV  (at/near hard cap — buying more is structurally impossible)
  [C4]  Cash < 2% NAV  (no free capital to deploy — constraint-forced)
  [C5]  Bearish indicators OR high volatility regime → hold is rational outright

Incoherent-hold conditions (agent says hold but should buy):
  [D1]  Position < 10% NAV  AND  cash ≥ 10% NAV  AND  indicators are clearly bullish
        AND  rationale cites NO sizing, risk, or signal-quality reason
        (i.e., "real gap" — agent was given the tools to buy but didn't)

Bucket mapping:
  A = C4 (cash < 2%)               → coherent by construction
  B = C3 (pos ≥ 19%)               → coherent by construction
  C = C1/C2 (10% ≤ pos < 19%)      → coherent by construction (OLD JUDGE IS WRONG HERE)
  D = potential D1                  → needs signal check; many have caution language → C5
""")

print()
print("=" * 60)
print("SUMMARY FOR VALIDATION")
print("=" * 60)
n_coherent_by_construction = bucket_counts["A"] + bucket_counts["B"] + bucket_counts["C"]
n_d = bucket_counts["D"]
n_d_caution = has_caution
n_d_real_gap = len(bucket_d) - has_caution
print(f"""
  Total holds:                  {len(holds)}
  Coherent by construction:     {n_coherent_by_construction}  (buckets A+B+C)
    ↳ cash-forced  (A):         {bucket_counts['A']}
    ↳ cap-forced   (B):         {bucket_counts['B']}
    ↳ at-target    (C):         {bucket_counts['C']}  ← OLD JUDGE MARKS MANY AS UNFAITHFUL

  Bucket D (pos<10%, cash≥2%):  {n_d}
    ↳ with caution language:    {n_d_caution}  ← C5, rational hold
    ↳ without caution language: {n_d_real_gap}  ← true D1 candidates (potential unfaithful)

  Old judge assumption: deploy to 20% unless at cap → marks C (at-target) as unfaithful
  New judge assumption: hold at 10-15% target is coherent → only D1 is unfaithful
""")
