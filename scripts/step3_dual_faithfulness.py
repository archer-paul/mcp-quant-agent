"""Step 3: Dual-judge faithfulness recalculation.

Computes faithfulness with BOTH judges side-by-side on the same decisions.jsonl.

  V1 judge: "max-deploy" (original baseline) — assumes rational agent deploys
            to 20% cap unless constrained; marks at-target holds as unfaithful.
  V2 judge: "policy-aware" (pre-registered 2026-05-28) — knows agent targets
            10-15% NAV; at-target holds are faithful; only capacity constraints
            (cash < 1% NAV, pos >= 19%) are "constrained".

Conditions satisfied (per user validation 2026-05-28):
  COND 1: Bucket D classified by LLM judge, not by keyword grep.
  COND 2: Bucket C = faithful (not constrained) under v2.
  COND 3: Non-trim sells (holds where judge=sell) tabulated and kept unfaithful.
  COND 4: Thresholds documented (19%=cap-adjacent; 1%=min-trade-increment).

Usage::

    python scripts/step3_dual_faithfulness.py runs/gpt-4-1-mini_20260528_112645/decisions.jsonl

Output::

    results/<run_id>/faithfulness_dual.csv     (regime × judge comparison)
    results/<run_id>/verdict_movements.csv     (per-decision verdict changes)
    results/<run_id>/non_trim_sells.csv        (hold/judge=sell cases)
    results/<run_id>/bucket_d_details.csv      (156 bucket-D cases with v2 verdict)
    Prints comparison table + bucket D analysis + non-trim sell table.

Cost (estimate at gpt-4.1-mini pricing, 2026):
    2505 decisions × ~400 token input + ~80 token output
    = ~1.2M input tokens + ~200k output tokens
    ≈ $0.18 input + $0.12 output = ~$0.30 total (v2 only; v1 cached)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python scripts/step3_dual_faithfulness.py <decisions.jsonl>")
    sys.exit(1)

jsonl_path = Path(sys.argv[1])
if not jsonl_path.exists():
    print(f"[ERROR] File not found: {jsonl_path}")
    sys.exit(1)

run_id = jsonl_path.parent.name
results_dir = Path("results") / run_id
results_dir.mkdir(parents=True, exist_ok=True)

# Inject API key
if not os.environ.get("OPENAI_API_KEY"):
    from mcp_quant_agent.config import settings
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

logging.getLogger("langfuse").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Load decisions
# ---------------------------------------------------------------------------

decisions: list[dict[str, Any]] = [
    json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()
]
print(f"Loaded {len(decisions)} decisions from {jsonl_path}")
print()

# ---------------------------------------------------------------------------
# Import eval functions
# ---------------------------------------------------------------------------

from mcp_quant_agent.eval.reasoning import (
    _extract_portfolio_snapshot,
    _segment_by_regime,
    compute_faithfulness_llm,
)

JUDGE_MODEL = "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Helper: position pct
# ---------------------------------------------------------------------------

def _pos_pct(d: dict, ticker: str) -> float:
    portfolio = _extract_portfolio_snapshot(d)
    for pos in portfolio.get("positions", []):
        if pos.get("ticker") == ticker:
            return float(pos.get("pct_of_nav", 0.0))
    return 0.0


def _cash_pct(d: dict) -> float:
    portfolio = _extract_portfolio_snapshot(d)
    nav = float(portfolio.get("nav", 1.0))
    return float(portfolio.get("cash", 0.0)) / nav if nav > 0 else 0.0


def _bucket(d: dict) -> str:
    ticker = d.get("ticker", "")
    pos = _pos_pct(d, ticker)
    cash = _cash_pct(d)
    if d.get("action") != "hold":
        return "non-hold"
    if cash < 0.01:           # < 1% NAV: cash-constrained
        return "A"
    if pos >= 0.19:           # cap-adjacent
        return "B"
    if pos >= 0.10:           # at/near target
        return "C"
    return "D"                # pos < 10%, cash >= 1%


# ---------------------------------------------------------------------------
# Compute both judges (regime-segmented)
# ---------------------------------------------------------------------------

regimes = _segment_by_regime(decisions)

print("=" * 72)
print("COMPUTING V1 JUDGE (max-deploy) — loading from cache...")
print("=" * 72)

v1_all = compute_faithfulness_llm(decisions, judge_model=JUDGE_MODEL, policy_aware=False)
v1_by_regime = {
    r: compute_faithfulness_llm(rd, judge_model=JUDGE_MODEL, policy_aware=False)
    for r, rd in sorted(regimes.items())
}
print(f"  V1 done: faithful={v1_all['n_faithful']}, unfaithful={v1_all['n_unfaithful']}, "
      f"constrained={v1_all['n_constrained']}")
print()

print("=" * 72)
print("COMPUTING V2 JUDGE (policy-aware) — may call API for uncached decisions...")
print("=" * 72)

v2_all = compute_faithfulness_llm(decisions, judge_model=JUDGE_MODEL, policy_aware=True)
v2_by_regime = {
    r: compute_faithfulness_llm(rd, judge_model=JUDGE_MODEL, policy_aware=True)
    for r, rd in sorted(regimes.items())
}
print(f"  V2 done: faithful={v2_all['n_faithful']}, unfaithful={v2_all['n_unfaithful']}, "
      f"constrained={v2_all['n_constrained']}")
print()

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def _row(regime: str, n: int, v1: dict, v2: dict) -> list:
    df = v2["faithfulness"] - v1["faithfulness"]
    ds = v2["faithfulness_strict"] - v1["faithfulness_strict"]
    return [
        regime, n,
        f"{v1['faithfulness']:.3f}", f"{v1['faithfulness_strict']:.3f}",
        v1["n_unfaithful"], v1["n_constrained"],
        f"{v2['faithfulness']:.3f}", f"{v2['faithfulness_strict']:.3f}",
        v2["n_unfaithful"], v2["n_constrained"],
        f"{df:+.3f}", f"{ds:+.3f}",
    ]

header = [
    "regime", "n",
    "v1_faith", "v1_strict", "v1_unf", "v1_con",
    "v2_faith", "v2_strict", "v2_unf", "v2_con",
    "d_faith", "d_strict",
]

rows: list[list[Any]] = []
for r in sorted(regimes.keys()):
    rows.append(_row(r, len(regimes[r]), v1_by_regime[r], v2_by_regime[r]))
rows.append(_row("OVERALL", len(decisions), v1_all, v2_all))

print("=" * 110)
print(" FAITHFULNESS: V1 (max-deploy) vs V2 (policy-aware) by regime")
print("=" * 110)
col_w = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
sep = "  ".join("-" * w for w in col_w)
print(fmt.format(*header))
print(sep)
for row in rows:
    print(fmt.format(*[str(c) for c in row]))
print()

# Save dual CSV
dual_csv = results_dir / "faithfulness_dual.csv"
with open(dual_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print(f"  Saved: {dual_csv}")

# ---------------------------------------------------------------------------
# CONDITION 1: Bucket D analysis via LLM judge
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("CONDITION 1: BUCKET D — classified by v2 LLM judge (not keyword grep)")
print("=" * 72)

# Get v2 judge actions for all decisions
v2_judge_map: dict[tuple, str | None] = {}
for ja in v2_all.get("judge_actions", []):
    key = (ja.get("date"), ja.get("ticker"))
    v2_judge_map[key] = ja.get("judge")

# Bucket D decisions with their v2 verdicts
bucket_d_decisions = [d for d in decisions if _bucket(d) == "D"]
print(f"\n  Bucket D total: {len(bucket_d_decisions)}")
print(f"  Regime breakdown:")

d_regime_counts: Counter[str] = Counter()
d_regime_verdict: dict[str, Counter[str]] = defaultdict(Counter)
for d in bucket_d_decisions:
    regime = d.get("regime", "unknown") or "unknown"
    d_regime_counts[regime] += 1
    key = (d.get("date"), d.get("ticker"))
    v2_judge = v2_judge_map.get(key)
    actual = d.get("action", "hold")
    if v2_judge == actual:
        verdict = "faithful"
    elif v2_judge is None:
        verdict = "judge_failed"
    else:
        # check constrained
        from mcp_quant_agent.eval.reasoning import _is_constrained_hold_v2
        if _is_constrained_hold_v2(d, v2_judge, actual):
            verdict = "constrained"
        else:
            verdict = "unfaithful"
    d_regime_verdict[regime][verdict] += 1

print(f"\n  {'regime':>10}  {'n':>5}  {'faithful':>9}  {'constrained':>12}  {'unfaithful':>10}")
for r in sorted(d_regime_counts.keys()):
    n = d_regime_counts[r]
    vd = d_regime_verdict[r]
    print(f"  {r:>10}  {n:>5}  {vd['faithful']:>9}  {vd['constrained']:>12}  {vd['unfaithful']:>10}")

# Collect true D1 gaps (unfaithful in bucket D)
d1_gaps = []
for d in bucket_d_decisions:
    key = (d.get("date"), d.get("ticker"))
    v2_judge = v2_judge_map.get(key)
    actual = d.get("action", "hold")
    if v2_judge is None or v2_judge == actual:
        continue
    from mcp_quant_agent.eval.reasoning import _is_constrained_hold_v2
    if not _is_constrained_hold_v2(d, v2_judge, actual):
        d1_gaps.append((d, v2_judge))

print(f"\n  >>> TRUE D1 GAPS (v2 judge predicts buy/sell, agent holds, NOT constrained): {len(d1_gaps)}")

# Show full details for each D1 gap
for i, (d, judge_pred) in enumerate(d1_gaps, 1):
    portfolio = _extract_portfolio_snapshot(d)
    nav = float(portfolio.get("nav", 1.0))
    cash = float(portfolio.get("cash", 0.0))
    ticker = d.get("ticker", "")
    pos_pct = _pos_pct(d, ticker)
    inds = d.get("indicators") or {}
    print(f"\n  --- D1 Gap #{i} ---")
    print(f"  Date={d.get('date')}  Ticker={ticker}  Regime={d.get('regime')}")
    print(f"  Action=HOLD  V2-judge predicts={judge_pred.upper()}")
    print(f"  Position={pos_pct*100:.1f}% NAV  Cash={cash:.0f} ({cash/nav*100:.1f}% NAV)  NAV={nav:.0f}")
    print(f"  Indicators: close={inds.get('close')}, rsi_14={inds.get('rsi_14')}, "
          f"sma_20={inds.get('sma_20')}, macd_hist={inds.get('macd_histogram')}")
    print(f"  Rationale (FULL): {d.get('rationale', '')}")

# Save bucket D details
d_csv = results_dir / "bucket_d_details.csv"
with open(d_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "ticker", "regime", "pos_pct", "cash_pct", "nav",
                "v2_judge", "verdict", "rationale"])
    for d in bucket_d_decisions:
        key = (d.get("date"), d.get("ticker"))
        v2_judge = v2_judge_map.get(key, "?")
        ticker = d.get("ticker", "")
        portfolio = _extract_portfolio_snapshot(d)
        nav = float(portfolio.get("nav", 1.0))
        cash = float(portfolio.get("cash", 0.0))
        actual = d.get("action", "hold")
        if v2_judge == actual:
            verdict = "faithful"
        elif v2_judge is None:
            verdict = "judge_failed"
        else:
            from mcp_quant_agent.eval.reasoning import _is_constrained_hold_v2
            verdict = "constrained" if _is_constrained_hold_v2(d, v2_judge, actual) else "unfaithful"
        w.writerow([
            d.get("date"), ticker, d.get("regime"),
            f"{_pos_pct(d, ticker)*100:.1f}", f"{_cash_pct(d)*100:.1f}", f"{nav:.0f}",
            v2_judge, verdict,
            (d.get("rationale") or "")[:300],
        ])
print(f"\n  Saved: {d_csv}")

# ---------------------------------------------------------------------------
# CONDITION 3: Non-trim sells (hold/judge=sell)
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("CONDITION 3: NON-TRIM SELLS — holds where v2 judge predicts SELL")
print("Quantifies the 'failure-to-trim' gap (unfaithful by definition)")
print("=" * 72)

non_trim: list[dict] = []
non_trim_by_regime: Counter[str] = Counter()

for ja in v2_all.get("judge_actions", []):
    if ja.get("judge") == "sell" and ja.get("actual") == "hold":
        key = (ja.get("date"), ja.get("ticker"))
        verdict = ja.get("verdict", "")
        non_trim.append({
            "date": ja.get("date"),
            "ticker": ja.get("ticker"),
            "verdict": verdict,
        })
        # Find original decision for regime + details
        for d in decisions:
            if d.get("date") == ja.get("date") and d.get("ticker") == ja.get("ticker"):
                non_trim[-1]["regime"] = d.get("regime", "unknown")
                non_trim[-1]["pos_pct"] = _pos_pct(d, d.get("ticker", "")) * 100
                non_trim[-1]["cash_pct"] = _cash_pct(d) * 100
                non_trim[-1]["rationale"] = (d.get("rationale") or "")[:200]
                non_trim[-1]["indicators"] = d.get("indicators") or {}
                break

for nt in non_trim:
    r = nt.get("regime", "unknown")
    non_trim_by_regime[r] += 1

print(f"\n  Total hold/judge=sell (non-trim gap): {len(non_trim)}")
print(f"\n  {'regime':>10}  {'count':>7}  {'unfaithful':>12}  {'constrained':>12}")
for r in sorted(non_trim_by_regime.keys()):
    count = non_trim_by_regime[r]
    unf = sum(1 for nt in non_trim if nt.get("regime") == r and nt.get("verdict") == "unfaithful")
    con = sum(1 for nt in non_trim if nt.get("regime") == r and nt.get("verdict") == "constrained")
    print(f"  {r:>10}  {count:>7}  {unf:>12}  {con:>12}")

# Show up to 5 examples
print(f"\n  Examples (up to 5 of {len(non_trim)}):")
for i, nt in enumerate(non_trim[:5], 1):
    inds = nt.get("indicators", {})
    print(f"\n  [{i}] {nt['date']}  {nt['ticker']}  regime={nt.get('regime')}  verdict={nt.get('verdict')}")
    print(f"       pos={nt.get('pos_pct', 0):.1f}% NAV  cash={nt.get('cash_pct', 0):.1f}% NAV")
    print(f"       RSI={inds.get('rsi_14')}, close={inds.get('close')}, sma20={inds.get('sma_20')}")
    print(f"       rationale: {nt.get('rationale', '')!r}")

# Save non-trim CSV
nt_csv = results_dir / "non_trim_sells.csv"
with open(nt_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "ticker", "regime", "verdict", "pos_pct", "cash_pct", "rationale"])
    for nt in non_trim:
        w.writerow([nt["date"], nt["ticker"], nt.get("regime"), nt.get("verdict"),
                    f"{nt.get('pos_pct', 0):.1f}", f"{nt.get('cash_pct', 0):.1f}",
                    nt.get("rationale", "")])
print(f"\n  Saved: {nt_csv}")

# ---------------------------------------------------------------------------
# Verdict movements: v1 → v2
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("VERDICT MOVEMENTS: V1 -> V2 (per decision)")
print("=" * 72)

# Build v1 verdict map
v1_verdict_map: dict[tuple, str] = {}
for ja in v1_all.get("judge_actions", []):
    key = (ja.get("date"), ja.get("ticker"))
    v1_verdict_map[key] = ja.get("verdict", "")

# Build v2 verdict map
v2_verdict_map: dict[tuple, str] = {}
for ja in v2_all.get("judge_actions", []):
    key = (ja.get("date"), ja.get("ticker"))
    v2_verdict_map[key] = ja.get("verdict", "")

movement_counts: Counter[tuple[str, str]] = Counter()
movements_detail: list[dict] = []

all_keys = set(v1_verdict_map.keys()) | set(v2_verdict_map.keys())
for key in all_keys:
    v1v = v1_verdict_map.get(key, "?")
    v2v = v2_verdict_map.get(key, "?")
    movement_counts[(v1v, v2v)] += 1
    if v1v != v2v:
        movements_detail.append({"date": key[0], "ticker": key[1], "v1": v1v, "v2": v2v})

print(f"\n  {'v1_verdict':>12}  {'v2_verdict':>12}  {'count':>8}  meaning")
transitions = [
    ("unfaithful", "faithful",    "RECLASSIFIED AS FAITHFUL (false positive fixed)"),
    ("unfaithful", "constrained", "v1-unfaithful now capacity-constrained"),
    ("constrained","faithful",    "capacity-constrained now fully faithful"),
    ("faithful",   "unfaithful",  "newly unfaithful (v2 stricter here)"),
    ("faithful",   "constrained", "faithful now capacity-constrained"),
    ("unfaithful", "unfaithful",  "remains unfaithful"),
    ("faithful",   "faithful",    "remains faithful"),
    ("constrained","constrained", "remains constrained"),
    ("constrained","unfaithful",  "newly unfaithful (non-trim sell, v2 stricter)"),
]
for v1v, v2v, meaning in transitions:
    n = movement_counts.get((v1v, v2v), 0)
    if n > 0:
        print(f"  {v1v:>12} -> {v2v:>12}  {n:>8}  {meaning}")

# Save verdict movements CSV
vm_csv = results_dir / "verdict_movements.csv"
with open(vm_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "ticker", "v1_verdict", "v2_verdict"])
    for m in movements_detail:
        w.writerow([m["date"], m["ticker"], m["v1"], m["v2"]])
print(f"\n  Saved: {vm_csv}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"""
  FAITHFULNESS OVERALL:
    V1 (max-deploy):    {v1_all['faithfulness']:.4f}  (strict: {v1_all['faithfulness_strict']:.4f})
    V2 (policy-aware):  {v2_all['faithfulness']:.4f}  (strict: {v2_all['faithfulness_strict']:.4f})
    Delta:              {v2_all['faithfulness'] - v1_all['faithfulness']:+.4f}  (strict: {v2_all['faithfulness_strict'] - v1_all['faithfulness_strict']:+.4f})

  VERDICT COUNTS:
    Metric            V1        V2    Delta
    faithful:    {v1_all['n_faithful']:>7}   {v2_all['n_faithful']:>7}   {v2_all['n_faithful'] - v1_all['n_faithful']:>+7}
    unfaithful:  {v1_all['n_unfaithful']:>7}   {v2_all['n_unfaithful']:>7}   {v2_all['n_unfaithful'] - v1_all['n_unfaithful']:>+7}
    constrained: {v1_all['n_constrained']:>7}   {v2_all['n_constrained']:>7}   {v2_all['n_constrained'] - v1_all['n_constrained']:>+7}
    judge_fail:  {v1_all['n_judge_failed']:>7}   {v2_all['n_judge_failed']:>7}   {v2_all['n_judge_failed'] - v1_all['n_judge_failed']:>+7}

  KEY FINDING:
    Non-trim sells (hold/judge=sell): {len(non_trim)}  (remaining unfaithful gap under v2)
    True D1 gaps (bucket D unfaithful): {len(d1_gaps)}

  OUTPUT FILES:
    {dual_csv}
    {vm_csv}
    {d_csv}
    {nt_csv}
""")
