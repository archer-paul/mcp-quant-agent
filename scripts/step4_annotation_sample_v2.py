"""Step 4: Regenerate annotation sample with v2 judge columns.

Reads the v1 and v2 judge verdicts (already cached on disk from step3) and
produces a fresh annotation_sample_v2.csv + annotation_sample_v2.md for manual
human annotation.

New columns vs the original annotation_sample.csv:
  cash_available       — raw cash ($) in portfolio at decision time
  total_invested_pct   — fraction of NAV invested across all positions (%)
  feasible_buy         — can agent meaningfully buy? (cash >= 1% NAV AND pos < 19%)
  bucket               — A/B/C/D (capacity/cap/target/gap)
  verdict_old_judge    — v1 (max-deploy) verdict
  verdict_new_judge    — v2 (policy-aware) verdict
  human_predicted_action — empty for human to fill
  human_agrees_v2      — empty for human to fill (Y/N)
  notes                — empty for human to fill

Sampling strategy (same seed=42 as before):
  - Oversamples v2-unfaithful (all D1 gaps + all non-trim sells)
  - Samples from v2-faithful and v2-constrained for coverage
  - Also includes cases where v1 != v2 (verdict-change examples)
  - Target ~80 rows for tractable manual annotation

Usage::

    python scripts/step4_annotation_sample_v2.py runs/gpt-4-1-mini_20260528_112645/decisions.jsonl

Output::

    results/<run_id>/annotation_sample_v2.csv  (for annotation)
    results/<run_id>/annotation_sample_v2.md   (readable format)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python scripts/step4_annotation_sample_v2.py <decisions.jsonl>")
    sys.exit(1)

jsonl_path = Path(sys.argv[1])
run_id = jsonl_path.parent.name
results_dir = Path("results") / run_id
results_dir.mkdir(parents=True, exist_ok=True)

if not os.environ.get("OPENAI_API_KEY"):
    from mcp_quant_agent.config import settings
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

logging.getLogger("langfuse").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

decisions: list[dict[str, Any]] = [
    json.loads(l)
    for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()
]
print(f"Loaded {len(decisions)} decisions")

from mcp_quant_agent.eval.reasoning import (
    _extract_portfolio_snapshot,
    compute_faithfulness_llm,
    _is_constrained_hold_v2,
)

JUDGE_MODEL = "gpt-4.1-mini"

# ---------------------------------------------------------------------------
# Build verdict maps (uses cache — no new API calls if step3 was run)
# ---------------------------------------------------------------------------

print("Loading v1 judge verdicts from cache...")
v1_result = compute_faithfulness_llm(decisions, judge_model=JUDGE_MODEL, policy_aware=False)
print("Loading v2 judge verdicts from cache...")
v2_result = compute_faithfulness_llm(decisions, judge_model=JUDGE_MODEL, policy_aware=True)

# key = (date, ticker)
v1_map: dict[tuple, dict] = {}
for ja in v1_result.get("judge_actions", []):
    key = (ja.get("date"), ja.get("ticker"))
    v1_map[key] = ja

v2_map: dict[tuple, dict] = {}
for ja in v2_result.get("judge_actions", []):
    key = (ja.get("date"), ja.get("ticker"))
    v2_map[key] = ja

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _pos_pct(d: dict, ticker: str) -> float:
    portfolio = _extract_portfolio_snapshot(d)
    for pos in portfolio.get("positions", []):
        if pos.get("ticker") == ticker:
            return float(pos.get("pct_of_nav", 0.0))
    return 0.0


def _cash_info(d: dict) -> tuple[float, float]:
    """Return (cash_abs, cash_pct_of_nav)."""
    portfolio = _extract_portfolio_snapshot(d)
    nav = float(portfolio.get("nav", 1.0))
    cash = float(portfolio.get("cash", 0.0))
    return cash, cash / nav if nav > 0 else 0.0


def _total_invested_pct(d: dict) -> float:
    """Sum of all position pct_of_nav values."""
    portfolio = _extract_portfolio_snapshot(d)
    return sum(float(pos.get("pct_of_nav", 0.0)) for pos in portfolio.get("positions", []))


def _bucket(d: dict) -> str:
    ticker = d.get("ticker", "")
    pos = _pos_pct(d, ticker)
    _, cash_pct = _cash_info(d)
    if d.get("action") != "hold":
        return "non-hold"
    if cash_pct < 0.01:
        return "A"
    if pos >= 0.19:
        return "B"
    if pos >= 0.10:
        return "C"
    return "D"


def _feasible_buy(d: dict) -> bool:
    ticker = d.get("ticker", "")
    pos = _pos_pct(d, ticker)
    _, cash_pct = _cash_info(d)
    return cash_pct >= 0.01 and pos < 0.19


def _key_indicators(d: dict) -> str:
    inds = d.get("indicators") or {}
    parts = []
    for k in ("rsi_14", "sma_20", "macd_histogram", "atr_14", "close"):
        v = inds.get(k)
        if v is not None:
            parts.append(f"{k}={round(v, 2)}")
    return " | ".join(parts)


def _position_summary(d: dict) -> str:
    portfolio = _extract_portfolio_snapshot(d)
    ticker = d.get("ticker", "")
    nav = float(portfolio.get("nav", 1.0))
    for pos in portfolio.get("positions", []):
        if pos.get("ticker") == ticker:
            pct = float(pos.get("pct_of_nav", 0.0)) * 100
            qty = int(pos.get("quantity", 0))
            return f"{pct:.1f}% ({qty} sh)"
    return "0.0% (0 sh)"


# ---------------------------------------------------------------------------
# Assemble annotatable rows
# ---------------------------------------------------------------------------

rows: list[dict] = []

for d in decisions:
    ticker = d.get("ticker", "")
    key = (d.get("date"), ticker)

    v1_ja = v1_map.get(key, {})
    v2_ja = v2_map.get(key, {})

    v1v = v1_ja.get("verdict", "judge_failed")
    v2v = v2_ja.get("verdict", "judge_failed")
    v1_judge = v1_ja.get("judge", "?")
    v2_judge = v2_ja.get("judge", "?")

    cash_abs, cash_pct = _cash_info(d)
    portfolio = _extract_portfolio_snapshot(d)
    nav = float(portfolio.get("nav", 1.0))

    rows.append({
        "category": v2v,
        "verdict_changed": v1v != v2v,
        "date": d.get("date"),
        "ticker": ticker,
        "regime": d.get("regime", "unknown"),
        "action": d.get("action"),
        "v1_judge_predicted": v1_judge,
        "v2_judge_predicted": v2_judge,
        "verdict_old_judge": v1v,
        "verdict_new_judge": v2v,
        "cash_available": f"{cash_abs:.0f}",
        "total_invested_pct": f"{_total_invested_pct(d)*100:.1f}%",
        "feasible_buy": "yes" if _feasible_buy(d) else "no",
        "bucket": _bucket(d),
        "key_indicators": _key_indicators(d),
        "portfolio_pct_of_nav": _position_summary(d),
        "rationale": d.get("rationale", ""),
        "human_predicted_action": "",
        "human_agrees_v2": "",
        "notes": "",
    })

# ---------------------------------------------------------------------------
# Sampling strategy — target ~80 rows
# Designed for tractable manual annotation, stratified across verdict categories.
#
# Categories (with target counts):
#   [1] All D1 gaps (pos<10%, judge=buy)                    all  (~8)
#   [2] Non-trim sell sample (judge=sell, pos>0)            15 rows (stratified by regime)
#   [3] Other unfaithful (not D1, not non-trim sell)        12 rows
#   [4] verdict changed: constrained→unfaithful             10 rows (non-trim confirms)
#   [5] verdict changed: unfaithful→faithful (reclassified) 15 rows (check correctness)
#   [6] Persistent faithful (faithful in both judges)       10 rows
#   [7] Persistent unfaithful (unfaithful in both judges)   10 rows
#   Total target: ~80 rows
# ---------------------------------------------------------------------------

from collections import defaultdict

rng = random.Random(42)

# Categorise rows
d1_gaps = [r for r in rows
           if r["verdict_new_judge"] == "unfaithful"
           and r["bucket"] == "D"
           and r["v2_judge_predicted"] == "buy"]

non_trim_sells = [r for r in rows
                  if r["verdict_new_judge"] == "unfaithful"
                  and r["v2_judge_predicted"] == "sell"
                  and r["action"] == "hold"]

other_unfaithful = [r for r in rows
                    if r["verdict_new_judge"] == "unfaithful"
                    and r not in d1_gaps
                    and r not in non_trim_sells]

con_to_unf = [r for r in rows
              if r["verdict_old_judge"] == "constrained"
              and r["verdict_new_judge"] == "unfaithful"]

unf_to_faith = [r for r in rows
                if r["verdict_old_judge"] == "unfaithful"
                and r["verdict_new_judge"] == "faithful"]

persistent_faithful = [r for r in rows
                       if r["verdict_old_judge"] == "faithful"
                       and r["verdict_new_judge"] == "faithful"]

persistent_unfaithful = [r for r in rows
                         if r["verdict_old_judge"] == "unfaithful"
                         and r["verdict_new_judge"] == "unfaithful"]


def _regime_stratify(pool: list, n: int, seed_rng: random.Random) -> list:
    """Sample n items from pool, stratified by regime."""
    by_regime: dict = defaultdict(list)
    shuffled = list(pool)
    seed_rng.shuffle(shuffled)
    for r in shuffled:
        by_regime[r["regime"]].append(r)
    result = []
    per_r = max(1, n // max(len(by_regime), 1))
    for rr in by_regime.values():
        result += rr[:per_r]
    # top up from any regime if needed
    extra = [r for r in shuffled if r not in result]
    result += extra[:max(0, n - len(result))]
    return result[:n]


sample: list[dict] = []
added: set[int] = set()

def _add(items: list, max_n: int | None = None) -> None:
    subset = items if max_n is None else items[:max_n]
    for r in subset:
        if id(r) not in added:
            sample.append(r)
            added.add(id(r))

_add(d1_gaps)                                              # [1] all D1 gaps
_add(_regime_stratify(non_trim_sells, 15, rng))            # [2] non-trim sample
_add(rng.sample(other_unfaithful, min(12, len(other_unfaithful))))  # [3] other unfaithful
_add(rng.sample(con_to_unf, min(10, len(con_to_unf))))    # [4] constrained→unfaithful
_add(rng.sample(unf_to_faith, min(15, len(unf_to_faith))))  # [5] unfaithful→faithful
_add(rng.sample(persistent_faithful, min(10, len(persistent_faithful))))  # [6]
_add(rng.sample(persistent_unfaithful, min(10, len(persistent_unfaithful))))  # [7]

# Sort by date, ticker for readability
sample.sort(key=lambda r: (r["date"] or "", r["ticker"] or ""))

print(f"Sample size: {len(sample)} rows")
print(f"  v2-unfaithful: {sum(1 for r in sample if r['verdict_new_judge'] == 'unfaithful')}")
print(f"  v2-faithful:   {sum(1 for r in sample if r['verdict_new_judge'] == 'faithful')}")
print(f"  v2-constrained:{sum(1 for r in sample if r['verdict_new_judge'] == 'constrained')}")
print(f"  verdict-changed:{sum(1 for r in sample if r['verdict_changed'])}")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

csv_columns = [
    "category", "date", "ticker", "regime", "action",
    "v1_judge_predicted", "v2_judge_predicted",
    "verdict_old_judge", "verdict_new_judge",
    "cash_available", "total_invested_pct", "feasible_buy", "bucket",
    "key_indicators", "portfolio_pct_of_nav", "rationale",
    "human_predicted_action", "human_agrees_v2", "notes",
]

csv_path = results_dir / "annotation_sample_v2.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
    w.writeheader()
    w.writerows(sample)

print(f"\n  CSV  -> {csv_path}")

# ---------------------------------------------------------------------------
# Write Markdown (readable format)
# ---------------------------------------------------------------------------

md_lines = [
    "# Annotation Sample v2 (policy-aware judge)",
    "",
    "Pre-registered policy spec: base_target=10%, high_conv=15%, hard_cap=20% of NAV.",
    "",
    "**Instructions:**",
    "- `human_predicted_action`: what would YOU do with this market data? (buy/sell/hold)",
    "- `human_agrees_v2`: does the v2 judge verdict match your intuition? (Y/N/maybe)",
    "- `notes`: any nuance",
    "",
    "**Verdict semantics (v2):**",
    "- `faithful`: agent did what a rational policy-following agent would do",
    "- `constrained`: agent couldn't act — cash < 1% NAV (bucket A) or pos >= 19% (bucket B)",
    "- `unfaithful`: real gap — agent held when judge says buy/sell with no capacity reason",
    "",
    f"Sample: {len(sample)} rows (all v2-unfaithful + verdict-changes + faithful/constrained sample)",
    "",
]

for i, r in enumerate(sample, 1):
    md_lines += [
        f"---",
        f"",
        f"### [{i}] {r['date']} | {r['ticker']} | regime={r['regime']}",
        f"",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Action (agent) | **{r['action']}** |",
        f"| V1 judge predicts | {r['v1_judge_predicted']} → verdict: **{r['verdict_old_judge']}** |",
        f"| V2 judge predicts | {r['v2_judge_predicted']} → verdict: **{r['verdict_new_judge']}** |",
        f"| Verdict changed? | {'YES' if r['verdict_changed'] else 'no'} |",
        f"| Bucket | {r['bucket']} |",
        f"| Cash available | {r['cash_available']} ({r['total_invested_pct']} invested) |",
        f"| Feasible buy | {r['feasible_buy']} |",
        f"| Position | {r['portfolio_pct_of_nav']} |",
        f"| Key indicators | {r['key_indicators']} |",
        f"",
        f"**Rationale:**",
        f"> {r['rationale']}",
        f"",
        f"**Human annotation:**",
        f"- human_predicted_action: ___",
        f"- human_agrees_v2 (Y/N/maybe): ___",
        f"- notes: ___",
        f"",
    ]

md_path = results_dir / "annotation_sample_v2.md"
md_path.write_text("\n".join(md_lines), encoding="utf-8")
print(f"  MD   -> {md_path}")
print()
print("Next step: annotate the CSV/MD, then run:")
print("  python scripts/compute_agreement.py annotation_sample_v2.csv --verdict-col verdict_new_judge")
