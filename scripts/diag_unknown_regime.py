#!/usr/bin/env python3
"""Diagnostic: characterise the 'unknown' regime decisions and constrained-hold gaps."""
import json
import sys
from pathlib import Path

JSONL = Path("runs/gpt-4-1-mini_20260527_005408/decisions.jsonl")

decisions = []
with open(JSONL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            decisions.append(json.loads(line))

unknown = [d for d in decisions if d.get("regime") is None]

# ── Issue 1: what is in tool_outputs for unknown decisions ─────────────────
print("=== Issue 1: Sample unknown decisions (tool_outputs) ===")
for d in unknown[:3]:
    print(f"  Date={d['date']} Ticker={d['ticker']}")
    for to in d.get("tool_outputs", []):
        tool = to.get("tool")
        if tool == "get_price_history":
            print(f"    get_price_history: bars_count={to.get('bars_count')}, "
                  f"recent={len(to.get('bars_recent', []))}")
        elif tool == "get_current_regime":
            print(f"    regime: {to.get('regime')}")
        elif tool == "get_portfolio":
            print(f"    portfolio: nav={to.get('nav')}")
    ind = d.get("indicators", {})
    print(f"    indicators: {list(ind.keys())[:5] if ind else '(empty)'}")
    print(f"    rationale[:100]: {d.get('rationale', '')[:100]}")
    print()

# Compare with NVDA on same dates
print("=== NVDA first 3 bars (for comparison) ===")
nvda_early = [d for d in decisions if d["ticker"] == "NVDA"][:3]
for d in nvda_early:
    for to in d.get("tool_outputs", []):
        if to.get("tool") == "get_price_history":
            print(f"  {d['date']} NVDA bars_count={to.get('bars_count')}, "
                  f"regime={d.get('regime')}, ind_keys={list(d.get('indicators',{}).keys())[:3]}")
print()

# ── Confirm: are NVDA/JPM/XOM ever unknown in the same date window? ─────────
print("=== Unknown by ticker in 2022-07-01 to 2023-03-28 ===")
period = [d for d in decisions if d["date"] <= "2023-03-28"]
for tk in ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]:
    tk_decs = [d for d in period if d["ticker"] == tk]
    tk_unk = [d for d in tk_decs if d.get("regime") is None]
    print(f"  {tk}: {len(tk_decs)} decisions, {len(tk_unk)} unknown")
print()

# ── Issue 3: constrained-hold analysis ─────────────────────────────────────
print("=== Issue 3: agent=hold, judge=sell breakdown ===")
# Read faithfulness cache to get judge predictions
cache_dir = Path("runs/.faithfulness_cache")
# Build a map of (date, ticker) -> judge action from decisions we can infer
# Load reasoning metrics CSV instead
metrics_csv = Path("results/gpt-4-1-mini_20260527_005408/reasoning_metrics.csv")
if not metrics_csv.exists():
    print("  reasoning_metrics.csv not found, skipping issue 3")
    sys.exit(0)

# Load judge actions from faithfulness cache
from mcp_quant_agent.eval.reasoning import (
    _extract_portfolio_snapshot,
    _faithfulness_cache_key,
)

hold_sell_cases = []
for d in decisions:
    if d.get("action") != "hold":
        continue
    # Look up judge
    key = _faithfulness_cache_key("gpt-4.1-mini", d)
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        continue
    try:
        judge_data = json.loads(cache_file.read_text(encoding="utf-8"))
        if judge_data.get("action") == "sell":
            snap = _extract_portfolio_snapshot(d)
            ticker = d.get("ticker", "")
            pct = 0.0
            qty = 0.0
            for pos in snap.get("positions", []):
                if pos.get("ticker") == ticker:
                    pct = float(pos.get("pct_of_nav", 0.0))
                    qty = float(pos.get("quantity", 0.0))
            hold_sell_cases.append({
                "date": d["date"],
                "ticker": ticker,
                "regime": d.get("regime"),
                "pct_of_nav": pct,
                "qty": qty,
                "indicators": d.get("indicators", {}),
                "rationale": d.get("rationale", "")[:200],
            })
    except Exception:
        continue

print(f"  Total agent=hold, judge=sell: {len(hold_sell_cases)}")

# Break down by position size
buckets = {"qty==0": 0, "0<pct<18": 0, "18-25%": 0, "25-50%": 0, ">50%": 0}
for c in hold_sell_cases:
    if c["qty"] == 0:
        buckets["qty==0"] += 1
    elif c["pct_of_nav"] < 0.18:
        buckets["0<pct<18"] += 1
    elif c["pct_of_nav"] <= 0.25:
        buckets["18-25%"] += 1
    elif c["pct_of_nav"] <= 0.50:
        buckets["25-50%"] += 1
    else:
        buckets[">50%"] += 1
print(f"  Breakdown by position size (pct_of_nav):")
for k, v in buckets.items():
    print(f"    {k}: {v}")

print()
print("=== 8 representative agent=hold/judge=sell cases ===")
# Show cases spanning the position-size range
shown = 0
for c in sorted(hold_sell_cases, key=lambda x: x["pct_of_nav"]):
    if shown >= 8:
        break
    ind = c["indicators"]
    rsi = ind.get("rsi_14", "?")
    sma = ind.get("sma_20", "?")
    print(f"  {c['date']} {c['ticker']} [{c['regime']}] "
          f"pct={c['pct_of_nav']*100:.1f}% qty={c['qty']:.0f} "
          f"rsi={rsi} sma={sma}")
    print(f"    rationale: {c['rationale'][:120]}")
    shown += 1
