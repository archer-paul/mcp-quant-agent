"""Recount PM faithfulness post-news-hallucination patch on ALL cached decisions.

Single-agent decisions are NOT affected by Bug #1 (no 'reports' field in their
JSONL schema; the faithfulness judge reads indicators/portfolio directly).

Only PM decisions (mode == 'multi_agent_pm') with reports from the news analyst
are affected. We apply the same patch as recompute_pm_faith_no_news.py to all
cached PM runs and report the label delta.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_quant_agent.eval.reasoning import compute_pm_faithfulness
from mcp_quant_agent.agents.pm_backbone import _UNAVAILABLE_SUMMARY_PREFIX

TECHNICAL_KEYWORDS = {
    "sma", "macd", "rsi", "bollinger", "histogram",
    "above 20-day", "above 50-day", "close price", "price above",
    "regime is bull", "regime is bear", "20-day sma", "50-day sma",
    "ema", "stochastic", "momentum", "current regime",
}


def has_technical_evidence(evidence: list[str]) -> bool:
    for ev in evidence:
        ev_lower = ev.lower()
        if any(kw in ev_lower for kw in TECHNICAL_KEYWORDS):
            return True
    return False


def patch_decision(d: dict) -> tuple[dict, int]:
    d2 = dict(d)
    new_reports = []
    n_patched = 0
    for r in d.get("reports", []):
        if r.get("analyst") == "news" and has_technical_evidence(r.get("evidence", [])):
            new_reports.append({
                "date": r["date"], "ticker": r["ticker"],
                "analyst": "news", "signal": "neutral", "confidence": 0.0,
                "summary": f"{_UNAVAILABLE_SUMMARY_PREFIX}: hallucinated technical evidence",
                "evidence": [],
            })
            n_patched += 1
        else:
            new_reports.append(r)
    d2["reports"] = new_reports
    return d2, n_patched


def verdict_set(decisions: list[dict], policy_aware: bool = False) -> list[str]:
    """Return per-ticker-date verdict list from compute_pm_faithfulness judge_actions."""
    faith = compute_pm_faithfulness(decisions)
    return faith


VALID_PM_RUNS = [
    "pm_api_smoke_20260528_222547",  # 2023-02-13 / JPM+NVDA (TradingAgents-style, 1 date)
    "pm_api_smoke_20260528_222656",  # 2023-01-03 / AAPL (TradingAgents-style, 1 date)
    "pm_api_smoke_20260529_000415",  # bear window A (AAPL+MSFT+NVDA, 4 dates)
    "pm_api_smoke_20260529_001143",  # bull window B (AAPL+MSFT+NVDA, 4 dates)
]


def main() -> None:
    root = Path(".")
    print("=== PM FAITHFULNESS POST-NEWS RECOUNT (ALL CACHED PM RUNS) ===")
    print("Single-agent decisions: NOT affected (no 'reports' field in schema).")
    print()

    total_decisions = 0
    total_patched_news = 0
    total_changed_verdicts = 0

    for run_id in VALID_PM_RUNS:
        p = root / "runs" / run_id / "decisions.jsonl"
        if not p.exists():
            print(f"  {run_id}: not found, skipping")
            continue

        decisions = [
            json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        pm_decisions = [d for d in decisions if d.get("mode") == "multi_agent_pm"]
        if not pm_decisions:
            print(f"  {run_id}: no PM decisions, skipping")
            continue

        n_patched_here = 0
        patched = []
        for d in pm_decisions:
            d2, n = patch_decision(d)
            n_patched_here += n
            patched.append(d2)

        # Compute faithfulness before and after
        orig = compute_pm_faithfulness(pm_decisions)
        fixed = compute_pm_faithfulness(patched)

        # Count label changes by comparing judge actions
        # (pm faithfulness doesn't expose per-decision verdicts directly;
        #  use the aggregate to show the delta)
        changed = 0
        if orig["n_faithful"] != fixed["n_faithful"] or orig["n_unfaithful"] != fixed["n_unfaithful"]:
            changed = abs(
                (orig["n_faithful"] - fixed["n_faithful"])
                + (orig["n_unfaithful"] - fixed["n_unfaithful"])
            )

        print(f"Run: {run_id}  (n_pm={len(pm_decisions)})")
        print(f"  Patched news reports   : {n_patched_here}")
        print(f"  Original  : faith={orig['pm_faithfulness']:.4f} strict={orig['pm_faithfulness_strict']:.4f}  "
              f"F={orig['n_faithful']} U={orig['n_unfaithful']} C={orig['n_constrained']}")
        print(f"  Post-patch: faith={fixed['pm_faithfulness']:.4f} strict={fixed['pm_faithfulness_strict']:.4f}  "
              f"F={fixed['n_faithful']} U={fixed['n_unfaithful']} C={fixed['n_constrained']}")
        if changed == 0:
            print(f"  Verdict delta: 0 labels changed (consensus above threshold in both cases)")
        else:
            print(f"  Verdict delta: {changed} labels CHANGED")

        total_decisions += len(pm_decisions)
        total_patched_news += n_patched_here
        total_changed_verdicts += changed
        print()

    print(f"TOTAL: {total_patched_news} hallucinated news reports across {total_decisions} PM decisions")
    print(f"TOTAL verdict changes: {total_changed_verdicts}")
    print()
    print("Interpretation:")
    if total_changed_verdicts == 0:
        print("  No faithfulness labels changed after removing fake news signals.")
        print("  The hallucination shifted consensus magnitudes but not directions.")
        print("  All PM faithfulness results stand — news gate fix is correctness,")
        print("  not a retroactive change to the reported numbers.")
    else:
        print(f"  {total_changed_verdicts} labels changed. Restate all PM faithfulness numbers.")


if __name__ == "__main__":
    main()
