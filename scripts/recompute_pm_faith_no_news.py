"""Recompute PM faithfulness without the hallucinated news analyst.

Shows the quantitative impact of Bug #1 (news analyst hallucination) on the
PM faithfulness metric from the bull window B run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_quant_agent.eval.reasoning import compute_pm_faithfulness
from mcp_quant_agent.agents.pm_backbone import _UNAVAILABLE_SUMMARY_PREFIX

TECHNICAL_KEYWORDS = {
    "sma", "macd", "rsi", "bollinger", "regime", "price above", "close price",
    "histogram", "above", "ema", "momentum", "above 20-day", "above 50-day",
}


def is_fake_news_evidence(evidence: list[str]) -> bool:
    """True if any evidence item references technical indicator data."""
    for ev in evidence:
        ev_lower = str(ev).lower()
        if any(kw in ev_lower for kw in TECHNICAL_KEYWORDS):
            return True
    return False


def patch_decision(d: dict) -> dict:
    """Replace hallucinated news reports with unavailable markers."""
    d2 = dict(d)
    new_reports = []
    n_patched = 0
    for r in d.get("reports", []):
        if r.get("analyst") == "news" and is_fake_news_evidence(r.get("evidence", [])):
            new_reports.append({
                "date": r["date"],
                "ticker": r["ticker"],
                "analyst": "news",
                "signal": "neutral",
                "confidence": 0.0,
                "summary": f"{_UNAVAILABLE_SUMMARY_PREFIX}: patched (technical evidence detected)",
                "evidence": [],
            })
            n_patched += 1
        else:
            new_reports.append(r)
    d2["reports"] = new_reports
    return d2, n_patched


def main() -> None:
    run_id = "pm_api_smoke_20260529_001143"  # bull window B
    p = Path(f"runs/{run_id}/decisions.jsonl")
    decisions = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"Run: {run_id}")
    print(f"n_decisions: {len(decisions)}")
    print()

    # Check which decisions have hallucinated news
    total_patched = 0
    patched_decisions = []
    for d in decisions:
        d2, n = patch_decision(d)
        total_patched += n
        patched_decisions.append(d2)
        if n > 0:
            print(f"  {d['date']}: {n} hallucinated news report(s) patched")

    print(f"\nTotal hallucinated news reports: {total_patched}/{len(decisions)*3} possible")

    # Show consensus shift for one date
    from mcp_quant_agent.eval.reasoning import _pm_analyst_consensus
    print("\nConsensus comparison (2023-05-18, NVDA):")
    d_orig = decisions[-1]
    d_patch = patched_decisions[-1]
    orig_reports = d_orig.get("reports", [])
    patch_reports = d_patch.get("reports", [])
    c_orig = _pm_analyst_consensus(orig_reports, "NVDA")
    c_patch = _pm_analyst_consensus(patch_reports, "NVDA")
    print(f"  With hallucinated news: {c_orig:+.4f}")
    print(f"  Without news (patched): {c_patch:+.4f}")
    print(f"  Delta: {c_patch - c_orig:+.4f}")

    print()
    print("=== PM FAITHFULNESS: WITH fake news (original bug) ===")
    orig_faith = compute_pm_faithfulness(decisions)
    print(f"  overall={orig_faith['pm_faithfulness']:.4f}  strict={orig_faith['pm_faithfulness_strict']:.4f}")
    print(f"  faithful={orig_faith['n_faithful']} unfaithful={orig_faith['n_unfaithful']} constrained={orig_faith['n_constrained']}")
    if orig_faith["examples_unfaithful"]:
        for ex in orig_faith["examples_unfaithful"]:
            print(f"    {ex['date']} {ex['ticker']} consensus={ex['consensus']:+.3f} dir={ex['direction']:+d}")

    print()
    print("=== PM FAITHFULNESS: WITHOUT fake news (patched, bug fixed) ===")
    patch_faith = compute_pm_faithfulness(patched_decisions)
    print(f"  overall={patch_faith['pm_faithfulness']:.4f}  strict={patch_faith['pm_faithfulness_strict']:.4f}")
    print(f"  faithful={patch_faith['n_faithful']} unfaithful={patch_faith['n_unfaithful']} constrained={patch_faith['n_constrained']}")
    if patch_faith["examples_unfaithful"]:
        for ex in patch_faith["examples_unfaithful"]:
            print(f"    {ex['date']} {ex['ticker']} consensus={ex['consensus']:+.3f} dir={ex['direction']:+d}")
    if patch_faith["by_ticker"]:
        for tk, stats in sorted(patch_faith["by_ticker"].items()):
            print(f"    {tk}: {stats}")

    print()
    print("IMPACT: the hallucinated news analyst shifted the consensus upward by ~0.25 × fake_conf × fake_signal.")
    print("Future runs with the news gate active will not have this bias.")


if __name__ == "__main__":
    main()
