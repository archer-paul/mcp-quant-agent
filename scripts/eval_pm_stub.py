"""Evaluate a PM stub run: regime segmentation, PM faithfulness, grounding, memory log."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path so mcp_quant_agent is importable without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_quant_agent.eval.reasoning import (
    compute_grounding,
    compute_pm_faithfulness,
    _extract_decision_regime,
    _segment_by_regime,
)
from mcp_quant_agent.agents.pm_memory import PMDecisionLog


def main(run_id: str) -> None:
    jsonl_path = Path(f"runs/{run_id}/decisions.jsonl")
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found")
        sys.exit(1)

    decisions = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"\n=== PM STUB EVAL [{run_id}] ===")
    print(f"[!] STUB RESULTS — plumbing test only. NEVER cite in thesis.\n")
    print(f"n_decisions = {len(decisions)}")

    # --- Regime segmentation ---
    segs = _segment_by_regime(decisions)
    warmup = segs.pop("__warmup__", [])
    print(f"\n--- Regime breakdown ({len(warmup)} warm-up excluded) ---")
    for regime, decs in sorted(segs.items()):
        print(f"  {regime}: {len(decs)}")

    print("\n  Per-decision regime (PM worst-case aggregate):")
    for d in decisions:
        r = _extract_decision_regime(d)
        date = d.get("date", "?")
        regimes_raw = d.get("regimes", {})
        print(f"    {date}  aggregate={r}  per_ticker={regimes_raw}")

    # --- PM Faithfulness ---
    pm_faith = compute_pm_faithfulness(decisions)
    print(f"\n--- PM Faithfulness ---")
    print(f"  pm_faithfulness = {pm_faith['pm_faithfulness']:.3f}")
    print(f"  n_faithful    = {pm_faith['n_faithful']}")
    print(f"  n_unfaithful  = {pm_faith['n_unfaithful']}")
    print(f"  n_constrained = {pm_faith['n_constrained']}")
    print(f"  n_scoreable   = {pm_faith['n_scoreable']}")
    if pm_faith["by_ticker"]:
        print("  By ticker:")
        for ticker, stats in sorted(pm_faith["by_ticker"].items()):
            print(f"    {ticker}: {stats}")
    if pm_faith["examples_unfaithful"]:
        print("  Unfaithful examples (constrained-hold misses):")
        for ex in pm_faith["examples_unfaithful"][:5]:
            print(
                f"    {ex['date']} {ex['ticker']} "
                f"consensus={ex['consensus']:+.2f} dir={ex['direction']:+d} "
                f"tw={ex['target_weight']:.3f} pw={ex['previous_weight']:.3f}"
            )

    # --- Grounding ---
    grounding = compute_grounding(decisions)
    print(f"\n--- Grounding ---")
    print(f"  grounding     = {grounding['grounding']:.3f}")
    print(f"  n_claims      = {grounding['n_claims_total']}")
    print(f"  n_grounded    = {grounding['n_grounded']}")
    if grounding.get("ungrounded_examples"):
        print("  Ungrounded examples:")
        for ex in grounding["ungrounded_examples"][:3]:
            print(f"    {ex}")

    # --- Memory log ---
    log_path = Path(f"runs/{run_id}/pm_decision_log.md")
    if log_path.exists():
        raw = log_path.read_text(encoding="utf-8")
        pending = raw.count("| pending]")
        resolved = raw.count("| ret=")
        print(f"\n--- Memory Log (runs/{run_id}/pm_decision_log.md) ---")
        print(f"  pending entries  = {pending} (last date, awaiting J+1)")
        print(f"  resolved entries = {resolved} (J+1 returns filled in)")
        print("\n  Full log:")
        print("  " + raw.replace("\n", "\n  "))

        # Causal check via PMDecisionLog
        log = PMDecisionLog(log_path=log_path)
        log.load_from_file()
        print(f"\n  In-memory entries: {len(log._entries)}")
        for entry in log._entries:
            print(
                f"    {entry['date']} pending={entry['pending']} "
                f"weights={entry['weights']} realized={entry['realized_returns']}"
            )
    else:
        print(f"\nNo memory log found at {log_path}")

    print("\n[DONE] eval_pm_stub.py")


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "pm_stub_20260528_232402"
    main(run_id)
