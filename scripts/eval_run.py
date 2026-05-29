"""Quick eval script: run faithfulness + grounding + PM faithfulness on any decisions.jsonl.

Usage:
  python scripts/eval_run.py runs/gpt-4-1-mini_20260528_235825/decisions.jsonl
  python scripts/eval_run.py runs/pm_api_smoke_20260528_222547/decisions.jsonl --pm
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    jsonl: Path = typer.Argument(..., help="Path to decisions.jsonl"),
    model: str = typer.Option("gpt-4.1-mini", help="Judge model."),
    pm: bool = typer.Option(False, "--pm", help="Run PM faithfulness instead of single-agent."),
    policy_aware: bool = typer.Option(True, help="Use v2 policy-aware faithfulness judge."),
    skip_faithfulness: bool = typer.Option(False, help="Skip faithfulness (saves cost)."),
    label: str = typer.Option("", help="Label for this eval."),
) -> None:
    """Compute faithfulness + grounding on a decisions.jsonl file."""
    from mcp_quant_agent.eval.reasoning import (
        _extract_decision_regime,
        _segment_by_regime,
        compute_faithfulness_llm,
        compute_grounding,
        compute_mcp_time_machine_audit,
        compute_pm_evidence_grounding,
        compute_pm_faithfulness,
    )

    if not jsonl.exists():
        typer.echo(f"FATAL: {jsonl} not found", err=True)
        raise typer.Exit(1)

    decisions = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"\n=== EVAL: {jsonl.parent.name} {label} ===")
    print(f"  n_decisions = {len(decisions)}")
    print(f"  pm_mode     = {pm}")

    # Regime segmentation
    segs = _segment_by_regime(decisions)
    warmup = segs.pop("__warmup__", [])
    print(f"  n_warmup_excluded = {len(warmup)}")
    print("  Regime breakdown:")
    for regime_key, decs in sorted(segs.items()):
        print(f"    {regime_key}: {len(decs)}")

    # Per-decision regime
    for d in decisions[:3]:
        r = _extract_decision_regime(d)
        date = d.get("date", "?")
        ticker = d.get("ticker", "PORTFOLIO")
        print(f"  sample: {date} {ticker} -> regime={r}")

    # PM faithfulness
    if pm:
        from mcp_quant_agent.eval.reasoning import compute_pm_faithfulness
        pm_faith = compute_pm_faithfulness(decisions)
        print(f"\n  PM FAITHFULNESS: {pm_faith['pm_faithfulness']:.4f}")
        print(f"    strict         : {pm_faith['pm_faithfulness_strict']:.4f}")
        print(f"    n_faithful     : {pm_faith['n_faithful']}")
        print(f"    n_unfaithful   : {pm_faith['n_unfaithful']}")
        print(f"    n_constrained  : {pm_faith['n_constrained']}")
        if pm_faith["examples_unfaithful"]:
            print("    Unfaithful examples:")
            for ex in pm_faith["examples_unfaithful"][:5]:
                print(
                    f"      {ex['date']} {ex['ticker']} "
                    f"consensus={ex['consensus']:+.2f} dir={ex['direction']:+d} "
                    f"tw={ex['target_weight']:.3f} pw={ex['previous_weight']:.3f}"
                )
        if pm_faith["by_ticker"]:
            print("    By ticker:")
            for tk, stats in sorted(pm_faith["by_ticker"].items()):
                print(f"      {tk}: {stats}")

        time_audit = compute_mcp_time_machine_audit(decisions)
        print("\n  MCP TIME-MACHINE AUDIT:")
        print(
            f"    pass          : {time_audit['pass']} "
            f"({time_audit['n_violations']} violations / "
            f"{time_audit['n_timestamp_checks']} checks)"
        )
        print(f"    source warnings: {time_audit['n_live_source_warnings']}")
        if time_audit["violations"]:
            print(f"    first violation: {time_audit['violations'][0]}")

        pm_grounding = compute_pm_evidence_grounding(decisions)
        print("\n  PM EVIDENCE GROUNDING:")
        print(
            f"    grounded      : {pm_grounding['pm_evidence_grounding']:.4f} "
            f"({pm_grounding['n_grounded']}/{pm_grounding['n_checkable']} checkable)"
        )
        print(
            f"    coverage      : {pm_grounding['pm_evidence_coverage']:.4f} "
            f"({pm_grounding['n_checkable']}/{pm_grounding['n_evidence_items']} evidence items)"
        )
        if pm_grounding["examples"]:
            print(f"    first issue   : {pm_grounding['examples'][0]}")

    # Grounding
    grounding = compute_grounding(decisions)
    print(f"\n  GROUNDING: {grounding['grounding']:.4f}")
    print(f"    n_claims   = {grounding['n_claims_total']}")
    print(f"    n_grounded = {grounding['n_grounded']}")
    if grounding.get("ungrounded_examples"):
        print("    Ungrounded examples:")
        for ex in grounding["ungrounded_examples"][:3]:
            print(f"      {ex}")

    # Faithfulness
    if not skip_faithfulness and not pm:
        print(f"\n  FAITHFULNESS (v2={policy_aware}, judge={model}):")
        faith = compute_faithfulness_llm(decisions, judge_model=model, policy_aware=policy_aware)
        print(f"    overall        = {faith['faithfulness']:.4f}")
        print(f"    strict         = {faith['faithfulness_strict']:.4f}")
        print(f"    n_faithful     = {faith['n_faithful']}")
        print(f"    n_unfaithful   = {faith['n_unfaithful']}")
        print(f"    n_constrained  = {faith['n_constrained']}")
        print(f"    n_judge_failed = {faith['n_judge_failed']}")

        print("\n  BY REGIME:")
        for regime_key, rdecs in sorted(segs.items()):
            if not rdecs:
                continue
            f_r = compute_faithfulness_llm(rdecs, judge_model=model, policy_aware=policy_aware)
            g_r = compute_grounding(rdecs)
            print(
                f"    {regime_key:10s}: n={len(rdecs):3d}  "
                f"faith={f_r['faithfulness']:.4f} strict={f_r['faithfulness_strict']:.4f}  "
                f"ground={g_r['grounding']:.4f}"
            )

        if faith.get("examples_unfaithful"):
            print("\n  UNFAITHFUL EXAMPLES (first 5):")
            for ex in faith["examples_unfaithful"][:5]:
                print(
                    f"    {ex['date']} {ex['ticker']} "
                    f"judge={ex['judge_action']} agent={ex['action']} "
                    f"regime={ex['regime']}"
                )

    print("\n[DONE]")


if __name__ == "__main__":
    app()
