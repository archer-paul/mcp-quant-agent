#!/usr/bin/env python
"""Bounded single-agent real-LLM run + full eval pipeline.

Tiers (--tier flag):
  smoke   — max 3 tickers × 5 dates (default; no --acknowledge-cost needed)
  medium  — max 3 tickers × 22 dates (~1 month; requires --acknowledge-cost)
  full    — unrestricted; requires --acknowledge-cost

Guardrails:
  - Requires --acknowledge-cost for medium/full tiers.
  - LLM cache ON by default (repeat runs cost $0 on cached prompts).
  - Transaction costs ON by default (10 bps = settings.transaction_cost_bps).
  - All results labelled with tier name ("smoke-scale", "medium-scale", etc.).

Usage examples:
  # Smoke (default tier, no cost ack needed):
  python scripts/run_bounded_eval.py --tickers AAPL --tickers MSFT --tickers NVDA \\
      --start 2023-01-03 --end 2023-01-09

  # Medium (~1 month, regime transition):
  python scripts/run_bounded_eval.py --tickers AAPL --tickers MSFT --tickers NVDA \\
      --start 2023-01-03 --end 2023-01-31 --tier medium --acknowledge-cost

  # Dry-run check only (no API calls):
  python scripts/run_bounded_eval.py --tickers AAPL --start 2023-01-03 --end 2023-01-31 \\
      --tier medium --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_APPROX_COST_PER_DECISION_USD = 0.00015  # gpt-4.1-mini ~0.15¢ per decision

app = typer.Typer(add_completion=False)


def _count_trading_dates(start: str, end: str) -> int:
    """Rough count of trading dates (weekdays only)."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    count = 0
    d = s
    while d <= e:
        if d.weekday() < 5:  # Mon-Fri
            count += 1
        d += timedelta(days=1)
    return count


def _print_eval_table(label: str, metrics: dict[str, Any]) -> None:
    print(f"\n  {label}:")
    print(f"    faithfulness       = {metrics.get('faithfulness', 0):.4f}")
    print(f"    faithfulness_strict= {metrics.get('faithfulness_strict', 0):.4f}")
    print(f"    n_faithful         = {metrics.get('n_faithful', 0)}")
    print(f"    n_unfaithful       = {metrics.get('n_unfaithful', 0)}")
    print(f"    n_constrained      = {metrics.get('n_constrained', 0)}")
    grounding = metrics.get("grounding", 0)
    n_claims = metrics.get("n_claims_total", 0)
    n_grounded = metrics.get("n_grounded", 0)
    print(f"    grounding          = {grounding:.4f}  ({n_grounded}/{n_claims} claims)")


@app.command()
def main(
    tickers: list[str] = typer.Option(..., help="Ticker universe (max per tier)."),
    start: str = typer.Option(..., help="Start date ISO-8601."),
    end: str = typer.Option(..., help="End date ISO-8601."),
    tier: str = typer.Option("smoke", help="Tier: smoke | medium | full."),
    model: str = typer.Option("gpt-4.1-mini", help="OpenAI model."),
    initial_cash: float = typer.Option(100_000.0),
    acknowledge_cost: bool = typer.Option(
        False, "--acknowledge-cost",
        help="Required for medium/full tiers.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print guardrail check only, do not call OpenAI.",
    ),
    no_cache: bool = typer.Option(
        False,
        help="Disable LLM cache. Use only for final thesis runs.",
    ),
    skip_faithfulness: bool = typer.Option(
        False,
        help="Skip faithfulness judge (saves cost if already computed).",
    ),
    label: str = typer.Option(
        "",
        help="Optional label for this run.",
    ),
    output_root: Path = typer.Option(Path("."), help="Root for runs/ and results/."),
) -> None:
    """Bounded single-agent real-LLM run + full eval pipeline."""
    from mcp_quant_agent.backtest.tier import TierName, validate_tier

    # ── Tier validation ───────────────────────────────────────────────────────
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        typer.echo("FATAL: must specify at least one ticker", err=True)
        raise typer.Exit(1)
    if tier not in ("smoke", "medium", "full"):
        typer.echo(f"FATAL: --tier must be smoke | medium | full, got {tier!r}", err=True)
        raise typer.Exit(1)
    tier_name: TierName = tier  # type: ignore[assignment]

    n_dates = _count_trading_dates(start, end)
    try:
        validate_tier(
            tier_name,
            n_tickers=len(tickers),
            n_trading_dates=n_dates,
            acknowledge_cost=acknowledge_cost or dry_run,
        )
    except ValueError as exc:
        typer.echo(f"FATAL: {exc}", err=True)
        raise typer.Exit(1) from exc

    max_decisions = len(tickers) * n_dates
    est_cost = max_decisions * _APPROX_COST_PER_DECISION_USD

    typer.echo(f"\n=== {tier.upper()}-SCALE EVAL RUN ===")
    typer.echo(f"  tickers  : {', '.join(tickers)}")
    typer.echo(f"  window   : {start} -> {end}  (~{n_dates} trading dates)")
    typer.echo(f"  tier     : {tier}")
    typer.echo(f"  model    : {model}")
    typer.echo(f"  cache    : {'ON (repeat runs free)' if not no_cache else 'OFF'}")
    typer.echo(f"  max_decisions (guard): {max_decisions}")
    typer.echo(f"  est. cost (uncached) : ${est_cost:.4f}")
    if label:
        typer.echo(f"  label    : {label}")

    if dry_run:
        typer.echo(f"\n[DRY RUN] Tier '{tier}' guardrail check passed. No API calls made.")
        return

    # validate_tier already checked acknowledge_cost for non-smoke tiers.

    # ── Run the backtest ──────────────────────────────────────────────────────
    from mcp_quant_agent.backtest.engine import BacktestEngine

    typer.echo(f"\n[1/3] Running single-agent backtest ...")
    try:
        engine = BacktestEngine(
            tickers=tickers,
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            model=model,
            use_stub=False,
            use_llm_cache=not no_cache,
        )
        results = engine.run()
    except Exception as exc:
        typer.echo(f"FATAL: backtest failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    n_decisions = results.get("n_decisions", 0)
    assert n_decisions <= max_decisions + 5, (
        f"TRIPWIRE: got {n_decisions} decisions but guard allowed max {max_decisions}. "
        "This is a bug — review the engine."
    )

    # Load decisions.jsonl
    run_id = engine.run_id
    jsonl_path = Path(output_root) / "runs" / run_id / "decisions.jsonl"
    if not jsonl_path.exists():
        typer.echo(f"FATAL: decisions.jsonl not found at {jsonl_path}", err=True)
        raise typer.Exit(1)

    decisions = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Financial metrics
    metrics = results.get("metrics", {})
    sharpe_ci = results.get("sharpe_ci", {})
    typer.echo(f"\n[FINANCIAL — {tier}-scale, NOT thesis-final]")
    typer.echo(f"  run_id         = {run_id}")
    typer.echo(f"  tier           = {tier}")
    typer.echo(f"  n_decisions    = {n_decisions}  (guard: {max_decisions})")
    typer.echo(f"  AnnReturn      = {metrics.get('annualised_return', 0)*100:+.1f}%")
    typer.echo(f"  Sharpe         = {metrics.get('sharpe', 0):.3f}")
    ci_lo = sharpe_ci.get("ci_lower", 0)
    ci_hi = sharpe_ci.get("ci_upper", 0)
    typer.echo(f"  Sharpe CI      = [{ci_lo:.3f}, {ci_hi:.3f}]")
    typer.echo(f"  MaxDD          = {metrics.get('max_drawdown', 0)*100:.1f}%")
    typer.echo(f"  cost_drag_bps  = {metrics.get('cost_drag_bps', 0):.2f}")
    typer.echo(f"  turnover_pct   = {metrics.get('turnover_pct', 0):.2f}%")

    # Action distribution
    actions = [d.get("action", "hold") for d in decisions]
    typer.echo(
        f"  actions        = buy={actions.count('buy')} sell={actions.count('sell')} "
        f"hold={actions.count('hold')} error={actions.count('error')}"
    )

    # Regime breakdown
    from mcp_quant_agent.eval.reasoning import (
        _extract_decision_regime,
        _segment_by_regime,
        compute_faithfulness_llm,
        compute_grounding,
    )

    segs = _segment_by_regime(decisions)
    warmup = segs.pop("__warmup__", [])
    typer.echo(f"\n[2/3] Regime segmentation ({len(warmup)} warm-up excluded):")
    for regime_key, rdecs in sorted(segs.items()):
        typer.echo(f"  {regime_key}: {len(rdecs)} decisions")

    # ── Eval: grounding ───────────────────────────────────────────────────────
    typer.echo("\n[3/3] Computing eval metrics ...")

    grounding_overall = compute_grounding(decisions)
    typer.echo(
        f"\n  GROUNDING (overall): {grounding_overall['grounding']:.4f}  "
        f"({grounding_overall['n_grounded']}/{grounding_overall['n_claims_total']} claims)"
    )

    # ── Eval: faithfulness (v2 policy-aware, cached) ─────────────────────────
    if not skip_faithfulness:
        faith_v2 = compute_faithfulness_llm(
            decisions,
            judge_model=model,
            policy_aware=True,
        )
        typer.echo(
            f"\n  FAITHFULNESS v2 (overall): {faith_v2['faithfulness']:.4f}  "
            f"(strict={faith_v2['faithfulness_strict']:.4f})"
        )
        typer.echo(
            f"    faithful={faith_v2['n_faithful']} "
            f"unfaithful={faith_v2['n_unfaithful']} "
            f"constrained={faith_v2['n_constrained']}"
        )

        typer.echo("\n  BY REGIME:")
        for regime_key, rdecs in sorted(segs.items()):
            if not rdecs:
                continue
            f_r = compute_faithfulness_llm(rdecs, judge_model=model, policy_aware=True)
            g_r = compute_grounding(rdecs)
            typer.echo(
                f"    {regime_key:10s}: n={len(rdecs):3d}  "
                f"faith={f_r['faithfulness']:.3f}  "
                f"ground={g_r['grounding']:.3f}"
            )

        # Unfaithful examples
        if faith_v2.get("examples_unfaithful"):
            typer.echo("\n  UNFAITHFUL EXAMPLES (first 5):")
            for ex in faith_v2["examples_unfaithful"][:5]:
                typer.echo(
                    f"    {ex['date']} {ex['ticker']}  "
                    f"judge={ex['judge_action']}  agent={ex['action']}  "
                    f"regime={ex['regime']}"
                )
    else:
        typer.echo("  [faithfulness skipped]")

    # ── Cost estimate ─────────────────────────────────────────────────────────
    typer.echo(f"\n=== COST SUMMARY ===")
    typer.echo(f"  Backtest decisions : {n_decisions}")
    typer.echo(f"  Est. backtest cost : ${n_decisions * _APPROX_COST_PER_DECISION_USD:.4f} (0 if cache hit)")
    if not skip_faithfulness:
        faith_calls = n_decisions
        typer.echo(f"  Faithfulness judge : {faith_calls} calls × ${_APPROX_COST_PER_DECISION_USD:.5f}")
        typer.echo(f"  Est. judge cost    : ${faith_calls * _APPROX_COST_PER_DECISION_USD:.4f} (0 if cache hit)")
    typer.echo(f"  NOTE: All results labelled 'smoke-scale'. Cache makes reruns free.")
    typer.echo(f"  decisions.jsonl: runs/{run_id}/decisions.jsonl")

    # ── Write run manifest ────────────────────────────────────────────────────
    manifest = {
        "run_id": run_id,
        "label": label or f"bounded-{start}-{end}",
        "tickers": tickers,
        "start": start,
        "end": end,
        "model": model,
        "n_decisions": n_decisions,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "sharpe_ci": sharpe_ci,
        "grounding_overall": grounding_overall.get("grounding", 0),
    }
    if not skip_faithfulness:
        manifest["faithfulness_v2"] = faith_v2.get("faithfulness", 0)
        manifest["faithfulness_v2_strict"] = faith_v2.get("faithfulness_strict", 0)

    manifest_path = Path(output_root) / "results" / run_id / "bounded_eval_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    typer.echo(f"\nManifest: {manifest_path}")
    typer.echo("[DONE] run_bounded_eval.py")


if __name__ == "__main__":
    app()
