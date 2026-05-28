#!/usr/bin/env python
"""Bounded multi-day PM multi-agent real run + eval pipeline.

Guardrails:
  - Max 10 trading dates, max 3 tickers.
  - Dev model only, LLM cache on, --acknowledge-cost required.
  - Runs eval (grounding + PM faithfulness) on the output JSONL.
  - Labels results 'smoke-scale, not thesis-final'.

Usage:
  # Bear window A (4 dates, 3 tickers):
  python scripts/run_pm_multiday.py --tickers AAPL --tickers MSFT --tickers NVDA \\
      --start 2023-01-03 --end 2023-01-09 --acknowledge-cost --allow-empty-news

  # Dry-run check:
  python scripts/run_pm_multiday.py --tickers AAPL --start 2023-01-03 --end 2023-01-09 --dry-run
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Cost: 11 LLM calls per PM decision date (TradingAgents chain)
_APPROX_COST_PER_PM_DATE_USD = 11 * 0.00015

app = typer.Typer(add_completion=False)


@app.command()
def main(
    tickers: list[str] = typer.Option(..., help="Ticker universe (max 3)."),
    start: str = typer.Option(..., help="Start date ISO-8601."),
    end: str = typer.Option(..., help="End date ISO-8601."),
    model: str = typer.Option(None, help="OpenAI model (defaults to settings.agent_model_dev)."),
    initial_cash: float = typer.Option(100_000.0),
    acknowledge_cost: bool = typer.Option(
        False, "--acknowledge-cost",
        help="Required for real LLM runs.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Guardrail check only, no API calls."),
    allow_empty_news: bool = typer.Option(False, help="Allow missing news cache."),
    label: str = typer.Option("", help="Optional label for this run."),
    output_root: Path = typer.Option(Path("."), help="Root for runs/ and results/."),
) -> None:
    """Bounded multi-day PM multi-agent real run + eval pipeline."""
    from mcp_quant_agent.backtest.pm_smoke_guard import validate_pm_multiday_request
    from mcp_quant_agent.config import settings

    chosen_model = model or settings.agent_model_dev

    # ── Validate guardrails ───────────────────────────────────────────────────
    try:
        guard = validate_pm_multiday_request(
            tickers=tickers,
            start_date=start,
            end_date=end,
            model=chosen_model,
            dev_model=settings.agent_model_dev,
            use_llm_cache=True,
            acknowledge_cost=acknowledge_cost or dry_run,
        )
    except RuntimeError as exc:
        typer.echo(f"FATAL guardrail: {exc}", err=True)
        raise typer.Exit(1) from exc

    est_cost = guard.n_trading_dates * _APPROX_COST_PER_PM_DATE_USD
    typer.echo("\n=== PM MULTI-DAY BOUNDED RUN ===")
    typer.echo(f"  tickers   : {', '.join(guard.tickers)}")
    typer.echo(f"  window    : {start} -> {end}  (~{guard.n_trading_dates} trading dates)")
    typer.echo(f"  model     : {chosen_model}")
    typer.echo(f"  est. PM calls       : {guard.estimated_pm_calls}")
    typer.echo(f"  est. LLM calls      : {guard.estimated_llm_calls}")
    typer.echo(f"  est. cost (uncached): ${est_cost:.4f}")
    if label:
        typer.echo(f"  label     : {label}")

    if dry_run:
        typer.echo("\n[DRY RUN] Guardrail check passed. No API calls made.")
        return

    if not acknowledge_cost:
        typer.echo("FATAL: pass --acknowledge-cost to confirm spend.", err=True)
        raise typer.Exit(1)

    # ── Run the PM backtest ───────────────────────────────────────────────────
    from mcp_quant_agent.backtest.pm_engine import PMBacktestEngine

    typer.echo("\n[1/3] Running PM multi-agent backtest ...")
    try:
        engine = PMBacktestEngine(
            tickers=list(guard.tickers),
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            use_stub=False,
            model=chosen_model,
            use_llm_cache=True,
            smoke_guard=guard,
            allow_empty_news=allow_empty_news,
            write_artifacts=True,
            output_root=output_root,
        )
        results = engine.run()
    except Exception as exc:
        typer.echo(f"FATAL: PM backtest failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    run_id = engine.run_id
    n_decisions = results.get("n_decisions", 0)
    metrics = results.get("metrics", {})
    sharpe_ci = results.get("sharpe_ci", {})

    typer.echo(f"\n[FINANCIAL - smoke-scale, NOT thesis-final]")
    typer.echo(f"  run_id      = {run_id}")
    typer.echo(f"  n_decisions = {n_decisions}")
    typer.echo(f"  AnnReturn   = {metrics.get('annualised_return', 0)*100:+.1f}%")
    typer.echo(f"  Sharpe      = {metrics.get('sharpe', 0):.3f}")
    typer.echo(f"  MaxDD       = {metrics.get('max_drawdown', 0)*100:.1f}%")

    # Load decisions
    jsonl_path = Path(output_root) / "runs" / run_id / "decisions.jsonl"
    if not jsonl_path.exists():
        typer.echo(f"FATAL: decisions.jsonl not found at {jsonl_path}", err=True)
        raise typer.Exit(1)

    decisions = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # ── Eval ─────────────────────────────────────────────────────────────────
    from mcp_quant_agent.eval.reasoning import (
        _segment_by_regime,
        compute_grounding,
        compute_pm_faithfulness,
    )

    typer.echo("\n[2/3] Regime segmentation:")
    segs = _segment_by_regime(decisions)
    warmup = segs.pop("__warmup__", [])
    typer.echo(f"  n_warmup_excluded = {len(warmup)}")
    for regime_key, decs in sorted(segs.items()):
        typer.echo(f"  {regime_key}: {len(decs)}")

    typer.echo("\n[3/3] Eval metrics:")

    # Grounding
    grounding = compute_grounding(decisions)
    typer.echo(
        f"  GROUNDING (overall) : {grounding['grounding']:.4f}  "
        f"({grounding['n_grounded']}/{grounding['n_claims_total']} claims)"
    )
    for regime_key, rdecs in sorted(segs.items()):
        g_r = compute_grounding(rdecs)
        typer.echo(
            f"    {regime_key}: n={len(rdecs)} ground={g_r['grounding']:.4f}"
        )

    # PM Faithfulness
    pm_faith = compute_pm_faithfulness(decisions)
    typer.echo(f"\n  PM FAITHFULNESS:")
    typer.echo(f"    overall   = {pm_faith['pm_faithfulness']:.4f}")
    typer.echo(f"    strict    = {pm_faith['pm_faithfulness_strict']:.4f}")
    typer.echo(f"    faithful  = {pm_faith['n_faithful']}")
    typer.echo(f"    unfaithful= {pm_faith['n_unfaithful']}")
    typer.echo(f"    constrained={pm_faith['n_constrained']}")
    if pm_faith["by_ticker"]:
        for tk, stats in sorted(pm_faith["by_ticker"].items()):
            typer.echo(f"    {tk}: {stats}")
    if pm_faith["examples_unfaithful"]:
        typer.echo("    Unfaithful examples:")
        for ex in pm_faith["examples_unfaithful"][:5]:
            print(
                f"      {ex['date']} {ex['ticker']} "
                f"consensus={ex['consensus']:+.2f} dir={ex['direction']:+d} "
                f"tw={ex['target_weight']:.3f} pw={ex['previous_weight']:.3f}"
            )

    # Memory log
    log_path = Path(output_root) / "runs" / run_id / "pm_decision_log.md"
    if log_path.exists():
        raw = log_path.read_text(encoding="utf-8")
        pending = raw.count("| pending]")
        resolved = raw.count("| ret=")
        typer.echo(f"\n  MEMORY LOG: {pending} pending, {resolved} resolved")

    # ── Cost summary ─────────────────────────────────────────────────────────
    typer.echo(f"\n=== COST SUMMARY ===")
    typer.echo(f"  PM decisions  : {n_decisions}")
    typer.echo(f"  Est. LLM calls: {n_decisions} PM dates x 11 calls")
    typer.echo(f"  Est. cost     : ${n_decisions * _APPROX_COST_PER_PM_DATE_USD:.4f} (0 if cache hit)")
    typer.echo(f"  decisions.jsonl: runs/{run_id}/decisions.jsonl")

    # Manifest
    manifest = {
        "run_id": run_id,
        "label": label or f"pm-multiday-{start}-{end}",
        "mode": "pm_multi_agent",
        "tickers": list(guard.tickers),
        "start": start,
        "end": end,
        "model": chosen_model,
        "n_decisions": n_decisions,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "grounding_overall": grounding.get("grounding", 0),
        "pm_faithfulness": pm_faith.get("pm_faithfulness", 0),
        "pm_faithfulness_strict": pm_faith.get("pm_faithfulness_strict", 0),
    }
    manifest_path = Path(output_root) / "results" / run_id / "pm_multiday_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    typer.echo(f"\nManifest: {manifest_path}")
    typer.echo("[DONE] run_pm_multiday.py")


if __name__ == "__main__":
    app()
