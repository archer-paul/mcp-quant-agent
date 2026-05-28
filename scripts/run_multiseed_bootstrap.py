#!/usr/bin/env python
"""Multi-seed real-LLM run + bootstrap CI demonstration.

Runs n_seeds instances of the single-agent on the same bounded window.
With LLM cache ON, all re-runs cost $0 (same prompts -> same cached responses).
The per-run variation comes from portfolio order randomness (there is none: same
tickers, same dates, same cache -> identical decisions). But bootstrap is applied
to the NAV series WITHIN each run to compute Sharpe CI.

Then: the across-seed Sharpe distribution is computed to show run-to-run stability.

Usage:
  python scripts/run_multiseed_bootstrap.py --tickers AAPL --tickers MSFT --tickers NVDA \\
      --start 2023-01-03 --end 2023-01-09 --acknowledge-cost --n-seeds 3
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_APPROX_COST_PER_DECISION = 0.00015

app = typer.Typer(add_completion=False)


@app.command()
def main(
    tickers: list[str] = typer.Option(...),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    model: str = typer.Option("gpt-4.1-mini"),
    n_seeds: int = typer.Option(3, help="Number of seeds (default 3)."),
    acknowledge_cost: bool = typer.Option(False, "--acknowledge-cost"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_root: Path = typer.Option(Path(".")),
) -> None:
    """Multi-seed single-agent run + bootstrap CI."""
    from mcp_quant_agent.eval.financial import bootstrap_sharpe_ci, compute_all_metrics

    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if len(tickers) > 3:
        typer.echo("FATAL: max 3 tickers", err=True)
        raise typer.Exit(1)
    if n_seeds > 5:
        typer.echo("FATAL: max 5 seeds in this script", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n=== MULTI-SEED BOOTSTRAP ({n_seeds} seeds) ===")
    typer.echo(f"  tickers : {', '.join(tickers)}")
    typer.echo(f"  window  : {start} -> {end}")
    typer.echo(f"  model   : {model}")
    typer.echo("  NOTE: with cache ON, re-runs cost $0 (same prompts -> same responses)")

    if dry_run:
        typer.echo("\n[DRY RUN] Guardrail check passed.")
        return
    if not acknowledge_cost:
        typer.echo("FATAL: pass --acknowledge-cost", err=True)
        raise typer.Exit(1)

    from mcp_quant_agent.backtest.engine import BacktestEngine

    seed_results: list[dict] = []
    for seed in range(n_seeds):
        typer.echo(f"\n[Seed {seed}/{n_seeds-1}] Running ...")
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"multiseed_s{seed}_{ts}"

        engine = BacktestEngine(
            tickers=tickers,
            start_date=start,
            end_date=end,
            model=model,
            use_stub=False,
            use_llm_cache=True,
            run_id=run_id,
        )
        results = engine.run()
        nav = results.get("nav_series", [])
        metrics = results.get("metrics", {})
        ci = bootstrap_sharpe_ci(nav, n_bootstrap=1000, seed=seed) if len(nav) > 5 else {}
        seed_results.append({
            "seed": seed,
            "run_id": run_id,
            "n_decisions": results.get("n_decisions", 0),
            "nav_series": nav,
            "sharpe": metrics.get("sharpe", 0),
            "annualised_return": metrics.get("annualised_return", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "ci_lower": ci.get("ci_lower", 0),
            "ci_upper": ci.get("ci_upper", 0),
        })
        typer.echo(
            f"  seed={seed}: sharpe={metrics.get('sharpe', 0):.3f}  "
            f"CI=[{ci.get('ci_lower', 0):.3f}, {ci.get('ci_upper', 0):.3f}]  "
            f"n_decisions={results.get('n_decisions', 0)}"
        )

    # Across-seed distribution
    sharpes = [r["sharpe"] for r in seed_results]
    typer.echo(f"\n=== MULTI-SEED SUMMARY ===")
    typer.echo(f"  n_seeds     = {n_seeds}")
    typer.echo(f"  Sharpe mean = {sum(sharpes)/len(sharpes):.3f}")
    typer.echo(f"  Sharpe min  = {min(sharpes):.3f}")
    typer.echo(f"  Sharpe max  = {max(sharpes):.3f}")
    typer.echo(f"  Sharpe std  = {_std(sharpes):.3f}")

    typer.echo("\n  Per-seed table:")
    typer.echo(f"  {'Seed':>6} {'Sharpe':>8} {'CI lo':>8} {'CI hi':>8} {'AnnRet':>8}")
    for r in seed_results:
        typer.echo(
            f"  {r['seed']:>6} {r['sharpe']:>8.3f} "
            f"{r['ci_lower']:>8.3f} {r['ci_upper']:>8.3f} "
            f"{r['annualised_return']*100:>+7.1f}%"
        )

    # Note on same-cache behavior
    n_unique_sharpes = len(set(round(s, 6) for s in sharpes))
    if n_unique_sharpes == 1:
        typer.echo(
            "\n  NOTE: All seeds produced identical Sharpe (expected with cache ON + "
            "deterministic engine). Multi-seed CI is meaningful only with --no-cache "
            "or on stochastic LLMs. Bootstrap CI within a single run IS meaningful."
        )

    # Save manifest
    manifest_path = Path(output_root) / "results" / f"multiseed_{start}_{end}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(seed_results, indent=2, default=str), encoding="utf-8")
    typer.echo(f"\nManifest: {manifest_path}")
    typer.echo("[DONE]")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


if __name__ == "__main__":
    app()
