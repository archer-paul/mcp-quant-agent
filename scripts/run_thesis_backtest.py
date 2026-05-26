#!/usr/bin/env python
"""Thesis run: AAPL/MSFT/NVDA/JPM/XOM, 2022-07-01 -> 2024-06-30, gpt-4.1-mini.

Usage::

    python scripts/run_thesis_backtest.py
    python scripts/run_thesis_backtest.py --dry-run   # cost estimate only

Output
------
- runs/<run_id>/decisions.jsonl  — full decision schema per bar (rejouable)
- results/<run_id>/summary.csv   — human-readable performance table

This is the primary thesis experiment.  Results are cached; re-runs cost $0.
"""

from __future__ import annotations

import logging

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
START = "2022-07-01"
END = "2024-06-30"
MODEL = "gpt-4.1-mini"

# Cost estimate: 2520 calls * $0.00094/call ≈ $2.37 (first run)
ESTIMATED_COST_USD = 2.37


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show cost estimate and exit."),
    tickers: list[str] = typer.Option(TICKERS, help="Ticker universe."),
    start: str = typer.Option(START, help="Start date (ISO-8601)."),
    end: str = typer.Option(END, help="End date (ISO-8601)."),
    model: str = typer.Option(MODEL, help="OpenAI model."),
    seed: int = typer.Option(42, help="Random seed (for future sampling; noted in results)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip cost confirmation."),
) -> None:
    """Run the full thesis backtest on the equity universe."""
    import datetime as dt

    n_tickers = len(tickers)
    # Approximate trading days in the range
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    calendar_days = (end_d - start_d).days
    approx_trading_days = int(calendar_days * 252 / 365)
    n_calls = n_tickers * approx_trading_days
    estimated_cost = n_calls * 0.00094  # per-call estimate

    typer.echo()
    typer.echo("=" * 70)
    typer.echo(" THESIS BACKTEST — COST ESTIMATE")
    typer.echo("=" * 70)
    typer.echo(f"  Tickers   : {', '.join(tickers)}")
    typer.echo(f"  Period    : {start} -> {end}")
    typer.echo(f"  Model     : {model}")
    typer.echo(f"  Est. calls: {n_calls:,} ({n_tickers} tickers x ~{approx_trading_days} bars)")
    typer.echo(f"  Est. cost : ${estimated_cost:.2f} (first run; ${0:.2f} with full cache)")
    typer.echo()

    if dry_run:
        typer.echo("[--dry-run] Exiting without running.")
        raise typer.Exit(0)

    if estimated_cost > 5.0 and not yes:
        typer.echo(f"[WARNING] Estimated cost ${estimated_cost:.2f} exceeds $5.00.")
        confirmed = typer.confirm("Proceed?")
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    typer.echo(f"[!] Launching real OpenAI run -- cost ~${estimated_cost:.2f}")
    typer.echo()

    from mcp_quant_agent.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        tickers=tickers,
        start_date=start,
        end_date=end,
        model=model,
        use_stub=False,
        use_llm_cache=True,
    )

    results = engine.run()

    # ── Print summary ─────────────────────────────────────────────────────────
    metrics = results.get("metrics", {})
    sharpe_ci = results.get("sharpe_ci", {})
    run_id = results.get("run_id", "unknown")

    typer.echo()
    typer.echo("=" * 70)
    typer.echo(" THESIS RUN RESULTS")
    typer.echo(f" run_id  : {run_id}")
    typer.echo(f" Period  : {start} -> {end}")
    typer.echo(f" Tickers : {', '.join(tickers)}")
    typer.echo(f" Model   : {model}")
    typer.echo("=" * 70)
    typer.echo(
        f"  Bars       : {results['n_bars']}"
        f"  |  Decisions: {results['n_decisions']}"
    )
    typer.echo(
        f"  AnnRet     : {metrics.get('annualised_return', 0)*100:+.1f}%"
        f"  |  Sharpe: {metrics.get('sharpe', 0):.2f}"
        f"  [{sharpe_ci.get('ci_lower', 0):.2f}, {sharpe_ci.get('ci_upper', 0):.2f}]"
    )
    typer.echo(
        f"  Sortino    : {metrics.get('sortino', 0):.2f}"
        f"  |  Calmar: {metrics.get('calmar', 0):.2f}"
        f"  |  MaxDD: {metrics.get('max_drawdown', 0)*100:.1f}%"
    )
    typer.echo(
        f"  HitRate    : {metrics.get('hit_rate', 0)*100:.1f}%"
        f"  |  FinalNAV: {results['nav_series'][-1] if results['nav_series'] else 0:.0f}"
    )
    typer.echo()

    # Regime distribution from decisions
    all_decisions = results.get("decisions_all", [])
    regime_counts: dict[str, int] = {}
    for d in all_decisions:
        r = str(d.get("regime") or "unknown")
        regime_counts[r] = regime_counts.get(r, 0) + 1
    if regime_counts:
        typer.echo("  Regime distribution (decision-days):")
        for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(all_decisions) if all_decisions else 0
            typer.echo(f"    {regime:<12}: {count:4d} days ({pct:.0f}%)")
    typer.echo()

    # Last few decisions
    typer.echo("  Last 10 decisions:")
    for d in results.get("decisions_log", [])[-10:]:
        fill_info = ""
        if d.get("fill"):
            fill = d["fill"]
            fill_info = f" -> {fill.get('side','?')} {fill.get('quantity','?')} @ {fill.get('price','?'):.2f}"
        errs = f" ERRORS: {d['errors']}" if d.get("errors") else ""
        typer.echo(
            f"    {d['date']} {d.get('ticker',''):<5}  "
            f"{d.get('action','?'):<5} [{d.get('regime','?')}]{fill_info}{errs}"
        )
    typer.echo()

    typer.echo(
        f"  decisions.jsonl : runs/{run_id}/decisions.jsonl"
    )
    typer.echo(
        f"  summary.csv     : results/{run_id}/summary.csv"
    )


if __name__ == "__main__":
    app()
