#!/usr/bin/env python
"""Run the PM multi-agent stub backtest.

This script is deliberately stub-first. It does not call OpenAI. News access is
cache-first/offline by default, so a missing news cache fails loudly unless
`--allow-empty-news` is passed for plumbing-only local smoke tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = typer.Typer(add_completion=False)


def _format_pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _print_summary(results: dict[str, Any]) -> None:
    metrics = results.get("metrics", {})
    tickers = results.get("tickers", [])
    print()
    print("=" * 72)
    print(" PM MULTI-AGENT STUB SUMMARY")
    print(" [!] STUB RESULTS -- plumbing test only. NEVER cite in thesis.")
    print("=" * 72)
    print(f" Run ID    : {results.get('run_id')}")
    print(f" Period    : {results.get('start_date')} -> {results.get('end_date')}")
    print(f" Universe  : {', '.join(tickers)}")
    print(f" Bars      : {results.get('n_bars')} | Decisions: {results.get('n_decisions')}")
    if metrics:
        print(
            " Metrics   : "
            f"AnnRet {_format_pct(float(metrics.get('annualised_return', 0.0)))} | "
            f"Sharpe {float(metrics.get('sharpe', 0.0)):.2f} | "
            f"MaxDD {_format_pct(float(metrics.get('max_drawdown', 0.0)))}"
        )
    print()
    for decision in results.get("decisions_log", [])[-5:]:
        fills = decision.get("fills", [])
        fill_text = f"{len(fills)} fills" if fills else "no fills"
        weights = decision.get("targets", {}).get("weights", {})
        print(f" {decision['date']} {fill_text} targets={weights}")
    print()


@app.command()
def main(
    tickers: list[str] = typer.Option(
        ["AAPL", "MSFT"],
        help="Ticker universe for the PM stub.",
    ),
    start: str = typer.Option("2023-01-03", help="Backtest start date."),
    end: str = typer.Option("2023-01-31", help="Backtest end date."),
    initial_cash: float = typer.Option(100_000.0, help="Starting capital."),
    allow_empty_news: bool = typer.Option(
        False,
        help=(
            "Allow missing news cache and continue with empty news. "
            "Use only for local plumbing smoke tests."
        ),
    ),
    output_root: Path = typer.Option(Path("."), help="Root for runs/ and results/."),
) -> None:
    """Run a short PM stub backtest and write decisions.jsonl."""
    from mcp_quant_agent.backtest.pm_engine import PMBacktestEngine

    typer.echo(
        "Running PM stub backtest "
        f"tickers={tickers} period={start}->{end} "
        f"allow_empty_news={allow_empty_news}"
    )
    engine = PMBacktestEngine(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        use_stub=True,
        allow_empty_news=allow_empty_news,
        write_artifacts=True,
        output_root=output_root,
    )
    try:
        results = engine.run()
    except RuntimeError as exc:
        typer.echo(f"FATAL: {exc}", err=True)
        raise typer.Exit(1) from exc

    if results.get("error"):
        typer.echo(f"ERROR: {results['error']}", err=True)
        raise typer.Exit(1)

    _print_summary(results)
    typer.echo(f"decisions.jsonl: {output_root / 'runs' / results['run_id'] / 'decisions.jsonl'}")
    typer.echo(f"summary.csv    : {output_root / 'results' / results['run_id'] / 'summary.csv'}")


if __name__ == "__main__":
    app()
