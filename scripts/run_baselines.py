#!/usr/bin/env python
"""Run all systematic baselines and print a performance comparison table.

Usage::

    python scripts/run_baselines.py \\
        --universe AAPL MSFT NVDA JPM XOM \\
        --start 2022-01-01 \\
        --end 2024-12-31

This script runs Buy&Hold, time-series momentum, and mean-reversion
(Bollinger) for each ticker over the specified date range, then prints
a Sharpe/Sortino/Calmar/maxDD comparison table.

These baselines are the yardstick against which the agent is evaluated.
Run them BEFORE evaluating the agent — if the agent cannot beat Buy&Hold
on Sharpe, that is an important and honest result.

Status: stub — depends on backtest/baselines.py (J3 sprint).
"""

from __future__ import annotations

import sys

import typer

app = typer.Typer()


@app.command()
def main(
    universe: list[str] = typer.Option(
        ["AAPL", "MSFT", "NVDA", "JPM", "XOM"], help="Ticker universe."
    ),
    start: str = typer.Option("2022-01-01", help="Backtest start date (ISO-8601)."),
    end: str = typer.Option("2024-12-31", help="Backtest end date (ISO-8601)."),
    initial_cash: float = typer.Option(100_000.0, help="Starting capital."),
) -> None:
    """Run all systematic baselines and print the performance table."""
    try:
        from mcp_quant_agent.backtest.baselines import run_all_baselines
    except ImportError as exc:
        typer.echo(f"Import error: {exc}", err=True)
        raise typer.Exit(1) from exc

    try:
        results = run_all_baselines(universe, start, end, initial_cash)
    except NotImplementedError as exc:
        typer.echo(f"⚠  Not yet implemented: {exc}")
        typer.echo("See docs/PLAN.md (J3) for the implementation sprint.")
        raise typer.Exit(1) from exc

    # TODO: pretty-print results table
    import json
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
