#!/usr/bin/env python
"""Run the agent backtest.

Usage::

    python scripts/run_agent_backtest.py --config configs/single_agent.yaml

This is the main experiment script for the thesis.  It:
1. Sets up the simulation clock.
2. Initialises the paper-trading portfolio.
3. Runs the backtest engine (bar by bar, advancing the clock).
4. Computes performance metrics and saves results.
5. Logs to Langfuse for CoT evaluation.

Status: stub — depends on backtest/engine.py (J5 sprint).
"""

from __future__ import annotations

import typer

app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option("configs/single_agent.yaml", help="Config file path."),
    dry_run: bool = typer.Option(False, help="Validate config and exit without running."),
) -> None:
    """Run the agent backtest using the given config."""
    import yaml  # type: ignore[import]
    from pathlib import Path

    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}", err=True)
        raise typer.Exit(1)

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    typer.echo(f"Config: {cfg}")

    if dry_run:
        typer.echo("Dry run — exiting.")
        return

    try:
        from mcp_quant_agent.backtest.engine import BacktestEngine
    except ImportError as exc:
        typer.echo(f"Import error: {exc}", err=True)
        raise typer.Exit(1) from exc

    try:
        engine = BacktestEngine(
            tickers=cfg.get("tickers", ["AAPL"]),
            start_date=cfg.get("start_date", "2022-01-01"),
            end_date=cfg.get("end_date", "2024-12-31"),
            initial_cash=cfg.get("initial_cash", 100_000.0),
            model=cfg.get("model", "gpt-4.1-mini"),
        )
        results = engine.run()
    except NotImplementedError as exc:
        typer.echo(f"⚠  Not yet implemented: {exc}")
        raise typer.Exit(1) from exc

    import json
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
