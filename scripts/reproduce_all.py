#!/usr/bin/env python
"""Reproduce all thesis results with a single command.

Usage::

    python scripts/reproduce_all.py

This script runs (in order):
1. Baseline backtest (Buy&Hold, momentum, mean-reversion).
2. Agent backtest (single-agent, multi-agent if available).
3. Regime labelling.
4. CoT evaluation (faithfulness, grounding, sophistication).
5. Final metric table + figures.

All outputs are saved to ``runs/<timestamp>/``.

Status: stub — implementation after all components are in place (J14 sprint).
"""

from __future__ import annotations

import typer

app = typer.Typer()


@app.command()
def main(
    dry_run: bool = typer.Option(False, help="Print the steps without running them."),
) -> None:
    """Reproduce all thesis results."""
    steps = [
        "1. Run baselines: python scripts/run_baselines.py",
        "2. Run agent backtest: python scripts/run_agent_backtest.py",
        "3. Regime labelling: (integrated in engine)",
        "4. CoT evaluation: (integrated in engine)",
        "5. Generate metric table and figures",
    ]
    for step in steps:
        typer.echo(step)

    if dry_run:
        typer.echo("Dry run — steps listed above, not executed.")
        return

    typer.echo("⚠  reproduce_all.py not yet fully implemented — see docs/PLAN.md J14.")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
