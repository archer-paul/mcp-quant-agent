#!/usr/bin/env python
"""Run the guarded Portfolio Manager API smoke test.

The script is dry-run by default and does not call OpenAI. Passing
`--no-dry-run --acknowledge-cost` validates the same guardrails, then hands
off to `PMBacktestEngine(use_stub=False)` for one bounded TradingAgents-style
decision chain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    tickers: list[str] = typer.Option(
        ["AAPL"],
        help="One or two tickers only.",
    ),
    date: str = typer.Option(
        "2023-01-03",
        help="Single smoke-test decision date.",
    ),
    model: str | None = typer.Option(
        None,
        help="Must equal settings.agent_model_dev. Defaults to that setting.",
    ),
    no_cache: bool = typer.Option(
        False,
        help="Disable LLM cache. This is rejected for PM API smoke.",
    ),
    acknowledge_cost: bool = typer.Option(
        False,
        "--acknowledge-cost",
        help="Required acknowledgement that this would burn OpenAI credits.",
    ),
    dry_run: bool = typer.Option(
        True,
        help="Validate guardrails only. Default true; use --no-dry-run to attempt.",
    ),
    initial_cash: float = typer.Option(100_000.0, help="Starting paper cash."),
    allow_empty_news: bool = typer.Option(
        False,
        help="Allow missing news cache and continue with empty news.",
    ),
    output_root: Path = typer.Option(Path("."), help="Root for runs/ and results/."),
) -> None:
    """Validate PM API smoke-test constraints."""
    from mcp_quant_agent.backtest.pm_smoke_guard import (
        validate_pm_api_smoke_request,
    )
    from mcp_quant_agent.config import settings

    chosen_model = model or settings.agent_model_dev
    try:
        result = validate_pm_api_smoke_request(
            tickers=tickers,
            start_date=date,
            end_date=date,
            model=chosen_model,
            dev_model=settings.agent_model_dev,
            use_llm_cache=not no_cache,
            acknowledge_cost=acknowledge_cost,
        )
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"FATAL: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo("PM API smoke guardrails validated:")
    typer.echo(f"  date      : {result.start_date}")
    typer.echo(f"  tickers   : {', '.join(result.tickers)}")
    typer.echo(f"  model     : {result.model}")
    typer.echo(f"  llm_cache : {result.use_llm_cache}")
    typer.echo(f"  PM calls  : {result.estimated_pm_calls}")
    typer.echo(f"  LLM calls : {result.estimated_llm_calls}")

    if dry_run:
        typer.echo("[dry-run] No API call made.")
        raise typer.Exit(0)

    from mcp_quant_agent.backtest.pm_engine import PMBacktestEngine

    try:
        results: dict[str, Any] = PMBacktestEngine(
            tickers=result.tickers,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_cash=initial_cash,
            use_stub=False,
            model=result.model,
            use_llm_cache=result.use_llm_cache,
            smoke_guard=result,
            allow_empty_news=allow_empty_news,
            write_artifacts=True,
            output_root=output_root,
        ).run()
    except RuntimeError as exc:
        typer.echo(f"FATAL: {exc}", err=True)
        raise typer.Exit(1) from exc

    if results.get("error"):
        typer.echo(f"ERROR: {results['error']}", err=True)
        raise typer.Exit(1)

    decisions = results.get("decisions_all", [])
    grounding = {}
    try:
        from mcp_quant_agent.eval.reasoning import compute_grounding

        grounding = compute_grounding(decisions)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Grounding skipped: {exc}", err=True)

    typer.echo("PM API smoke completed:")
    typer.echo(f"  run_id      : {results.get('run_id')}")
    typer.echo(f"  decisions   : {results.get('n_decisions')}")
    typer.echo(f"  final_nav   : {results.get('nav_series', [0.0])[-1] if results.get('nav_series') else 0.0:.2f}")
    if grounding:
        typer.echo(
            "  grounding   : "
            f"{grounding['grounding']:.4f} "
            f"({grounding['n_grounded']}/{grounding['n_claims_total']} claims)"
        )
    typer.echo(f"decisions.jsonl: {output_root / 'runs' / results['run_id'] / 'decisions.jsonl'}")
    typer.echo(f"summary.csv    : {output_root / 'results' / results['run_id'] / 'summary.csv'}")


if __name__ == "__main__":
    app()
