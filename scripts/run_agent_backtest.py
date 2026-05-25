#!/usr/bin/env python
"""Run the LLM-agent backtest and print a performance table.

Usage (stub -- no API key needed)::

    python scripts/run_agent_backtest.py --stub
    python scripts/run_agent_backtest.py --stub --tickers AAPL --start 2023-01-01 --end 2023-01-31

Usage (real OpenAI backbone -- costs money!)::

    # [!] SIGNAL THE USER FIRST -- this burns OpenAI credits.
    python scripts/run_agent_backtest.py --no-stub --tickers AAPL --start 2023-01-01 --end 2023-02-28

Outputs
-------
- Stdout : performance table identical in format to run_baselines.py
- results/agent_<backbone>_<timestamp>.parquet   (if --save-results, default True)
- results/agent_<backbone>_<timestamp>.csv

Notes
-----
- Default period is deliberately SHORT (1 month, 1 ticker) so first-pass smoke
  tests run in seconds and cost nothing.
- Stub results MUST NOT be cited in the thesis.  They prove the plumbing works.
- For the real run (--no-stub) make sure OPENAI_API_KEY is set in .env.
- LLM response cache is ON by default (--no-cache to disable for final runs).
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Formatting helpers (mirrors run_baselines.py for visual consistency)
# ---------------------------------------------------------------------------


def _format_pct(v: float) -> str:
    return f"{v * 100:+.1f}%"


def _format_float(v: float, decimals: int = 2) -> str:
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.{decimals}f}"


def _print_agent_table(results: dict[str, Any], backbone: str) -> None:
    """Print the agent performance table to stdout."""
    metrics = results.get("metrics", {})
    sharpe_ci = results.get("sharpe_ci", {})
    n_bars = results.get("n_bars", 0)
    n_decisions = results.get("n_decisions", 0)
    tickers = results.get("tickers", [])
    start_date = results.get("start_date", "?")
    end_date = results.get("end_date", "?")

    header = (
        f"{'Strategy':<22} {'Tickers':<20} {'AnnRet':>7} {'Sharpe':>7} "
        f"{'Sortino':>7} {'Calmar':>7} {'MaxDD':>7} {'HitRate':>8} {'Decisions':>10}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(" AGENT PERFORMANCE SUMMARY")
    print(f" Backbone : {backbone}")
    print(f" Period   : {start_date} -> {end_date}")
    print(f" Universe : {', '.join(tickers)}")
    print(f" Bars     : {n_bars}  |  Decisions: {n_decisions}")
    if backbone.startswith("stub"):
        print(" [!] STUB RESULTS -- plumbing test only. NEVER cite in thesis.")
    print("=" * len(header))
    print(header)
    print(sep)

    if metrics:
        ci_str = ""
        if sharpe_ci:
            ci_str = (
                f"  [Sharpe CI: {_format_float(sharpe_ci.get('ci_lower', 0.0))}"
                f"-{_format_float(sharpe_ci.get('ci_upper', 0.0))}]"
            )
        print(
            f"{'LLM Agent':<22} {', '.join(tickers):<20} "
            f"{_format_pct(metrics.get('annualised_return', 0.0)):>7} "
            f"{_format_float(metrics.get('sharpe', 0.0)):>7} "
            f"{_format_float(metrics.get('sortino', 0.0)):>7} "
            f"{_format_float(metrics.get('calmar', 0.0)):>7} "
            f"{_format_pct(metrics.get('max_drawdown', 0.0)):>7} "
            f"{_format_pct(metrics.get('hit_rate', 0.0)):>8} "
            f"{n_decisions:>10}"
            f"{ci_str}"
        )
    else:
        print("  (no metrics -- check errors below)")
    print(sep)
    print()

    # Recent decisions
    decisions_log = results.get("decisions_log", [])
    if decisions_log:
        print(f"  Last {len(decisions_log)} decisions:")
        for d in decisions_log:
            fill_str = ""
            if d.get("fill"):
                f = d["fill"]
                fill_str = (
                    f" -> filled {f.get('quantity', 0)} @ {f.get('price', 0):.2f}"
                )
            err_str = ""
            if d.get("errors"):
                err_str = f" ERRORS: {d['errors']}"
            regime_str = f" [{d['regime']}]" if d.get("regime") else ""
            print(
                f"    {d['date']} {d['ticker']:6} {d['action']:5}"
                f"{fill_str}{regime_str}{err_str}"
            )
        print()


def _save_results(results: dict[str, Any], backbone: str, output_dir: Path) -> None:
    """Save agent results to parquet and CSV."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_backbone = backbone.replace("/", "-").replace(".", "-")

    metrics = results.get("metrics", {})
    sharpe_ci = results.get("sharpe_ci", {})
    rows = [
        {
            "backbone": backbone,
            "tickers": ",".join(results.get("tickers", [])),
            "start_date": results.get("start_date"),
            "end_date": results.get("end_date"),
            "n_bars": results.get("n_bars", 0),
            "n_decisions": results.get("n_decisions", 0),
            "annualised_return": metrics.get("annualised_return", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "sharpe_ci_lower": sharpe_ci.get("ci_lower", 0.0),
            "sharpe_ci_upper": sharpe_ci.get("ci_upper", 0.0),
            "sortino": metrics.get("sortino", 0.0),
            "calmar": metrics.get("calmar", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "hit_rate": metrics.get("hit_rate", 0.0),
        }
    ]

    df = pd.DataFrame(rows)
    parquet_path = output_dir / f"agent_{safe_backbone}_{ts}.parquet"
    csv_path = output_dir / f"agent_{safe_backbone}_{ts}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    typer.echo(f"Results saved to {parquet_path}")
    typer.echo(f"           and {csv_path}")

    # Also save full decisions log as CSV
    decisions_log = results.get("decisions_log", [])
    if decisions_log:
        log_path = output_dir / f"agent_{safe_backbone}_{ts}_decisions.csv"
        pd.DataFrame(decisions_log).to_csv(log_path, index=False)
        typer.echo(f"      decisions: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    tickers: list[str] = typer.Option(
        ["AAPL"],
        help="Ticker universe (default: AAPL only for fast smoke test).",
    ),
    start: str = typer.Option(
        "2023-01-01",
        help="Backtest start date (ISO-8601). Default: 1 month for fast smoke test.",
    ),
    end: str = typer.Option(
        "2023-01-31",
        help="Backtest end date (ISO-8601). Default: 1 month for fast smoke test.",
    ),
    initial_cash: float = typer.Option(100_000.0, help="Starting capital."),
    model: str = typer.Option(
        "gpt-4.1-mini",
        help="OpenAI model ID (ignored when --stub is set).",
    ),
    stub: bool = typer.Option(
        True,
        help=(
            "Use deterministic stub backbone (no API key). "
            "DEFAULT TRUE for safety -- set --no-stub for real OpenAI run "
            "(SIGNAL USER FIRST -- costs money)."
        ),
    ),
    no_cache: bool = typer.Option(
        False,
        help="Disable LLM response disk cache (for final thesis runs).",
    ),
    save_results: bool = typer.Option(True, help="Save results to results/ directory."),
) -> None:
    """Run the LLM agent backtest and print a performance table.

    Defaults to a stub backbone on a short period (1 month, AAPL) so the
    first smoke test runs in seconds without any API key.

    [!] USE --no-stub ONLY AFTER SIGNALLING THE USER -- it burns OpenAI credits.
    """
    if not stub:
        typer.echo(
            "[!] --no-stub: real OpenAI backbone selected -- this costs money!\n"
            "   Make sure OPENAI_API_KEY is set in .env and you intended this.",
            err=True,
        )

    from mcp_quant_agent.backtest.engine import BacktestEngine

    backbone_label = "stub" if stub else model
    typer.echo(
        f"Running agent backtest  backbone={backbone_label}  "
        f"tickers={tickers}  {start} -> {end} ..."
    )

    try:
        engine = BacktestEngine(
            tickers=list(tickers),
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            model=model,
            use_stub=stub,
            use_llm_cache=not no_cache,
        )
        results = engine.run()
    except RuntimeError as exc:
        typer.echo(f"FATAL: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc

    if results.get("error"):
        typer.echo(f"[!] Backtest error: {results['error']}", err=True)
        raise typer.Exit(1)

    _print_agent_table(results, backbone_label)

    if save_results:
        _save_results(results, backbone_label, Path("results"))


if __name__ == "__main__":
    app()
