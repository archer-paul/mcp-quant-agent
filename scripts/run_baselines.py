#!/usr/bin/env python
"""Run all systematic baselines and print a performance comparison table.

Usage::

    python scripts/run_baselines.py
    python scripts/run_baselines.py --universe AAPL MSFT --start 2022-01-01 --end 2022-12-31
    python scripts/run_baselines.py --output results/baselines.parquet

Outputs
-------
- Stdout: rich performance table (Ann.Ret | Sharpe | Sortino | Calmar | MaxDD | HitRate | Trades)
- results/baselines_<timestamp>.parquet (if --output specified or always saved to results/)
- results/baselines_<timestamp>.csv     (human-readable copy)

These baselines are the yardstick against which the agent is evaluated.
Run them BEFORE evaluating the agent -- if the agent cannot beat Buy&Hold
on Sharpe, that is an important and honest result.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _format_pct(v: float) -> str:
    return f"{v * 100:+.1f}%"


def _format_float(v: float, decimals: int = 2) -> str:
    if v == float("inf"):
        return "inf"
    return f"{v:.{decimals}f}"


def _print_table(results: dict[str, Any]) -> None:
    """Print a readable comparison table to stdout."""
    header = (
        f"{'Strategy':<22} {'Ticker':<8} {'AnnRet':>7} {'Sharpe':>7} "
        f"{'Sortino':>7} {'Calmar':>7} {'MaxDD':>7} {'HitRate':>8} {'Trades':>7}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(" BASELINE PERFORMANCE SUMMARY")
    print(f" Period: {results['meta']['start_date']} -> {results['meta']['end_date']}")
    print(f" Universe: {', '.join(results['meta']['tickers'])}")
    print(f" Initial cash: ${results['meta']['initial_cash']:,.0f}/ticker")
    print(
        f" Transaction cost: {results['meta']['cost_per_trade_bps']:.0f} bps/trade (one-way)"
    )
    print("=" * len(header))
    print(header)
    print(sep)

    strategies = ["buy_and_hold", "momentum_ts", "mean_reversion_bb"]
    strategy_labels = {
        "buy_and_hold": "Buy & Hold",
        "momentum_ts": "Momentum TS",
        "mean_reversion_bb": "Mean-Rev BB",
    }

    per_ticker = results.get("per_ticker", {})
    for ticker in sorted(per_ticker):
        for strategy in strategies:
            if strategy not in per_ticker[ticker]:
                continue
            r = per_ticker[ticker][strategy]
            m = r["metrics"]
            label = strategy_labels.get(strategy, strategy)
            print(
                f"{label:<22} {ticker:<8} "
                f"{_format_pct(m['annualised_return']):>7} "
                f"{_format_float(m['sharpe']):>7} "
                f"{_format_float(m['sortino']):>7} "
                f"{_format_float(m['calmar']):>7} "
                f"{_format_pct(m['max_drawdown']):>7} "
                f"{_format_pct(m['hit_rate']):>8} "
                f"{r['n_trades']:>7}"
            )
        print(sep)

    equal_weight = results.get("equal_weight", {})
    if equal_weight:
        print("  EQUAL-WEIGHTED AGGREGATE")
        print(sep)
        for strategy in strategies:
            if strategy not in equal_weight:
                continue
            r = equal_weight[strategy]
            m = r["metrics"]
            ci = r.get("sharpe_ci", {})
            label = strategy_labels.get(strategy, strategy)
            ci_str = ""
            if ci:
                ci_str = (
                    f"  [Sharpe CI: {_format_float(ci['ci_lower'])}"
                    f"-{_format_float(ci['ci_upper'])}]"
                )
            print(
                f"{label:<22} {'EQ-WT':<8} "
                f"{_format_pct(m['annualised_return']):>7} "
                f"{_format_float(m['sharpe']):>7} "
                f"{_format_float(m['sortino']):>7} "
                f"{_format_float(m['calmar']):>7} "
                f"{_format_pct(m['max_drawdown']):>7} "
                f"{_format_pct(m['hit_rate']):>8} "
                f"{r['n_trades']:>7}"
                f"{ci_str}"
            )
        print(sep)
    print()


def _save_results(results: dict[str, Any], output_dir: Path) -> None:
    """Save results to parquet and CSV."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    for ticker, strategies in results.get("per_ticker", {}).items():
        for strategy, r in strategies.items():
            m = r["metrics"]
            ci = r.get("sharpe_ci", {})
            rows.append(
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "annualised_return": m.get("annualised_return", 0.0),
                    "sharpe": m.get("sharpe", 0.0),
                    "sharpe_ci_lower": ci.get("ci_lower", 0.0),
                    "sharpe_ci_upper": ci.get("ci_upper", 0.0),
                    "sortino": m.get("sortino", 0.0),
                    "calmar": m.get("calmar", 0.0),
                    "max_drawdown": m.get("max_drawdown", 0.0),
                    "hit_rate": m.get("hit_rate", 0.0),
                    "cost_drag_bps": m.get("cost_drag_bps", 0.0),
                    "turnover_pct": m.get("turnover_pct", 0.0),
                    "n_trades": r.get("n_trades", 0),
                    "turnover": r.get("turnover", 0.0),
                    "start_date": results["meta"]["start_date"],
                    "end_date": results["meta"]["end_date"],
                }
            )
    for strategy, r in results.get("equal_weight", {}).items():
        m = r["metrics"]
        ci = r.get("sharpe_ci", {})
        rows.append(
            {
                "ticker": "EQUAL-WT",
                "strategy": strategy,
                "annualised_return": m.get("annualised_return", 0.0),
                "sharpe": m.get("sharpe", 0.0),
                "sharpe_ci_lower": ci.get("ci_lower", 0.0),
                "sharpe_ci_upper": ci.get("ci_upper", 0.0),
                "sortino": m.get("sortino", 0.0),
                "calmar": m.get("calmar", 0.0),
                "max_drawdown": m.get("max_drawdown", 0.0),
                "hit_rate": m.get("hit_rate", 0.0),
                "cost_drag_bps": m.get("cost_drag_bps", 0.0),
                "turnover_pct": m.get("turnover_pct", 0.0),
                "n_trades": r.get("n_trades", 0),
                "turnover": r.get("turnover", 0.0),
                "start_date": results["meta"]["start_date"],
                "end_date": results["meta"]["end_date"],
            }
        )

    df = pd.DataFrame(rows)
    parquet_path = output_dir / f"baselines_{ts}.parquet"
    csv_path = output_dir / f"baselines_{ts}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    typer.echo(f"Results saved to {parquet_path}")
    typer.echo(f"           and {csv_path}")


@app.command()
def main(
    universe: list[str] = typer.Option(
        ["AAPL", "MSFT", "NVDA", "JPM", "XOM"],
        help="Ticker universe.",
    ),
    start: str = typer.Option("2022-01-01", help="Backtest start date (ISO-8601)."),
    end: str = typer.Option("2024-12-31", help="Backtest end date (ISO-8601)."),
    initial_cash: float = typer.Option(100_000.0, help="Starting capital per ticker."),
    momentum_lookback: int = typer.Option(
        252, help="Momentum lookback in trading days."
    ),
    bb_window: int = typer.Option(
        20, help="Bollinger-band lookback in trading days."
    ),
    bb_num_std: float = typer.Option(
        2.0, help="Bollinger-band standard-deviation multiplier."
    ),
    save_results: bool = typer.Option(True, help="Save results to results/ directory."),
    json_output: bool = typer.Option(
        False, help="Print full JSON results (in addition to table)."
    ),
) -> None:
    """Run all systematic baselines and print the performance table.

    Produces Buy & Hold, Time-Series Momentum, and Mean-Reversion (Bollinger)
    for each ticker in the universe, plus an equal-weighted aggregate.
    """
    from mcp_quant_agent.backtest.baselines import run_all_baselines

    typer.echo(f"Running baselines on {universe}, {start} -> {end}...")

    try:
        results = run_all_baselines(
            tickers=list(universe),
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            momentum_lookback=momentum_lookback,
            bb_window=bb_window,
            bb_num_std=bb_num_std,
        )
    except Exception as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not results.get("per_ticker"):
        typer.echo("No results produced -- check tickers and date range.", err=True)
        raise typer.Exit(1)

    _print_table(results)

    if save_results:
        _save_results(results, Path("results"))

    if json_output:
        # Exclude large nav_series lists from JSON output
        compact = {
            "meta": results["meta"],
            "per_ticker": {
                ticker: {
                    strategy: {k: v for k, v in r.items() if k != "nav_series"}
                    for strategy, r in strategies.items()
                }
                for ticker, strategies in results["per_ticker"].items()
            },
            "equal_weight": {
                strategy: {k: v for k, v in r.items() if k != "nav_series"}
                for strategy, r in results.get("equal_weight", {}).items()
            },
        }
        typer.echo(json.dumps(compact, indent=2))


if __name__ == "__main__":
    app()
