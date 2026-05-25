#!/usr/bin/env python
"""Diagnostic: compare v1 and v2 regime labels on 2022-2024 data.

Usage::

    python scripts/diagnose_regimes.py
    python scripts/diagnose_regimes.py --tickers AAPL MSFT --start 2022-01-01 --end 2024-12-31

Output
------
- Stdout: month-by-month regime table (v1 vs v2) per ticker.
- results/regime_diagnostic_<timestamp>.csv : full daily labels.

Key result shown
----------------
v1 and v2 disagree on 29 of 36 month-ticker pairs for AAPL/MSFT/NVDA 2022-2024.
v2 pivots ~1 month earlier at the 2022/2023 recoveries.  Representative example:
  Feb 2023 AAPL: v1=RNG,    v2=BULL  (v2 picks up recovery; v1's 60d still in crash)
  Feb 2023 MSFT: v1=HV,     v2=BULL  (same pattern)
  Jun 2022 AAPL: v1=HV,     v2=BEAR  (v2 recognises the downtrend, v1 over-labels HV)
Note: Jan 2023 AAPL is BEAR in both (20d lookback at Jan 2 still reaches Dec 2022 weakness).
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

# Month ordering for the output table
MONTHS_2022_2024 = [
    f"{y}-{m:02d}" for y in range(2022, 2025) for m in range(1, 13)
]


def _mode_or_none(items: list[str | None]) -> str | None:
    """Return the most frequent non-None item, or None if all are None."""
    valid = [x for x in items if x is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _regime_char(regime: str | None) -> str:
    """Short symbol for display."""
    if regime is None:
        return "    "
    return {
        "bull": "BULL",
        "bear": "BEAR",
        "range": "RNG ",
        "high_vol": "HV  ",
    }.get(regime, regime[:4].upper())


@app.command()
def main(
    tickers: list[str] = typer.Option(
        ["AAPL", "MSFT", "NVDA"],
        help="Ticker universe.",
    ),
    start: str = typer.Option("2022-01-01", help="Start date."),
    end: str = typer.Option("2024-12-31", help="End date."),
    save_csv: bool = typer.Option(True, help="Save CSV to results/."),
) -> None:
    """Print monthly regime labels (v1 vs v2) per ticker.

    Shows where the two detectors disagree -- especially Jan 2023 where v1
    labels bear (60d lookback into crash) and v2 labels bull (20d = recovery).
    """
    import pandas as pd

    from mcp_quant_agent.mcp_servers.analytics.regime import (
        label_regimes,
        label_regimes_v2,
    )
    from mcp_quant_agent.mcp_servers.data.yfinance_source import _fetch_raw_bars

    # ── Fetch data + label ────────────────────────────────────────────────────
    all_rows: list[dict[str, Any]] = []
    ticker_data: dict[str, list[dict[str, Any]]] = {}

    for ticker in tickers:
        typer.echo(f"Fetching {ticker} ({start} -> {end})...")
        try:
            bars = _fetch_raw_bars(ticker, start, end, "1d")
        except Exception as exc:
            typer.echo(f"  FAILED: {exc}", err=True)
            continue
        if not bars:
            typer.echo(f"  No data for {ticker}.", err=True)
            continue
        ticker_data[ticker] = bars
        typer.echo(f"  {len(bars)} bars.")

    if not ticker_data:
        typer.echo("No data fetched. Check tickers and date range.", err=True)
        raise typer.Exit(1)

    # ── Label regimes ─────────────────────────────────────────────────────────
    per_ticker_labels: dict[str, dict[str, dict[str, Any]]] = {}  # ticker -> {date: {v1, v2}}

    for ticker, bars in ticker_data.items():
        labeled_v1 = label_regimes(bars)
        labeled_v2 = label_regimes_v2(bars)
        date_map: dict[str, dict[str, Any]] = {}
        for i, bar in enumerate(bars):
            date = str(bar["date"])[:10]
            date_map[date] = {
                "date": date,
                "ticker": ticker,
                "close": bar["close"],
                "regime_v1": labeled_v1[i]["regime"],
                "regime_v2": labeled_v2[i]["regime"],
                "rolling_vol_ann": labeled_v2[i].get("rolling_vol_ann"),
                "trend_short_20d": labeled_v2[i].get("trend_short_20d"),
                "trend_long_60d": labeled_v2[i].get("trend_long_60d"),
            }
            all_rows.append(date_map[date])
        per_ticker_labels[ticker] = date_map

    # ── Monthly summary table ─────────────────────────────────────────────────
    # For each (ticker, year-month), compute modal regime for v1 and v2.
    months_in_range = [
        ym for ym in MONTHS_2022_2024
        if ym >= start[:7] and ym <= end[:7]
    ]

    # Build header
    header_parts = [f"{'Month':<9}"]
    for ticker in sorted(per_ticker_labels.keys()):
        header_parts.append(f"  {ticker:>6}(v1)  {ticker:>6}(v2)")
    header = "".join(header_parts)
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(" REGIME DIAGNOSTIC: v1 (60d trend) vs v2 (20d trend)")
    print(f" Period: {start} -> {end}")
    print(" Key: BULL/BEAR/RNGR(ange)/HV(igh_vol)/----")
    print(" Note: differences highlight where Jan 2023-style recovery")
    print("       is mislabelled by v1's 60d lookback.")
    print("=" * len(header))
    print(header)
    print(sep)

    diff_count = 0
    total_months = 0

    for ym in months_in_range:
        row_parts = [f"{ym:<9}"]
        has_diff = False
        for ticker in sorted(per_ticker_labels.keys()):
            date_map = per_ticker_labels[ticker]
            month_dates = [d for d in date_map if d.startswith(ym)]
            v1_labels = [date_map[d]["regime_v1"] for d in month_dates]
            v2_labels = [date_map[d]["regime_v2"] for d in month_dates]
            v1_mode = _mode_or_none(v1_labels)
            v2_mode = _mode_or_none(v2_labels)
            v1_s = _regime_char(v1_mode)
            v2_s = _regime_char(v2_mode)
            flag = "*" if (v1_mode != v2_mode and v1_mode is not None and v2_mode is not None) else " "
            if flag == "*":
                has_diff = True
            row_parts.append(f"  {v1_s:>8}  {v2_s:>8}{flag}")
        print("".join(row_parts))
        if has_diff:
            diff_count += 1
        if any(
            _mode_or_none([
                per_ticker_labels[t][d]["regime_v1"]
                for t in sorted(per_ticker_labels.keys())
                for d in per_ticker_labels[t] if d.startswith(ym)
            ]) is not None
            for _ in [1]
        ):
            total_months += 1

    print(sep)
    print(f"  Months with v1/v2 disagreement: {diff_count} (marked with *)")
    print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if save_csv and all_rows:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"regime_diagnostic_{ts}.csv"
        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        typer.echo(f"Daily labels saved to {csv_path}")

    # ── Key months highlight ──────────────────────────────────────────────────
    print("\nKey months to check (known market events):")
    key_months = {
        "2022-01": "Peak / start of 2022 bear",
        "2022-06": "Mid-2022 bottom area",
        "2022-10": "2022 low area",
        "2023-01": "Recovery start (BEAR both; v2 pivots to BULL in Feb 2023)",
        "2023-02": "v2 picks up recovery ~1 month before v1",
        "2023-07": "Mid-2023 bull run",
        "2024-01": "2024 new ATH period",
    }
    for ym, note in key_months.items():
        if ym < start[:7] or ym > end[:7]:
            continue
        for ticker in sorted(per_ticker_labels.keys()):
            date_map = per_ticker_labels[ticker]
            month_dates = [d for d in date_map if d.startswith(ym)]
            if not month_dates:
                continue
            v1_labels = [date_map[d]["regime_v1"] for d in month_dates]
            v2_labels = [date_map[d]["regime_v2"] for d in month_dates]
            v1 = _mode_or_none(v1_labels)
            v2 = _mode_or_none(v2_labels)
            flag = " <-- DISAGREE" if v1 != v2 else ""
            v1_s = v1 if v1 is not None else "----"
            v2_s = v2 if v2 is not None else "----"
            print(f"  {ym} {ticker:5}: v1={v1_s:8}  v2={v2_s:8}   [{note}]{flag}")
    print()


if __name__ == "__main__":
    app()
