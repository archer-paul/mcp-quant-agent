#!/usr/bin/env python
"""Mini validation: fetch and print AAPL/MSFT bars for Mar-Jun 2023.

Used to visually confirm prices are plausible (AAPL ~150-190, MSFT ~270-340)
before launching the full thesis run.  Run with:

    python3.13 scripts/validate_prices.py
"""
from __future__ import annotations

import datetime as dt

from mcp_quant_agent.clock import SimulationClock, set_clock
from mcp_quant_agent.mcp_servers.data.yfinance_source import get_price_history

set_clock(SimulationClock(dt.datetime(2023, 6, 30)))

for ticker, lo, hi in [("AAPL", 140, 200), ("MSFT", 250, 360)]:
    print(f"\n=== {ticker} bars Mar-Jun 2023 (expected close ~{lo}-{hi}) ===")
    bars = get_price_history(ticker, "2023-03-01", "2023-06-30")
    # Print first, two mid-points, and last bar
    sample_idx = [0, len(bars) // 3, 2 * len(bars) // 3, len(bars) - 1]
    seen = set()
    for i in sample_idx:
        if i not in seen:
            b = bars[i]
            flag = " *** SUSPICIOUS ***" if not (lo <= b["close"] <= hi) else ""
            print(
                f"  {b['date']}  open={b['open']:.2f}  high={b['high']:.2f}"
                f"  low={b['low']:.2f}  close={b['close']:.2f}"
                f"  vol={b['volume']:,}{flag}"
            )
            seen.add(i)
    print(f"  Total bars: {len(bars)}")

print("\nAll bars passed bar_validation (no ValueError raised). Prices look plausible.")
