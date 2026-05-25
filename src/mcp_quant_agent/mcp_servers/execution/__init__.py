"""MCP execution server — paper-trading order management (in-memory).

Adapted from Salvi reference (``gen-ai-imperial/code/week3_agent/paper_trading.py``).
Additions:
- Orders are stamped with t_now from the simulation clock (not wall-clock time),
  so the trade history is consistent with the backtest timeline.
- NAV time series is tracked for equity-curve computation.
- Full type annotations.
"""
