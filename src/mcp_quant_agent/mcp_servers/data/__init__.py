"""MCP data wrappers around external market-data sources.

Every function in this package enforces the t_now filter via the module-level
clock — it is structurally impossible for any returned bar or news item to be
dated after t_now.

Sub-modules
-----------
- ``cache``           — parquet disk cache keyed by (ticker, interval)
- ``yfinance_source`` — yfinance wrapper with t_now filter + cache
- ``finnhub_source``  — Finnhub wrapper with t_now filter
- ``server``          — FastMCP tool server exposing the above to MCP clients
"""
