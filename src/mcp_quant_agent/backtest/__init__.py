"""Backtest modules — baselines and event-driven engine.

Two complementary components:
- ``baselines`` — vectorised Buy&Hold / momentum / mean-reversion (vectorbt).
  These are your yardstick: run them BEFORE judging the agent.
- ``engine``    — event-driven drip-feed engine for the agent loop (bar by bar,
  advances the simulation clock, calls the orchestrator at each step).
"""
