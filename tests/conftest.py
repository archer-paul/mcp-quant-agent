"""Shared pytest fixtures and configuration.

CRITICAL: The ``reset_global_clock`` autouse fixture resets the module-level
``_CLOCK`` to ``None`` before and after every test.  Without it, a test that
calls ``set_clock()`` would leak its clock state into subsequent tests, causing
non-deterministic failures and false positives on anti-look-ahead assertions.
"""

from __future__ import annotations

import datetime as dt

import pytest

import mcp_quant_agent.clock as _clock_module
from mcp_quant_agent.clock import SimulationClock

# ---------------------------------------------------------------------------
# Global clock isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_clock() -> None:  # type: ignore[return]
    """Reset the module-level clock to ``None`` before and after every test.

    This is the most important fixture in the test suite: it ensures that
    ``set_clock()`` calls in one test cannot affect another.  It is ``autouse``
    so it runs automatically for every test in every module.
    """
    _clock_module._CLOCK = None
    yield
    _clock_module._CLOCK = None


# ---------------------------------------------------------------------------
# Convenience clock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clock_june15() -> SimulationClock:
    """A SimulationClock set to 2022-06-15 (mid-backtest reference date)."""
    from mcp_quant_agent.clock import set_clock

    clock = SimulationClock("2022-06-15")
    set_clock(clock)
    return clock


@pytest.fixture()
def t_past() -> dt.datetime:
    """Timestamp one day before the reference date."""
    return dt.datetime(2022, 6, 14)


@pytest.fixture()
def t_future() -> dt.datetime:
    """Timestamp one day after the reference date."""
    return dt.datetime(2022, 6, 16)
