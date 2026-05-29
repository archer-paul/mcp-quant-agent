"""Backtest tier system — defines the three scale levels.

Tiers
-----
smoke   -- Quick sanity check (days).  Used during development and CI.
           Max 3 tickers, 5 trading dates. Cache makes re-runs free.
medium  -- ~1 month of real decisions.  First real-LLM evaluation with
           transaction costs and regime coverage.
           Max 3 tickers, 22 trading dates.
full    -- The canonical 2-year thesis run (NEVER during development).
           Unrestricted date range; requires explicit --tier full confirmation.

Design rationale
----------------
Keeping the tier in one place prevents the 380/130/422 proliferation: any
script that validates a user's --tier flag just imports this module and
checks ``TIER_LIMITS[tier]``.

Warmup note
-----------
All tiers require a data window = decision_window + REGIME_WARMUP_CALENDAR_DAYS
BEFORE the first decision date to ensure n_warmup_excluded == 0.  This is
enforced by the engine (BacktestEngine, PMBacktestEngine) which always
pre-fetches REGIME_WARMUP_CALENDAR_DAYS of warmup bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TierName = Literal["smoke", "medium", "full"]


@dataclass(frozen=True)
class TierLimits:
    """Hard limits for a backtest tier."""

    max_tickers: int
    max_trading_dates: int
    requires_confirmation: bool
    description: str


TIER_LIMITS: dict[TierName, TierLimits] = {
    "smoke": TierLimits(
        max_tickers=3,
        max_trading_dates=5,
        requires_confirmation=False,
        description="Quick sanity check (days). Cache makes re-runs free.",
    ),
    "medium": TierLimits(
        max_tickers=3,
        max_trading_dates=22,
        requires_confirmation=True,
        description="~1 month of real decisions. First real-LLM eval with costs.",
    ),
    "full": TierLimits(
        max_tickers=10,
        max_trading_dates=9999,
        requires_confirmation=True,
        description=(
            "Canonical 2-year thesis run. NEVER during development. "
            "Requires explicit --acknowledge-cost and user review."
        ),
    ),
}


def validate_tier(
    tier: TierName,
    n_tickers: int,
    n_trading_dates: int,
    acknowledge_cost: bool = False,
) -> None:
    """Raise ValueError if the requested run exceeds the tier limits.

    Parameters
    ----------
    tier:
        Requested tier ('smoke', 'medium', or 'full').
    n_tickers:
        Number of tickers in the run.
    n_trading_dates:
        Approximate number of trading dates (weekdays) in the run window.
    acknowledge_cost:
        Required for 'medium' and 'full' tiers.

    Raises
    ------
    ValueError
        If any limit is exceeded.
    """
    limits = TIER_LIMITS[tier]

    if limits.requires_confirmation and not acknowledge_cost:
        raise ValueError(
            f"Tier '{tier}' requires --acknowledge-cost to confirm OpenAI spend."
        )

    if n_tickers > limits.max_tickers:
        raise ValueError(
            f"Tier '{tier}' allows max {limits.max_tickers} tickers, "
            f"got {n_tickers}."
        )

    if n_trading_dates > limits.max_trading_dates:
        raise ValueError(
            f"Tier '{tier}' allows max {limits.max_trading_dates} trading dates, "
            f"got ~{n_trading_dates}. "
            f"Use --tier full for longer windows (with explicit cost acknowledgement)."
        )
