"""Guardrails for PM API runs.

This module does not call OpenAI. It validates that any PM API request
stays within the credit budget envelope agreed for Step 2.

Two guard tiers:
  - ``PMSmokeGuardResult``: original 1-date / 1-2 tickers guard (smoke only).
  - ``PMMultiDayGuardResult``: bounded multi-date guard (max 10 dates / 3 tickers).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from mcp_quant_agent.agents.pm_backbone import TRADINGAGENTS_STYLE_LLM_CALLS

_MAX_MULTIDAY_DATES = 10
_MAX_MULTIDAY_TICKERS = 3


@dataclass(frozen=True)
class PMSmokeGuardResult:
    """Validated PM API smoke-test request (1 date, 1-2 tickers)."""

    tickers: list[str]
    start_date: str
    end_date: str
    model: str
    use_llm_cache: bool
    estimated_pm_calls: int
    estimated_llm_calls: int


@dataclass(frozen=True)
class PMMultiDayGuardResult:
    """Validated multi-day bounded PM API request (max 10 dates, max 3 tickers).

    Same safety properties as PMSmokeGuardResult except the date range is
    extended to allow meaningful regime coverage.  Still requires dev model,
    LLM cache on, and explicit cost acknowledgement.
    """

    tickers: list[str]
    start_date: str
    end_date: str
    model: str
    use_llm_cache: bool
    estimated_pm_calls: int
    estimated_llm_calls: int
    n_trading_dates: int


def _validate_date(value: str, field_name: str) -> str:
    try:
        dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO date YYYY-MM-DD") from exc
    return value


def validate_pm_api_smoke_request(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model: str,
    dev_model: str,
    use_llm_cache: bool,
    acknowledge_cost: bool,
) -> PMSmokeGuardResult:
    """Validate the explicit guardrails for a future PM API smoke run.

    Required envelope:
    - exactly one decision date (`start_date == end_date`)
    - one or two tickers only
    - model equals the configured cheap/dev model
    - LLM cache is enabled
    - user explicitly acknowledges cost
    """
    start = _validate_date(start_date, "start_date")
    end = _validate_date(end_date, "end_date")
    if start != end:
        raise RuntimeError(
            "PM API smoke must be limited to exactly 1 date "
            f"(got {start}->{end})."
        )

    cleaned_tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    if not cleaned_tickers:
        raise RuntimeError("PM API smoke requires 1 or 2 tickers.")
    if len(cleaned_tickers) > 2:
        raise RuntimeError(
            "PM API smoke must use at most 2 tickers "
            f"(got {len(cleaned_tickers)}: {cleaned_tickers})."
        )

    if model != dev_model:
        raise RuntimeError(
            f"PM API smoke must use the configured dev model {dev_model!r}; "
            f"got {model!r}."
        )
    if not use_llm_cache:
        raise RuntimeError("PM API smoke must run with LLM cache enabled.")
    if not acknowledge_cost:
        raise RuntimeError(
            "PM API smoke requires explicit cost acknowledgement "
            "via --acknowledge-cost."
        )

    return PMSmokeGuardResult(
        tickers=cleaned_tickers,
        start_date=start,
        end_date=end,
        model=model,
        use_llm_cache=use_llm_cache,
        estimated_pm_calls=1,
        estimated_llm_calls=TRADINGAGENTS_STYLE_LLM_CALLS,
    )


def _count_weekdays(start: str, end: str) -> int:
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    count = 0
    d = s
    while d <= e:
        if d.weekday() < 5:
            count += 1
        d += dt.timedelta(days=1)
    return count


def validate_pm_multiday_request(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model: str,
    dev_model: str,
    use_llm_cache: bool,
    acknowledge_cost: bool,
) -> PMMultiDayGuardResult:
    """Validate a bounded multi-day PM API run (max 10 dates, max 3 tickers).

    Relaxes the 1-date constraint from ``validate_pm_api_smoke_request`` while
    keeping all other safety properties: dev model only, LLM cache on,
    explicit cost acknowledgement.

    The date range is validated but NOT required to be a single date.
    """
    start = _validate_date(start_date, "start_date")
    end = _validate_date(end_date, "end_date")
    if dt.date.fromisoformat(start) > dt.date.fromisoformat(end):
        raise RuntimeError(f"start_date must be <= end_date (got {start} > {end}).")

    n_dates = _count_weekdays(start, end)
    if n_dates > _MAX_MULTIDAY_DATES:
        raise RuntimeError(
            f"PM multi-day run must span at most {_MAX_MULTIDAY_DATES} trading dates "
            f"(got ~{n_dates} between {start} and {end}). "
            "Use run_thesis_backtest.py for full production runs."
        )

    cleaned_tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    if not cleaned_tickers:
        raise RuntimeError("PM multi-day run requires at least 1 ticker.")
    if len(cleaned_tickers) > _MAX_MULTIDAY_TICKERS:
        raise RuntimeError(
            f"PM multi-day run must use at most {_MAX_MULTIDAY_TICKERS} tickers "
            f"(got {len(cleaned_tickers)}: {cleaned_tickers})."
        )

    if model != dev_model:
        raise RuntimeError(
            f"PM multi-day run must use the configured dev model {dev_model!r}; "
            f"got {model!r}."
        )
    if not use_llm_cache:
        raise RuntimeError("PM multi-day run must run with LLM cache enabled.")
    if not acknowledge_cost:
        raise RuntimeError(
            "PM multi-day run requires explicit cost acknowledgement "
            "via --acknowledge-cost."
        )

    est_pm_calls = n_dates
    est_llm_calls = n_dates * TRADINGAGENTS_STYLE_LLM_CALLS
    return PMMultiDayGuardResult(
        tickers=cleaned_tickers,
        start_date=start,
        end_date=end,
        model=model,
        use_llm_cache=use_llm_cache,
        estimated_pm_calls=est_pm_calls,
        estimated_llm_calls=est_llm_calls,
        n_trading_dates=n_dates,
    )
