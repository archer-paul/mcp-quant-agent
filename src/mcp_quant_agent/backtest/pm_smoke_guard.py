"""Guardrails for future PM API smoke tests.

This module does not call OpenAI. It validates that any future PM API smoke
request stays within the credit budget envelope agreed for Step 2.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from mcp_quant_agent.agents.pm_backbone import TRADINGAGENTS_STYLE_LLM_CALLS


@dataclass(frozen=True)
class PMSmokeGuardResult:
    """Validated PM API smoke-test request."""

    tickers: list[str]
    start_date: str
    end_date: str
    model: str
    use_llm_cache: bool
    estimated_pm_calls: int
    estimated_llm_calls: int


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
