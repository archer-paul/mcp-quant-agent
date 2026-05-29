"""Tests for PM API smoke-test credit guardrails."""

from __future__ import annotations

import pytest

from mcp_quant_agent.backtest.pm_engine import PMBacktestEngine
from mcp_quant_agent.backtest.pm_smoke_guard import (
    validate_pm_api_smoke_request,
    validate_pm_multiday_request,
)


def _validate(**overrides: object) -> object:
    params = {
        "tickers": ["AAPL"],
        "start_date": "2023-01-03",
        "end_date": "2023-01-03",
        "model": "gpt-4.1-mini",
        "dev_model": "gpt-4.1-mini",
        "use_llm_cache": True,
        "acknowledge_cost": True,
    }
    params.update(overrides)
    return validate_pm_api_smoke_request(**params)  # type: ignore[arg-type]


def test_valid_pm_api_smoke_request_is_one_pm_chain() -> None:
    result = _validate(tickers=["msft", "AAPL", "AAPL"])

    assert result.tickers == ["AAPL", "MSFT"]  # type: ignore[attr-defined]
    assert result.estimated_pm_calls == 1  # type: ignore[attr-defined]
    assert result.estimated_llm_calls == 11  # type: ignore[attr-defined]


def test_rejects_more_than_two_tickers() -> None:
    with pytest.raises(RuntimeError, match="at most 2 tickers"):
        _validate(tickers=["AAPL", "MSFT", "NVDA"])


def test_rejects_more_than_one_date() -> None:
    with pytest.raises(RuntimeError, match="exactly 1 date"):
        _validate(end_date="2023-01-04")


def test_rejects_non_dev_model() -> None:
    with pytest.raises(RuntimeError, match="dev model"):
        _validate(model="gpt-4.1")


def test_rejects_disabled_cache() -> None:
    with pytest.raises(RuntimeError, match="cache enabled"):
        _validate(use_llm_cache=False)


def test_rejects_missing_cost_acknowledgement() -> None:
    with pytest.raises(RuntimeError, match="acknowledgement"):
        _validate(acknowledge_cost=False)


def test_pm_api_engine_requires_validated_smoke_guard() -> None:
    engine = PMBacktestEngine(
        tickers=["AAPL"],
        start_date="2023-01-03",
        end_date="2023-01-03",
        use_stub=False,
        price_data={"AAPL": []},
        news_data={"AAPL": []},
        write_artifacts=False,
    )

    with pytest.raises(RuntimeError, match="validated PMSmokeGuardResult"):
        engine.run()


def test_reference_run_allows_one_year_three_tickers() -> None:
    result = validate_pm_multiday_request(
        tickers=["AAPL", "MSFT", "NVDA"],
        start_date="2023-01-03",
        end_date="2023-12-29",
        model="gpt-4.1-mini",
        dev_model="gpt-4.1-mini",
        use_llm_cache=True,
        acknowledge_cost=True,
        reference_run=True,
    )

    assert result.tickers == ["AAPL", "MSFT", "NVDA"]
    assert result.n_trading_dates > 200


def test_non_reference_multiday_still_rejects_one_year() -> None:
    with pytest.raises(RuntimeError, match="at most 22 trading dates"):
        validate_pm_multiday_request(
            tickers=["AAPL", "MSFT", "NVDA"],
            start_date="2023-01-03",
            end_date="2023-12-29",
            model="gpt-4.1-mini",
            dev_model="gpt-4.1-mini",
            use_llm_cache=True,
            acknowledge_cost=True,
        )
