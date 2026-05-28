"""Tests for the guarded PM API backbone parser."""

from __future__ import annotations

import pytest

from mcp_quant_agent.agents.pm_backbone import (
    parse_analyst_reports_response,
    parse_discussion_response,
    parse_pm_target_response,
)


def test_pm_parser_accepts_valid_target_json() -> None:
    targets = parse_pm_target_response(
        '{"weights": {"aapl": 0.4}, "cash_weight": 0.6, "rationale": "balanced"}',
        date="2022-02-09",
        allowed_tickers=["AAPL"],
    )

    assert targets.weights == {"AAPL": 0.4}
    assert targets.cash_weight == pytest.approx(0.6)
    assert targets.rationale == "balanced"


def test_pm_parser_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="negative"):
        parse_pm_target_response(
            '{"weights": {"AAPL": -0.1}, "cash_weight": 1.0, "rationale": "bad"}',
            date="2022-02-09",
            allowed_tickers=["AAPL"],
        )


def test_pm_parser_rejects_leverage() -> None:
    with pytest.raises(ValueError, match="exceed"):
        parse_pm_target_response(
            '{"weights": {"AAPL": 0.7}, "cash_weight": 0.4, "rationale": "bad"}',
            date="2022-02-09",
            allowed_tickers=["AAPL"],
        )


def test_pm_parser_rejects_unknown_ticker() -> None:
    with pytest.raises(ValueError, match="unknown tickers"):
        parse_pm_target_response(
            '{"weights": {"MSFT": 0.4}, "cash_weight": 0.6, "rationale": "bad"}',
            date="2022-02-09",
            allowed_tickers=["AAPL"],
        )


def test_pm_parser_rejects_missing_cash_weight() -> None:
    with pytest.raises(ValueError, match="cash_weight"):
        parse_pm_target_response(
            '{"weights": {"AAPL": 0.4}, "rationale": "bad"}',
            date="2022-02-09",
            allowed_tickers=["AAPL"],
        )


def test_analyst_parser_accepts_one_report_per_ticker() -> None:
    reports = parse_analyst_reports_response(
        (
            '{"reports": ['
            '{"ticker": "aapl", "signal": "bullish", "confidence": 0.7, '
            '"summary": "trend positive", "evidence": ["close=139.00"]}'
            "]}"
        ),
        date="2022-02-09",
        role="technical",
        allowed_tickers=["AAPL"],
    )

    assert len(reports) == 1
    assert reports[0].ticker == "AAPL"
    assert reports[0].analyst == "technical"
    assert reports[0].signal == "bullish"


def test_analyst_parser_rejects_missing_ticker_report() -> None:
    with pytest.raises(ValueError, match="omitted"):
        parse_analyst_reports_response(
            '{"reports": []}',
            date="2022-02-09",
            role="risk",
            allowed_tickers=["AAPL"],
        )


def test_discussion_parser_bounds_refs_to_allowed_tickers() -> None:
    turn = parse_discussion_response(
        '{"speaker": "risk", "message": "Prefer smaller risk.", '
        '"referenced_tickers": ["AAPL", "MSFT"]}',
        role="risk",
        allowed_tickers=["AAPL"],
    )

    assert turn["speaker"] == "risk"
    assert turn["referenced_tickers"] == ["AAPL"]


def test_discussion_parser_accepts_tradingagents_manager_shape() -> None:
    turn = parse_discussion_response(
        (
            '{"speaker": "research_manager", "rating": "Overweight", '
            '"investment_plan": "Build exposure gradually from the debate.", '
            '"referenced_tickers": ["AAPL"], "key_evidence": ["bull case stronger"]}'
        ),
        role="research_manager",
        allowed_tickers=["AAPL"],
        stage="research_manager",
    )

    assert turn["stage"] == "research_manager"
    assert turn["speaker"] == "research_manager"
    assert turn["rating"] == "Overweight"
    assert turn["message"] == "Build exposure gradually from the debate."
    assert turn["key_evidence"] == ["bull case stronger"]
