"""Tests for the guarded PM API backbone parser."""

from __future__ import annotations

import pytest

from mcp_quant_agent.agents.pm_backbone import (
    _UNAVAILABLE_SUMMARY_PREFIX,
    _make_unavailable_news_reports,
    _news_item_counts,
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


# ---------------------------------------------------------------------------
# News analyst gating — Bug #1 regression tests
# ---------------------------------------------------------------------------


def test_news_item_counts_returns_zero_when_no_items() -> None:
    """_news_item_counts must return 0 for tickers with empty news tool output."""
    tool_outputs = [
        {"tool": "get_news_items_cache_first", "ticker": "AAPL", "items_count": 0, "max_timestamp": None},
        {"tool": "compute_indicators", "ticker": "AAPL", "values": {"sma_20": 168.9}},
    ]
    counts = _news_item_counts(tool_outputs, ["AAPL", "MSFT"])
    assert counts["AAPL"] == 0
    assert counts["MSFT"] == 0


def test_news_item_counts_detects_nonzero_items() -> None:
    """_news_item_counts returns the correct item count when news exists."""
    tool_outputs = [
        {"tool": "get_news_items_cache_first", "ticker": "NVDA", "items_count": 3, "max_timestamp": "2023-05-17"},
    ]
    counts = _news_item_counts(tool_outputs, ["NVDA"])
    assert counts["NVDA"] == 3


def test_make_unavailable_news_reports_returns_one_per_ticker() -> None:
    """Unavailable marker reports must be produced for all tickers."""
    reports = _make_unavailable_news_reports("2023-01-03", ["AAPL", "MSFT", "NVDA"])
    assert len(reports) == 3
    for report in reports:
        assert report.analyst == "news"
        assert report.confidence == 0.0
        assert report.signal == "neutral"
        assert report.summary.startswith(_UNAVAILABLE_SUMMARY_PREFIX)
        assert report.evidence == []


def test_unavailable_news_contributes_zero_to_consensus() -> None:
    """A news report with confidence=0 must not shift the PM consensus.

    This is the key invariant: if the news analyst is unavailable, it must be
    as if it was never called — no signal in either direction.
    """
    from mcp_quant_agent.eval.reasoning import _pm_analyst_consensus

    # Only news reports (unavailable), no technical or risk reports
    news_reports = _make_unavailable_news_reports("2023-01-03", ["AAPL"])
    reports_as_dicts = [r.model_dump() for r in news_reports]

    consensus = _pm_analyst_consensus(reports_as_dicts, "AAPL")
    assert consensus == 0.0, (
        f"Unavailable news report with confidence=0 must not shift consensus; "
        f"got {consensus}"
    )


def test_pm_parser_raises_attaches_raw_completion() -> None:
    """parse_pm_target_response must attach raw_completion to raised ValueError."""
    bad_raw = '{"weights": {"AAPL": 0.4}, "rationale": "missing cash"}'
    exc: ValueError | None = None
    try:
        parse_pm_target_response(bad_raw, date="2022-02-09", allowed_tickers=["AAPL"])
    except ValueError as e:
        exc = e
    assert exc is not None
    assert hasattr(exc, "_raw_completion"), "ValueError must have _raw_completion attribute"
    assert "missing cash" in str(exc._raw_completion) or bad_raw[:50] in str(exc._raw_completion)


def test_news_report_evidence_must_not_contain_technical_indicator_keywords() -> None:
    """Regression: news report evidence must NOT contain SMA/MACD/RSI/Bollinger.

    This is the anti-hallucination guard: if a news report's evidence contains
    technical indicator keywords, it was fabricated from the wrong data source.
    This test documents the expected contract. When the news gating is active
    and tool outputs are empty, the unavailable marker has no evidence at all.
    """
    reports = _make_unavailable_news_reports("2023-05-18", ["AAPL", "NVDA"])
    forbidden_keywords = {"SMA", "MACD", "RSI", "Bollinger", "EMA", "regime", "close", "price"}
    for report in reports:
        for ev_item in report.evidence:
            for kw in forbidden_keywords:
                assert kw.lower() not in ev_item.lower(), (
                    f"News unavailable report for {report.ticker} contains technical "
                    f"indicator keyword {kw!r} in evidence: {ev_item!r}"
                )
