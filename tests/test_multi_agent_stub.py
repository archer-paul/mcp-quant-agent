"""Tests for the deterministic PM multi-agent stub graph."""

from __future__ import annotations

import datetime as dt

from mcp_quant_agent.agents.multi_agent import parse_pm_stub_output, run_pm_stub


def _bars(start: str = "2022-01-01", n: int = 35) -> list[dict[str, object]]:
    start_d = dt.date.fromisoformat(start)
    return [
        {
            "date": (start_d + dt.timedelta(days=i)).isoformat(),
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1_000_000,
        }
        for i in range(n)
    ]


def _market(close: float = 140.0, sma20: float = 120.0) -> dict[str, object]:
    return {
        "bars": _bars(),
        "news": [
            {
                "datetime": "2022-02-04T12:00:00",
                "headline": "Company reports strong growth and raises guidance",
            }
        ],
        "indicators": {
            "close": close,
            "sma_20": sma20,
            "rsi_14": 61.0,
            "macd": 1.2,
        },
        "regime": "bull",
    }


def test_pm_stub_receives_all_tickers_and_outputs_parseable_targets(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    state = run_pm_stub(
        date="2022-02-04",
        tickers=["AAPL", "MSFT"],
        market_data={"AAPL": _market(), "MSFT": _market(close=90.0, sma20=110.0)},
        portfolio={"cash": 100_000.0, "positions": [], "nav": 100_000.0},
    )
    reports, targets = parse_pm_stub_output(state)

    assert state["pm_received_tickers"] == ["AAPL", "MSFT"]
    assert len(reports) == 6
    assert targets.gross_weight + targets.cash_weight <= 1.0 + 1e-9
    assert set(targets.weights).issubset({"AAPL", "MSFT"})


def test_reports_are_structured_and_bounded() -> None:
    state = run_pm_stub(
        date="2022-02-04",
        tickers=["AAPL"],
        market_data={"AAPL": _market()},
        portfolio={"cash": 100_000.0, "positions": [], "nav": 100_000.0},
    )
    reports, _ = parse_pm_stub_output(state)

    assert {report.analyst for report in reports} == {"technical", "news", "risk"}
    assert all(len(report.summary) <= 500 for report in reports)
    assert all(len(report.evidence) <= 5 for report in reports)


def test_pm_stub_has_no_fixed_20_percent_cap_for_single_winner() -> None:
    state = run_pm_stub(
        date="2022-02-04",
        tickers=["AAPL"],
        market_data={"AAPL": _market()},
        portfolio={"cash": 100_000.0, "positions": [], "nav": 100_000.0},
    )
    _, targets = parse_pm_stub_output(state)

    assert targets.weights["AAPL"] > 0.20
