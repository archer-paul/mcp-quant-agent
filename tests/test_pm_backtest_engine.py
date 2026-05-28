"""Tests for the PM multi-agent backtest engine."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from mcp_quant_agent.agents.pm_backbone import PMDiscussionResult
from mcp_quant_agent.agents.pm_schemas import AnalystReport, PMTargetWeights
from mcp_quant_agent.backtest.pm_engine import PMBacktestEngine
from mcp_quant_agent.backtest.pm_smoke_guard import validate_pm_api_smoke_request
from mcp_quant_agent.eval.reasoning import compute_grounding


def _make_price_rows(
    *,
    start: str,
    n: int,
    start_close: float,
    step: float,
) -> list[dict[str, Any]]:
    start_d = dt.date.fromisoformat(start)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        close = start_close + step * i
        rows.append(
            {
                "date": (start_d + dt.timedelta(days=i)).isoformat(),
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return rows


def _price_data() -> dict[str, list[dict[str, Any]]]:
    return {
        "AAPL": _make_price_rows(
            start="2022-01-01",
            n=45,
            start_close=100.0,
            step=1.0,
        ),
        "MSFT": _make_price_rows(
            start="2022-01-01",
            n=45,
            start_close=150.0,
            step=-0.5,
        ),
    }


def _news_data() -> dict[str, list[dict[str, Any]]]:
    return {
        "AAPL": [
            {
                "datetime": "2022-02-02T09:00:00",
                "headline": "Apple reports strong growth and raises guidance",
                "summary": "",
                "source": "test",
                "url": "past",
                "ticker": "AAPL",
            },
            {
                "datetime": "2022-03-01T09:00:00",
                "headline": "Future headline must not leak",
                "summary": "",
                "source": "test",
                "url": "future",
                "ticker": "AAPL",
            },
        ],
        "MSFT": [],
    }


def _run_engine(write_artifacts: bool = False, output_root: Path | None = None) -> dict[str, Any]:
    engine = PMBacktestEngine(
        tickers=["AAPL", "MSFT"],
        start_date="2022-02-01",
        end_date="2022-02-05",
        price_data=_price_data(),
        news_data=_news_data(),
        write_artifacts=write_artifacts,
        output_root=output_root or Path("."),
        run_id="pm_test",
    )
    return engine.run()


def test_pm_engine_stub_runs_without_openai_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    results = _run_engine()

    assert results["backbone"] == "pm_stub"
    assert results["n_decisions"] == 5
    assert len(results["nav_series"]) == 5
    assert results["order_history"]


def test_pm_engine_decisions_are_sorted_and_enriched() -> None:
    results = _run_engine()
    decisions = results["decisions_all"]

    keys = [(entry["date"], entry["ticker"]) for entry in decisions]
    assert keys == sorted(keys)
    first = decisions[0]
    assert first["mode"] == "multi_agent_pm"
    assert first["reports"]
    assert first["rationale"]
    assert first["targets"]["weights"]
    assert first["orders"]
    assert first["tool_outputs"]
    assert first["mcp_calls"]


def test_pm_engine_excludes_future_news_from_tool_outputs() -> None:
    results = _run_engine()

    headlines = [
        item["headline"]
        for entry in results["decisions_all"]
        for output in entry["tool_outputs"]
        if output.get("tool") == "get_news_items_cache_first"
        for item in output.get("items_recent", [])
    ]

    assert "Future headline must not leak" not in headlines


def test_pm_engine_is_deterministic_ignoring_latency() -> None:
    first = _run_engine()["decisions_all"]
    second = _run_engine()["decisions_all"]

    def stable_projection(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "date": entry["date"],
                "targets": entry["targets"],
                "orders": entry["orders"],
                "fills": entry["fills"],
                "nav_after": entry["nav_after"],
            }
            for entry in entries
        ]

    assert stable_projection(first) == stable_projection(second)


def test_pm_engine_writes_decisions_jsonl(tmp_path: Path) -> None:
    results = _run_engine(write_artifacts=True, output_root=tmp_path)
    jsonl_path = tmp_path / "runs" / results["run_id"] / "decisions.jsonl"
    summary_path = tmp_path / "results" / results["run_id"] / "summary.csv"
    memory_path = tmp_path / "runs" / results["run_id"] / "pm_decision_log.md"

    assert jsonl_path.exists()
    assert summary_path.exists()
    assert memory_path.exists()
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == results["n_decisions"]
    assert rows[0]["reports"]
    assert "ret=" in memory_path.read_text(encoding="utf-8")


def test_pm_engine_loads_prices_from_existing_cache_without_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_quant_agent.mcp_servers.data.cache import PriceCache

    cache = PriceCache(cache_dir=tmp_path)
    cached_rows = _make_price_rows(
        start="2022-01-01",
        n=45,
        start_close=100.0,
        step=1.0,
    )
    cache.write("AAPL", "1d", cached_rows)

    def fail_fetch(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        raise AssertionError("PM engine should use the existing price cache first")

    monkeypatch.setattr(
        "mcp_quant_agent.mcp_servers.data.yfinance_source._get_cache",
        lambda: cache,
    )
    monkeypatch.setattr(
        "mcp_quant_agent.mcp_servers.data.yfinance_source._fetch_raw_bars",
        fail_fetch,
    )

    engine = PMBacktestEngine(
        tickers=["AAPL"],
        start_date="2022-02-05",
        end_date="2022-02-09",
        write_artifacts=False,
    )
    loaded = engine._load_price_data()

    assert loaded["AAPL"]
    assert engine._price_sources["AAPL"] == "cache"


class _MockPMBackbone:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_cache_hit = True

    def decide(self, **kwargs: Any) -> PMDiscussionResult:
        self.calls.append(kwargs)
        date = str(kwargs["date"])
        reports = [
            AnalystReport(
                date=date,
                ticker="AAPL",
                analyst="technical",
                signal="bullish",
                confidence=0.8,
                summary="close is above trend",
                evidence=["close=139.00", "sma20=128.50"],
            ),
            AnalystReport(
                date=date,
                ticker="AAPL",
                analyst="news",
                signal="neutral",
                confidence=0.4,
                summary="limited news signal",
                evidence=[],
            ),
            AnalystReport(
                date=date,
                ticker="AAPL",
                analyst="risk",
                signal="neutral",
                confidence=0.5,
                summary="risk budget permits a position",
                evidence=["nav=100000.00"],
            ),
        ]
        return PMDiscussionResult(
            reports=reports,
            discussion=[
                {
                    "speaker": "risk",
                    "message": "Sizing should retain cash while allowing AAPL exposure.",
                    "referenced_tickers": ["AAPL"],
                }
            ],
            targets=PMTargetWeights(
                date=date,
                weights={"AAPL": 0.5},
                cash_weight=0.5,
                rationale=(
                    "The recent close (139.00) and NAV at 100000.00 are grounded; "
                    "a 50.00 percent target keeps cash available."
                ),
            ),
        )


def _one_day_guard() -> object:
    return validate_pm_api_smoke_request(
        tickers=["AAPL"],
        start_date="2022-02-09",
        end_date="2022-02-09",
        model="gpt-4.1-mini",
        dev_model="gpt-4.1-mini",
        use_llm_cache=True,
        acknowledge_cost=True,
    )


def test_pm_api_smoke_path_runs_with_mock_backbone_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backbone = _MockPMBackbone()
    engine = PMBacktestEngine(
        tickers=["AAPL"],
        start_date="2022-02-09",
        end_date="2022-02-09",
        price_data={
            "AAPL": _make_price_rows(
                start="2022-01-01",
                n=40,
                start_close=100.0,
                step=1.0,
            )
        },
        news_data={
            "AAPL": [
                {
                    "datetime": "2022-02-08T09:00:00",
                    "headline": "Apple reports strong growth",
                    "summary": "",
                    "source": "test",
                    "url": "past",
                    "ticker": "AAPL",
                },
                {
                    "datetime": "2022-02-10T09:00:00",
                    "headline": "Future headline must not leak",
                    "summary": "",
                    "source": "test",
                    "url": "future",
                    "ticker": "AAPL",
                },
            ]
        },
        use_stub=False,
        model="gpt-4.1-mini",
        smoke_guard=_one_day_guard(),
        pm_backbone=backbone,
        write_artifacts=False,
        output_root=tmp_path,
        run_id="pm_api_mock",
    )

    results = engine.run()

    assert backbone.calls
    assert results["backbone"] == "gpt-4.1-mini"
    assert results["n_decisions"] == 1
    decision = results["decisions_all"][0]
    assert decision["rationale"]
    assert decision["targets"]["weights"] == {"AAPL": 0.5}
    assert decision["tool_outputs"]
    assert decision["mcp_calls"]
    assert decision["reports"]
    assert decision["discussion"]
    assert decision["orders"]
    assert decision["fills"]
    assert decision["indicators"]

    jsonl_path = tmp_path / "runs" / "pm_api_mock" / "decisions.jsonl"
    summary_path = tmp_path / "results" / "pm_api_mock" / "summary.csv"
    assert jsonl_path.exists()
    assert summary_path.exists()

    for call in decision["mcp_calls"]:
        max_ts = call.get("max_timestamp")
        if max_ts:
            assert str(max_ts)[:10] <= "2022-02-09"

    headlines = [
        item["headline"]
        for output in decision["tool_outputs"]
        if output.get("tool") == "get_news_items_cache_first"
        for item in output.get("items_recent", [])
    ]
    assert "Future headline must not leak" not in headlines

    grounding = compute_grounding([decision])
    assert grounding["n_claims_total"] >= 2
    assert grounding["n_grounded"] >= 2


def test_pm_decision_log_causal_chain_on_3_dates(tmp_path: Path) -> None:
    """PMDecisionLog must build a strictly causal chain over 3 consecutive dates.

    Anti-lookahead contract:
    - After date D1 is processed, it is stored as "pending" (no realized return yet).
    - At date D2 start, D1's realized return is computed and D1 becomes resolved.
    - get_past_context(t_now="D2") returns only D1 (resolved, date < D2).
    - At date D3 start, D2 is resolved.
    - get_past_context(t_now="D3") returns D1 and D2.
    - pm_decision_log.md must have zero entries with "pending" status after the run
      (the last entry stays pending since D4 never arrives).
    """
    # 3 dates: 2022-02-07, 2022-02-08, 2022-02-09
    engine = PMBacktestEngine(
        tickers=["AAPL"],
        start_date="2022-02-07",
        end_date="2022-02-09",
        price_data={
            "AAPL": _make_price_rows(
                start="2022-01-01",
                n=45,
                start_close=100.0,
                step=1.0,
            )
        },
        news_data={"AAPL": []},
        allow_empty_news=True,
        write_artifacts=True,
        output_root=tmp_path,
        run_id="causal_chain_test",
    )
    results = engine.run()
    assert results["n_decisions"] == 3, "Expected 3 PM decisions (3 dates)"

    log_path = tmp_path / "runs" / "causal_chain_test" / "pm_decision_log.md"
    assert log_path.exists(), "pm_decision_log.md should be written when write_artifacts=True"

    raw = log_path.read_text(encoding="utf-8")

    # The first 2 entries should be resolved (ret= tag), the last is still pending
    # because no D4 was processed to provide D3's realized return.
    pending_count = raw.count("| pending]")
    resolved_count = raw.count("| ret=")
    assert pending_count == 1, f"Expected exactly 1 pending entry (last date), got {pending_count}"
    assert resolved_count == 2, f"Expected 2 resolved entries (D1+D2), got {resolved_count}"

    # Dates must appear in chronological order in the file
    pos_07 = raw.find("2022-02-07")
    pos_08 = raw.find("2022-02-08")
    pos_09 = raw.find("2022-02-09")
    assert pos_07 < pos_08 < pos_09, "Log entries should be chronological"

    # Anti-lookahead: the resolved entries' ret= values must reflect
    # the correct price direction (AAPL goes up by 1.0/day → positive returns)
    from mcp_quant_agent.agents.pm_memory import PMDecisionLog
    log = PMDecisionLog(log_path=log_path)
    log.load_from_file()
    resolved = [e for e in log._entries if not e["pending"]]
    assert len(resolved) == 2
    for entry in resolved:
        returns = entry["realized_returns"]
        if returns:  # AAPL is always in the portfolio by then
            aapl_ret = returns.get("AAPL", 0.0)
            # Price goes up 1.0 per day from 100.0 → ~1% return per day
            assert aapl_ret > 0, f"Expected positive return, got {aapl_ret}"


def test_pm_api_smoke_parse_failure_records_explicit_all_cash_error(tmp_path: Path) -> None:
    class FailingBackbone:
        last_cache_hit = False

        def decide(self, **_kwargs: Any) -> PMTargetWeights:
            raise ValueError("bad PM JSON")

    engine = PMBacktestEngine(
        tickers=["AAPL"],
        start_date="2022-02-09",
        end_date="2022-02-09",
        price_data={
            "AAPL": _make_price_rows(
                start="2022-01-01",
                n=40,
                start_close=100.0,
                step=1.0,
            )
        },
        news_data={"AAPL": []},
        use_stub=False,
        model="gpt-4.1-mini",
        smoke_guard=_one_day_guard(),
        pm_backbone=FailingBackbone(),
        output_root=tmp_path,
        run_id="pm_api_parse_error",
    )

    results = engine.run()
    decision = results["decisions_all"][0]

    assert decision["targets"]["weights"] == {}
    assert decision["targets"]["cash_weight"] == 1.0
    assert "pm_api_error" in decision["rationale"]
    assert any("pm_api_error" in err for err in decision["errors"])
