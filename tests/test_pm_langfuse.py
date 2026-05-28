"""Langfuse PM logging stays best-effort."""

from __future__ import annotations

from mcp_quant_agent.observability import langfuse_setup


def test_pm_langfuse_logging_noops_without_client(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(langfuse_setup, "get_langfuse_client", lambda: None)

    langfuse_setup.log_pm_decision(
        run_id="pm_api_test",
        t_now="2022-02-09",
        model="gpt-4.1-mini",
        tickers=["AAPL"],
        reports=[],
        discussion=[],
        portfolio={"cash": 100000.0, "nav": 100000.0, "positions": []},
        tool_outputs=[],
        mcp_calls=[],
        rationale="all cash",
        target_weights={"weights": {}, "cash_weight": 1.0},
        orders=[],
        fills=[],
    )
