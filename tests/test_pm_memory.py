"""Tests for causal PM decision memory context."""

from __future__ import annotations

from pathlib import Path

from mcp_quant_agent.agents.pm_memory import PMDecisionLog


def test_pm_decision_log_returns_only_resolved_past_context() -> None:
    log = PMDecisionLog(log_path=None)
    log.store_decision(
        date="2023-02-13",
        weights={"NVDA": 0.7, "JPM": 0.25},
        cash_weight=0.05,
        rationale="Deployed risk after bullish debate.",
        regimes={"NVDA": "bull", "JPM": "range"},
    )

    assert log.get_past_context(t_now="2023-02-14") == ""

    log.update_with_returns(
        date="2023-02-13",
        realized_returns={"NVDA": 0.01, "JPM": -0.002},
    )
    log.store_decision(
        date="2023-02-15",
        weights={"AAPL": 0.2},
        cash_weight=0.8,
        rationale="Future decision must not leak.",
    )
    log.update_with_returns(
        date="2023-02-15",
        realized_returns={"AAPL": 0.03},
    )

    context = log.get_past_context(t_now="2023-02-14")

    assert "2023-02-13" in context
    assert "NVDA=70.0%" in context
    assert "2023-02-15" not in context
    assert "Future decision" not in context


def test_pm_decision_log_persists_and_reloads_resolved_entries(tmp_path: Path) -> None:
    path = tmp_path / "pm_memory.md"
    log = PMDecisionLog(log_path=path)
    log.store_decision(
        date="2023-02-13",
        weights={"NVDA": 0.5},
        cash_weight=0.5,
        rationale="Persisted decision.",
    )
    log.update_with_returns(
        date="2023-02-13",
        realized_returns={"NVDA": 0.02},
    )

    reloaded = PMDecisionLog(log_path=path)
    reloaded.load_from_file()
    context = reloaded.get_past_context(t_now="2023-02-14")

    assert "2023-02-13" in context
    assert "NVDA=50.0%" in context
    assert "realized: NVDA:+2.00%" in context
