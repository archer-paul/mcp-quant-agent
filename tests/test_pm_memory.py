"""Tests for PMDecisionLog — causal context injection, persistence, anti-lookahead."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from mcp_quant_agent.agents.pm_memory import PMDecisionLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def log() -> PMDecisionLog:
    """In-memory log (no file I/O)."""
    return PMDecisionLog(log_path=None)


@pytest.fixture()
def log_on_disk(tmp_path: Path) -> PMDecisionLog:
    """Log that persists to a temp file."""
    return PMDecisionLog(log_path=tmp_path / "decisions.md")


# ---------------------------------------------------------------------------
# Basic write / read
# ---------------------------------------------------------------------------


def test_empty_log_returns_no_context(log: PMDecisionLog) -> None:
    assert log.get_past_context() == ""


def test_pending_decision_not_in_context(log: PMDecisionLog) -> None:
    log.store_decision(
        date="2023-01-03",
        weights={"AAPL": 0.5},
        cash_weight=0.5,
        rationale="bull signal",
    )
    assert log.get_past_context() == ""


def test_resolved_decision_appears_in_context(log: PMDecisionLog) -> None:
    log.store_decision(
        date="2023-01-03",
        weights={"AAPL": 0.5},
        cash_weight=0.5,
        rationale="bull signal",
    )
    log.update_with_returns(date="2023-01-03", realized_returns={"AAPL": 0.012})
    ctx = log.get_past_context()
    assert "2023-01-03" in ctx
    assert "AAPL" in ctx


def test_context_contains_realized_return(log: PMDecisionLog) -> None:
    log.store_decision(
        date="2023-01-03",
        weights={"AAPL": 0.5},
        cash_weight=0.5,
        rationale="test",
    )
    log.update_with_returns(date="2023-01-03", realized_returns={"AAPL": 0.025})
    ctx = log.get_past_context()
    assert "+2.50%" in ctx


def test_multiple_decisions_most_recent_first(log: PMDecisionLog) -> None:
    for i in range(3):
        date = f"2023-01-0{i + 3}"
        log.store_decision(
            date=date,
            weights={"AAPL": 0.5},
            cash_weight=0.5,
            rationale=f"day {i}",
        )
        log.update_with_returns(date=date, realized_returns={"AAPL": 0.01 * i})
    ctx = log.get_past_context()
    pos_03 = ctx.index("2023-01-03")
    pos_05 = ctx.index("2023-01-05")
    assert pos_05 < pos_03  # most recent (05) appears before oldest (03)


# ---------------------------------------------------------------------------
# Anti-lookahead: t_now filter
# ---------------------------------------------------------------------------


def test_context_excludes_same_day_entries(log: PMDecisionLog) -> None:
    """A resolved decision on t_now must NOT appear in the context."""
    log.store_decision(
        date="2023-01-05",
        weights={"AAPL": 0.5},
        cash_weight=0.5,
        rationale="same day",
    )
    log.update_with_returns(date="2023-01-05", realized_returns={"AAPL": 0.01})
    ctx = log.get_past_context(t_now="2023-01-05")
    assert ctx == ""


def test_context_includes_strictly_past_entries(log: PMDecisionLog) -> None:
    log.store_decision(
        date="2023-01-04",
        weights={"AAPL": 0.5},
        cash_weight=0.5,
        rationale="yesterday",
    )
    log.update_with_returns(date="2023-01-04", realized_returns={"AAPL": 0.005})
    ctx = log.get_past_context(t_now="2023-01-05")
    assert "2023-01-04" in ctx


def test_context_excludes_future_entries(log: PMDecisionLog) -> None:
    for date in ("2023-01-03", "2023-01-04", "2023-01-09"):
        log.store_decision(
            date=date,
            weights={"AAPL": 0.5},
            cash_weight=0.5,
            rationale="test",
        )
        log.update_with_returns(date=date, realized_returns={"AAPL": 0.01})
    ctx = log.get_past_context(t_now="2023-01-05")
    assert "2023-01-09" not in ctx
    assert "2023-01-03" in ctx
    assert "2023-01-04" in ctx


def test_no_context_leaks_future_entry_without_t_now(log: PMDecisionLog) -> None:
    """Without t_now, all resolved entries appear — caller should always pass t_now."""
    for date in ("2023-01-03", "2099-12-31"):
        log.store_decision(
            date=date,
            weights={"AAPL": 0.5},
            cash_weight=0.5,
            rationale="test",
        )
        log.update_with_returns(date=date, realized_returns={"AAPL": 0.01})
    ctx_all = log.get_past_context(t_now=None)
    assert "2099-12-31" in ctx_all  # no filter → future appears (expected)
    ctx_filtered = log.get_past_context(t_now="2023-01-05")
    assert "2099-12-31" not in ctx_filtered  # with t_now → future excluded


# ---------------------------------------------------------------------------
# Window capping (n_recent)
# ---------------------------------------------------------------------------


def test_n_recent_caps_context_size(log: PMDecisionLog) -> None:
    for i in range(10):
        date = (dt.date(2023, 1, 3) + dt.timedelta(days=i)).isoformat()
        log.store_decision(
            date=date,
            weights={"AAPL": 0.5},
            cash_weight=0.5,
            rationale=f"day {i}",
        )
        log.update_with_returns(date=date, realized_returns={"AAPL": 0.001 * i})
    ctx = log.get_past_context(n_recent=3)
    dates_in_ctx = [
        (dt.date(2023, 1, 3) + dt.timedelta(days=i)).isoformat()
        for i in range(10)
        if (dt.date(2023, 1, 3) + dt.timedelta(days=i)).isoformat() in ctx
    ]
    assert len(dates_in_ctx) <= 3


# ---------------------------------------------------------------------------
# Disk persistence and round-trip
# ---------------------------------------------------------------------------


def test_store_and_load_from_file(tmp_path: Path) -> None:
    log1 = PMDecisionLog(log_path=tmp_path / "decisions.md")
    log1.store_decision(
        date="2023-02-13",
        weights={"JPM": 0.25, "NVDA": 0.70},
        cash_weight=0.05,
        rationale="two-ticker allocation",
        regimes={"JPM": "bull", "NVDA": "bull"},
    )
    log1.update_with_returns(
        date="2023-02-13",
        realized_returns={"JPM": 0.012, "NVDA": -0.005},
    )

    log2 = PMDecisionLog(log_path=tmp_path / "decisions.md")
    log2.load_from_file()
    ctx = log2.get_past_context()
    assert "2023-02-13" in ctx
    assert "JPM" in ctx
    assert "NVDA" in ctx


def test_file_has_pending_tag_before_update(log_on_disk: PMDecisionLog, tmp_path: Path) -> None:
    log_on_disk.store_decision(
        date="2023-01-05",
        weights={"AAPL": 0.6},
        cash_weight=0.4,
        rationale="test",
    )
    raw = (tmp_path / "decisions.md").read_text(encoding="utf-8")
    assert "pending" in raw


def test_file_has_ret_tag_after_update(log_on_disk: PMDecisionLog, tmp_path: Path) -> None:
    log_on_disk.store_decision(
        date="2023-01-05",
        weights={"AAPL": 0.6},
        cash_weight=0.4,
        rationale="test",
    )
    log_on_disk.update_with_returns(
        date="2023-01-05",
        realized_returns={"AAPL": 0.02},
    )
    raw = (tmp_path / "decisions.md").read_text(encoding="utf-8")
    assert "pending" not in raw
    assert "ret=" in raw


def test_atomic_write_no_tmp_file_left(log_on_disk: PMDecisionLog, tmp_path: Path) -> None:
    log_on_disk.store_decision(
        date="2023-01-05",
        weights={"AAPL": 0.5},
        cash_weight=0.5,
        rationale="test",
    )
    log_on_disk.update_with_returns(
        date="2023-01-05",
        realized_returns={"AAPL": 0.01},
    )
    tmp_file = tmp_path / "decisions.tmp"
    assert not tmp_file.exists(), "Temp file should have been renamed away"


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotation_drops_oldest_resolved(tmp_path: Path) -> None:
    log = PMDecisionLog(log_path=tmp_path / "decisions.md", max_entries=3)
    for i in range(6):
        date = (dt.date(2023, 1, 3) + dt.timedelta(days=i)).isoformat()
        log.store_decision(
            date=date,
            weights={"AAPL": 0.5},
            cash_weight=0.5,
            rationale=f"day {i}",
        )
        log.update_with_returns(date=date, realized_returns={"AAPL": 0.001 * i})

    raw = (tmp_path / "decisions.md").read_text(encoding="utf-8")
    assert "2023-01-03" not in raw   # oldest 3 should be dropped
    assert "2023-01-08" in raw       # most recent should remain


# ---------------------------------------------------------------------------
# All-cash allocation
# ---------------------------------------------------------------------------


def test_all_cash_allocation_shows_in_context(log: PMDecisionLog) -> None:
    log.store_decision(
        date="2023-01-03",
        weights={},
        cash_weight=1.0,
        rationale="insufficient evidence",
    )
    log.update_with_returns(date="2023-01-03", realized_returns={})
    ctx = log.get_past_context()
    assert "all-cash" in ctx


# ---------------------------------------------------------------------------
# Regime info in context
# ---------------------------------------------------------------------------


def test_regime_info_appears_in_context(log: PMDecisionLog) -> None:
    log.store_decision(
        date="2023-01-03",
        weights={"NVDA": 0.5},
        cash_weight=0.5,
        rationale="bull run",
        regimes={"NVDA": "bull"},
    )
    log.update_with_returns(date="2023-01-03", realized_returns={"NVDA": 0.03})
    ctx = log.get_past_context()
    assert "bull" in ctx
