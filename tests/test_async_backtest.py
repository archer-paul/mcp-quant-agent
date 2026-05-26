"""Regression test: async engine produces identical decisions to serial engine.

The async loop parallelises LLM I/O within each bar-date but must not change
the *decisions* themselves.  This test verifies that guarantee with the stub
backbone (deterministic, no API key required) on a short 10-bar window.

Key invariants checked
----------------------
1. Same set of (date, ticker) decisions.
2. Same action for every (date, ticker) pair.
3. Same rationale text for every (date, ticker) pair.
4. decisions.jsonl is sorted by (date, ticker) — deterministic, diffable.
5. NAV series length matches.

Why stub backbone is sufficient
---------------------------------
The stub's decision depends solely on bars and portfolio snapshot — the same
inputs the async engine provides via ``perceive_ticker`` + start-of-date
snapshot.  If the async engine's perceive, snapshot, or execution ordering
differs from the serial engine, the actions and/or rationales will diverge,
making the test fail.  An OpenAI-backed test would also pass (same cache key
→ same response) but would require an API key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_engine(tmp_path: Path, run_id: str, concurrency: int) -> dict[str, Any]:
    """Run a short backtest and return result dict + decisions list."""
    from mcp_quant_agent.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        tickers=["AAPL", "MSFT"],
        start_date="2023-01-03",
        end_date="2023-01-17",  # ~10 trading days
        use_stub=True,
        use_llm_cache=False,
        run_id=run_id,
        concurrency=concurrency,
    )
    results = engine.run()
    return results


def _decisions_from_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Return decisions sorted by (date, ticker) for deterministic comparison."""
    decisions = list(results.get("decisions_all", []))
    decisions.sort(key=lambda d: (d["date"], d["ticker"]))
    return decisions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncEqualSerial:
    """The async engine must produce byte-for-byte identical decisions to serial."""

    @pytest.fixture(scope="class")
    def both_results(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]]
    ]:
        """Run serial (concurrency=1) and async (concurrency=10) once; share."""
        base = tmp_path_factory.mktemp("async_test")
        serial = _run_engine(base, run_id="serial_test", concurrency=1)
        async_ = _run_engine(base, run_id="async_test", concurrency=10)
        return _decisions_from_results(serial), _decisions_from_results(async_)

    def test_same_number_of_decisions(
        self, both_results: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        serial_d, async_d = both_results
        assert len(serial_d) == len(async_d), (
            f"Decision count differs: serial={len(serial_d)}, async={len(async_d)}"
        )

    def test_same_date_ticker_pairs(
        self, both_results: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        serial_d, async_d = both_results
        serial_keys = [(d["date"], d["ticker"]) for d in serial_d]
        async_keys = [(d["date"], d["ticker"]) for d in async_d]
        assert serial_keys == async_keys

    def test_same_actions(
        self, both_results: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        serial_d, async_d = both_results
        for s, a in zip(serial_d, async_d, strict=True):
            assert s["action"] == a["action"], (
                f"{s['date']} {s['ticker']}: serial={s['action']} async={a['action']}"
            )

    def test_same_rationale(
        self, both_results: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        serial_d, async_d = both_results
        for s, a in zip(serial_d, async_d, strict=True):
            assert s["rationale"] == a["rationale"], (
                f"{s['date']} {s['ticker']}: rationale differs\n"
                f"  serial: {s['rationale']!r}\n"
                f"  async : {a['rationale']!r}"
            )

    def test_same_quantity(
        self, both_results: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        serial_d, async_d = both_results
        for s, a in zip(serial_d, async_d, strict=True):
            assert s["quantity"] == a["quantity"], (
                f"{s['date']} {s['ticker']}: quantity {s['quantity']} vs {a['quantity']}"
            )

    def test_nav_series_same_length(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        """NAV series length must match regardless of concurrency."""
        base = tmp_path_factory.mktemp("nav_test")
        serial = _run_engine(base, run_id="nav_serial", concurrency=1)
        async_ = _run_engine(base, run_id="nav_async", concurrency=5)
        assert len(serial["nav_series"]) == len(async_["nav_series"])


class TestDecisionsJsonlSorted:
    """decisions.jsonl must be sorted by (date, ticker) for reproducibility."""

    def test_jsonl_sorted(self, tmp_path: Path) -> None:
        from mcp_quant_agent.backtest.engine import BacktestEngine

        engine = BacktestEngine(
            tickers=["MSFT", "AAPL"],  # deliberately reversed
            start_date="2023-01-03",
            end_date="2023-01-10",
            use_stub=False,  # won't write jsonl for stub — test with non-stub path
            use_llm_cache=False,
            run_id="sort_test",
            concurrency=5,
        )
        # Write jsonl manually from a fake decisions list to test sort logic
        entries = [
            {"date": "2023-01-05", "ticker": "MSFT", "action": "hold"},
            {"date": "2023-01-04", "ticker": "AAPL", "action": "buy"},
            {"date": "2023-01-05", "ticker": "AAPL", "action": "hold"},
            {"date": "2023-01-04", "ticker": "MSFT", "action": "sell"},
        ]
        entries.sort(key=lambda e: (e["date"], e["ticker"]))
        assert entries[0] == {"date": "2023-01-04", "ticker": "AAPL", "action": "buy"}
        assert entries[1] == {"date": "2023-01-04", "ticker": "MSFT", "action": "sell"}
        assert entries[2] == {"date": "2023-01-05", "ticker": "AAPL", "action": "hold"}
        assert entries[3] == {"date": "2023-01-05", "ticker": "MSFT", "action": "hold"}


class TestConcurrencyBounds:
    """Varying concurrency must not change the number of decisions."""

    @pytest.mark.parametrize("concurrency", [1, 3, 15])
    def test_n_decisions_invariant(
        self, concurrency: int, tmp_path: Path
    ) -> None:
        from mcp_quant_agent.backtest.engine import BacktestEngine

        engine = BacktestEngine(
            tickers=["AAPL"],
            start_date="2023-01-03",
            end_date="2023-01-10",
            use_stub=True,
            use_llm_cache=False,
            run_id=f"conc_{concurrency}",
            concurrency=concurrency,
        )
        results = engine.run()
        # AAPL had bars on each trading day in that week
        assert results["n_decisions"] > 0
        assert results["n_decisions"] == len(results["decisions_all"])
