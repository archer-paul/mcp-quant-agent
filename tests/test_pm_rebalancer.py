"""Tests for the deterministic PM rebalancer."""

from __future__ import annotations

import pytest

from mcp_quant_agent.agents.pm_schemas import PMTargetWeights
from mcp_quant_agent.backtest.rebalancer import (
    build_target_weights,
    rebalance_to_targets,
)


def _snapshot(
    *,
    cash: float = 100_000.0,
    positions: list[dict[str, object]] | None = None,
    nav: float | None = None,
) -> dict[str, object]:
    pos = positions or []
    if nav is None:
        nav = cash + sum(float(p["market_value"]) for p in pos)
    return {"cash": cash, "positions": pos, "nav": nav}


def test_pm_target_schema_allows_position_above_20_percent() -> None:
    targets = PMTargetWeights(
        date="2022-06-15",
        weights={"AAPL": 0.75},
        cash_weight=0.25,
        rationale="single high-conviction allocation",
    )

    assert targets.weights["AAPL"] == pytest.approx(0.75)


def test_rebalancer_buys_to_valid_weights_without_20_percent_cap() -> None:
    result = rebalance_to_targets(
        date="2022-06-15",
        portfolio_snapshot=_snapshot(cash=100_000.0, nav=100_000.0),
        current_prices={"AAPL": 100.0},
        target_weights=PMTargetWeights(
            date="2022-06-15",
            weights={"AAPL": 0.75},
            rationale="no PM-mode cap",
        ),
    )

    order = result.orders[0]
    assert order.side == "buy"
    assert order.quantity == 750
    assert order.notional == pytest.approx(75_000.0)
    assert result.cash_after_estimate == pytest.approx(25_000.0)


def test_rebalancer_never_generates_negative_cash() -> None:
    result = rebalance_to_targets(
        date="2022-06-15",
        portfolio_snapshot=_snapshot(cash=5_000.0, nav=100_000.0),
        current_prices={"AAPL": 100.0, "MSFT": 50.0},
        target_weights=PMTargetWeights(
            date="2022-06-15",
            weights={"AAPL": 0.5, "MSFT": 0.5},
            rationale="cash-limited buy",
        ),
    )

    assert result.cash_after_estimate >= 0.0
    assert sum(o.notional for o in result.orders if o.side == "buy") <= 5_000.0


def test_rebalancer_does_not_short_on_sell() -> None:
    result = rebalance_to_targets(
        date="2022-06-15",
        portfolio_snapshot=_snapshot(
            cash=0.0,
            positions=[
                {
                    "ticker": "AAPL",
                    "quantity": 10,
                    "market_value": 1_000.0,
                }
            ],
            nav=1_000.0,
        ),
        current_prices={"AAPL": 100.0},
        target_weights=PMTargetWeights(
            date="2022-06-15",
            weights={"AAPL": 0.0},
            rationale="exit",
        ),
    )

    order = result.orders[0]
    assert order.side == "sell"
    assert order.quantity == 10
    assert order.notional == pytest.approx(1_000.0)


def test_min_trade_notional_turns_small_diff_into_hold() -> None:
    result = rebalance_to_targets(
        date="2022-06-15",
        portfolio_snapshot=_snapshot(cash=100_000.0, nav=100_000.0),
        current_prices={"AAPL": 100.0},
        target_weights=PMTargetWeights(
            date="2022-06-15",
            weights={"AAPL": 0.0005},
            rationale="dust allocation",
        ),
        min_trade_notional=100.0,
    )

    assert result.orders[0].side == "hold"
    assert result.orders[0].quantity == 0


def test_overallocated_raw_weights_are_normalised_and_logged() -> None:
    targets = build_target_weights(
        date="2022-06-15",
        weights={"AAPL": 0.8, "MSFT": 0.8},
        rationale="raw PM output",
        normalize=True,
    )

    assert sum(targets.weights.values()) <= 1.0
    assert targets.weights["AAPL"] == pytest.approx(0.5)
    assert targets.weights["MSFT"] == pytest.approx(0.5)
    assert targets.adjustments


def test_schema_rejects_negative_or_levered_weights() -> None:
    with pytest.raises(ValueError, match="negative"):
        PMTargetWeights(date="2022-06-15", weights={"AAPL": -0.1})

    with pytest.raises(ValueError, match="exceed"):
        PMTargetWeights(date="2022-06-15", weights={"AAPL": 0.7}, cash_weight=0.4)
