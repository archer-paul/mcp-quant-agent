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


# ---------------------------------------------------------------------------
# Blocker #4 — PM sizing formula: floor(target_weight * NAV / price)
# ---------------------------------------------------------------------------


def test_rebalancer_buy_quantity_is_floor_target_weight_times_nav_over_price() -> None:
    """Rebalancer must buy floor((w * NAV - current_value) / price) shares.

    NAV=100_000, AAPL price=150, target=0.30 → target_value=30_000.
    Current holdings = 0 → buy floor(30_000 / 150) = 200 shares.
    """
    nav = 100_000.0
    price = 150.0
    target_weight = 0.30
    expected_qty = int(target_weight * nav / price)  # floor(30_000/150) = 200

    result = rebalance_to_targets(
        date="2023-01-03",
        portfolio_snapshot=_snapshot(cash=nav, nav=nav),
        current_prices={"AAPL": price},
        target_weights=PMTargetWeights(
            date="2023-01-03",
            weights={"AAPL": target_weight},
            rationale="sizing test",
        ),
    )

    buys = [o for o in result.orders if o.side == "buy"]
    assert buys, "Expected at least one buy order"
    assert buys[0].quantity == expected_qty, (
        f"Expected {expected_qty} shares, got {buys[0].quantity}"
    )
    assert buys[0].notional == pytest.approx(expected_qty * price)


def test_rebalancer_cash_zero_positions_at_target_produces_holds_not_freeze() -> None:
    """When cash=0 AND positions are already at target, rebalancer emits holds.

    This is the 'portfolio freeze' regression: the rebalancer must NOT try to
    buy with zero cash when positions are at target — it must produce hold orders.

    Scenario: cash=0, AAPL=50% of NAV (target 50%), MSFT=50% (target 50%).
    Diff is ~0 → no orders needed → holds.
    """
    nav = 30_000.0
    result = rebalance_to_targets(
        date="2023-01-03",
        portfolio_snapshot=_snapshot(
            cash=0.0,
            positions=[
                {"ticker": "AAPL", "quantity": 100, "market_value": 15_000.0},
                {"ticker": "MSFT", "quantity": 50, "market_value": 15_000.0},
            ],
            nav=nav,
        ),
        current_prices={"AAPL": 150.0, "MSFT": 300.0},
        target_weights=PMTargetWeights(
            date="2023-01-03",
            # Targets match current positions exactly → no trades needed
            weights={"AAPL": 0.50, "MSFT": 0.50},
            rationale="already at target — no trades",
        ),
    )
    buys = [o for o in result.orders if o.side == "buy"]
    sells = [o for o in result.orders if o.side == "sell"]
    assert not buys, f"Expected no buys when at target with no cash, got {buys}"
    assert not sells, f"Expected no sells when at target, got {sells}"
    assert result.cash_after_estimate >= 0.0


def test_rebalancer_sell_proceeds_fund_next_buy() -> None:
    """Sell proceeds from one ticker make cash available for the next buy.

    Scenario: AAPL over-weight → sell → cash increases → MSFT can be bought.
    """
    result = rebalance_to_targets(
        date="2023-01-03",
        portfolio_snapshot=_snapshot(
            cash=0.0,
            positions=[
                {"ticker": "AAPL", "quantity": 500, "market_value": 75_000.0},
            ],
            nav=75_000.0,
        ),
        current_prices={"AAPL": 150.0, "MSFT": 300.0},
        target_weights=PMTargetWeights(
            date="2023-01-03",
            weights={"AAPL": 0.40, "MSFT": 0.40},
            cash_weight=0.20,
            rationale="rebalance from concentrated AAPL",
        ),
    )

    sides = {o.ticker: o.side for o in result.orders}
    assert sides.get("AAPL") == "sell", "Expected AAPL to be trimmed"
    assert sides.get("MSFT") == "buy", "Expected MSFT to be bought with AAPL proceeds"
    assert result.cash_after_estimate >= 0.0


def test_rebalancer_partial_fill_when_cash_insufficient() -> None:
    """When cash covers only part of the target increase, partial fill is valid.

    100k NAV, AAPL target 60% (= 60k), cash only 10k → can buy at most 66 shares.
    """
    nav = 100_000.0
    price = 150.0
    cash = 10_000.0
    max_qty = int(cash / price)  # floor(10_000 / 150) = 66

    result = rebalance_to_targets(
        date="2023-01-03",
        portfolio_snapshot=_snapshot(
            cash=cash,
            positions=[
                {"ticker": "AAPL", "quantity": 200, "market_value": nav - cash},
            ],
            nav=nav,
        ),
        current_prices={"AAPL": price},
        target_weights=PMTargetWeights(
            date="2023-01-03",
            weights={"AAPL": 0.60},
            rationale="partial fill test",
        ),
    )

    buys = [o for o in result.orders if o.side == "buy"]
    if buys:
        assert buys[0].quantity <= max_qty, (
            f"Rebalancer bought {buys[0].quantity} > affordable {max_qty}"
        )
    assert result.cash_after_estimate >= 0.0
