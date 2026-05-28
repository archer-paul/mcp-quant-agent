"""Deterministic rebalancer for PM target weights.

The rebalancer is intentionally not an agent. It converts validated PM target
weights into buy/sell/hold orders while enforcing the structural constraints:
long-only, no shorting, no leverage, non-negative cash.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from mcp_quant_agent.agents.pm_schemas import (
    PMTargetWeights,
    RebalanceOrder,
    RebalanceResult,
)

logger = logging.getLogger(__name__)


def build_target_weights(
    *,
    date: str,
    weights: Mapping[str, float],
    cash_weight: float = 0.0,
    rationale: str = "",
    normalize: bool = True,
) -> PMTargetWeights:
    """Build validated PM target weights, optionally normalising over-allocation.

    This is the boundary used for untrusted PM output. The schema itself stays
    strict (`sum(weights) + cash <= 1`), while this helper can repair an
    over-allocated raw proposal and records the adjustment.
    """
    cleaned = {str(ticker).upper(): float(weight) for ticker, weight in weights.items()}
    cash = float(cash_weight)
    total = sum(cleaned.values()) + cash
    adjustments: list[str] = []

    if total > 1.0 + 1e-9:
        if not normalize:
            return PMTargetWeights(
                date=date,
                weights=cleaned,
                cash_weight=cash,
                rationale=rationale,
            )
        scale = 1.0 / total
        cleaned = {ticker: weight * scale for ticker, weight in cleaned.items()}
        cash *= scale
        adjustment = (
            f"normalised target weights by {scale:.6f} because raw total "
            f"{total:.6f} exceeded 1.0"
        )
        adjustments.append(adjustment)
        logger.info("PM target adjustment on %s: %s", date, adjustment)

    return PMTargetWeights(
        date=date,
        weights=cleaned,
        cash_weight=cash,
        rationale=rationale,
        adjustments=adjustments,
    )


def _positions_from_snapshot(portfolio_snapshot: Mapping[str, Any]) -> dict[str, float]:
    positions: dict[str, float] = {}
    for pos in portfolio_snapshot.get("positions", []):
        if not isinstance(pos, Mapping):
            continue
        ticker = str(pos.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        positions[ticker] = positions.get(ticker, 0.0) + float(pos.get("quantity", 0.0))
    return positions


def _market_values(
    quantities: Mapping[str, float],
    current_prices: Mapping[str, float],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for ticker, quantity in quantities.items():
        price = float(current_prices[ticker])
        values[ticker] = quantity * price
    return values


def _require_prices(tickers: set[str], current_prices: Mapping[str, float]) -> None:
    missing = sorted(ticker for ticker in tickers if ticker not in current_prices)
    if missing:
        raise ValueError(f"missing current prices for rebalancing: {missing}")
    non_positive = [
        ticker for ticker in tickers
        if float(current_prices[ticker]) <= 0.0
    ]
    if non_positive:
        raise ValueError(f"non-positive current prices for rebalancing: {non_positive}")


def _hold_order(
    *,
    date: str,
    ticker: str,
    price: float,
    current_weight: float,
    target_weight: float,
    reason: str,
) -> RebalanceOrder:
    return RebalanceOrder(
        date=date,
        ticker=ticker,
        side="hold",
        quantity=0,
        price=price,
        notional=0.0,
        current_weight=current_weight,
        target_weight=target_weight,
        reason=reason,
    )


def rebalance_to_targets(
    *,
    date: str,
    portfolio_snapshot: Mapping[str, Any],
    current_prices: Mapping[str, float],
    target_weights: PMTargetWeights | Mapping[str, float],
    cash_weight: float = 0.0,
    min_trade_notional: float = 100.0,
) -> RebalanceResult:
    """Convert target weights into executable deterministic orders.

    Parameters
    ----------
    date:
        Decision date.
    portfolio_snapshot:
        Portfolio summary from `Portfolio.get_portfolio_summary`.
    current_prices:
        Mapping of ticker to the current close at `t_now`.
    target_weights:
        Valid PMTargetWeights or raw ticker weights. Raw overweight proposals
        are normalised before order generation.
    cash_weight:
        Used only when `target_weights` is a raw mapping.
    min_trade_notional:
        Orders below this notional become holds.
    """
    if min_trade_notional < 0:
        raise ValueError("min_trade_notional must be non-negative")

    targets = (
        target_weights
        if isinstance(target_weights, PMTargetWeights)
        else build_target_weights(
            date=date,
            weights=target_weights,
            cash_weight=cash_weight,
            rationale="raw target mapping",
            normalize=True,
        )
    )

    quantities = _positions_from_snapshot(portfolio_snapshot)
    tickers = set(quantities) | set(targets.weights)
    _require_prices(tickers, current_prices)

    cash = float(portfolio_snapshot.get("cash", 0.0))
    current_values = _market_values(quantities, current_prices)
    nav = float(portfolio_snapshot.get("nav", cash + sum(current_values.values())))
    if nav < 0:
        raise ValueError(f"portfolio NAV must be non-negative, got {nav}")
    if nav == 0 or not tickers:
        return RebalanceResult(
            date=date,
            targets=targets,
            orders=[],
            adjustments=list(targets.adjustments),
            nav=max(nav, 0.0),
            cash_before=max(cash, 0.0),
            cash_after_estimate=max(cash, 0.0),
        )

    orders_by_ticker: dict[str, RebalanceOrder] = {}
    cash_after = cash

    def current_weight(ticker: str) -> float:
        return current_values.get(ticker, 0.0) / nav if nav > 0 else 0.0

    # Sells first, so buys can use sale proceeds without ever going negative.
    for ticker in sorted(tickers):
        price = float(current_prices[ticker])
        held_qty = max(0.0, quantities.get(ticker, 0.0))
        current_value = held_qty * price
        target_weight = float(targets.weights.get(ticker, 0.0))
        target_value = target_weight * nav
        diff = target_value - current_value
        weight_before = current_weight(ticker)

        if diff >= -min_trade_notional:
            continue

        requested_qty = int(math.floor(abs(diff) / price))
        sell_qty = min(int(math.floor(held_qty)), requested_qty)
        notional = round(sell_qty * price, 4)
        if sell_qty <= 0 or notional < min_trade_notional:
            orders_by_ticker[ticker] = _hold_order(
                date=date,
                ticker=ticker,
                price=price,
                current_weight=weight_before,
                target_weight=target_weight,
                reason="sell difference below executable share/min_trade threshold",
            )
            continue

        quantities[ticker] = held_qty - sell_qty
        current_values[ticker] = quantities[ticker] * price
        cash_after += notional
        orders_by_ticker[ticker] = RebalanceOrder(
            date=date,
            ticker=ticker,
            side="sell",
            quantity=sell_qty,
            price=price,
            notional=notional,
            current_weight=weight_before,
            target_weight=target_weight,
            reason="reduce position toward PM target weight",
        )

    # Buys second, capped by cash available after sells.
    for ticker in sorted(tickers):
        if ticker in orders_by_ticker:
            continue

        price = float(current_prices[ticker])
        held_qty = max(0.0, quantities.get(ticker, 0.0))
        current_value = held_qty * price
        target_weight = float(targets.weights.get(ticker, 0.0))
        target_value = target_weight * nav
        diff = target_value - current_value
        weight_before = current_weight(ticker)

        if diff < min_trade_notional:
            orders_by_ticker[ticker] = _hold_order(
                date=date,
                ticker=ticker,
                price=price,
                current_weight=weight_before,
                target_weight=target_weight,
                reason="target difference below min_trade_notional",
            )
            continue

        desired_qty = int(math.floor(diff / price))
        affordable_qty = int(math.floor(cash_after / price))
        buy_qty = min(desired_qty, affordable_qty)
        notional = round(buy_qty * price, 4)
        if buy_qty <= 0 or notional < min_trade_notional:
            orders_by_ticker[ticker] = _hold_order(
                date=date,
                ticker=ticker,
                price=price,
                current_weight=weight_before,
                target_weight=target_weight,
                reason="insufficient cash or share granularity for min trade",
            )
            continue

        cash_after -= notional
        if cash_after < -1e-7:
            raise AssertionError("rebalancer generated negative cash")
        quantities[ticker] = held_qty + buy_qty
        current_values[ticker] = quantities[ticker] * price
        orders_by_ticker[ticker] = RebalanceOrder(
            date=date,
            ticker=ticker,
            side="buy",
            quantity=buy_qty,
            price=price,
            notional=notional,
            current_weight=weight_before,
            target_weight=target_weight,
            reason="increase position toward PM target weight",
        )

    orders = [orders_by_ticker[ticker] for ticker in sorted(orders_by_ticker)]
    return RebalanceResult(
        date=date,
        targets=targets,
        orders=orders,
        adjustments=list(targets.adjustments),
        nav=round(nav, 4),
        cash_before=round(max(cash, 0.0), 4),
        cash_after_estimate=round(max(cash_after, 0.0), 4),
    )
