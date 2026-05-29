"""Tests for the paper-trading Portfolio engine."""

from __future__ import annotations

import pytest

from mcp_quant_agent.mcp_servers.execution.paper_trading import Portfolio


class TestPortfolio:
    @pytest.fixture(autouse=True)
    def portfolio(self) -> Portfolio:  # type: ignore[return]
        self.p = Portfolio()
        return self.p

    def test_initial_state(self) -> None:
        assert self.p.cash == 100_000.0
        assert self.p.positions == {}
        assert self.p.order_history == []

    def test_buy_reduces_cash(self) -> None:
        fill = self.p.place_order(
            "AAPL", "buy", 10, 150.0, simulation_time="2022-06-15"
        )
        assert self.p.cash == pytest.approx(100_000.0 - 10 * 150.0)
        assert fill.side == "buy"

    def test_buy_creates_position(self) -> None:
        self.p.place_order("AAPL", "buy", 10, 150.0, simulation_time="2022-06-15")
        assert self.p.positions["AAPL"] == pytest.approx(10.0)

    def test_cost_basis_after_buy(self) -> None:
        self.p.place_order("AAPL", "buy", 10, 150.0, simulation_time="2022-06-15")
        assert self.p.cost_basis["AAPL"] == pytest.approx(150.0)

    def test_sell_increases_cash(self) -> None:
        self.p.place_order("AAPL", "buy", 10, 150.0, simulation_time="2022-06-15")
        self.p.place_order("AAPL", "sell", 5, 160.0, simulation_time="2022-06-16")
        assert self.p.cash == pytest.approx(100_000.0 - 10 * 150.0 + 5 * 160.0)

    def test_sell_more_than_held_raises(self) -> None:
        self.p.place_order("AAPL", "buy", 5, 150.0, simulation_time="2022-06-15")
        with pytest.raises(ValueError, match="sell"):
            self.p.place_order("AAPL", "sell", 10, 150.0, simulation_time="2022-06-15")

    def test_buy_insufficient_cash_raises(self) -> None:
        with pytest.raises(ValueError, match="cash"):
            self.p.place_order(
                "AAPL", "buy", 10_000, 150.0, simulation_time="2022-06-15"
            )

    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side"):
            self.p.place_order("AAPL", "HOLD", 10, 150.0, simulation_time="2022-06-15")

    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            self.p.place_order("AAPL", "buy", 0, 150.0, simulation_time="2022-06-15")

    def test_sell_all_removes_position(self) -> None:
        self.p.place_order("AAPL", "buy", 10, 150.0, simulation_time="2022-06-15")
        self.p.place_order("AAPL", "sell", 10, 160.0, simulation_time="2022-06-16")
        assert "AAPL" not in self.p.positions

    def test_portfolio_summary_nav(self) -> None:
        self.p.place_order("AAPL", "buy", 10, 150.0, simulation_time="2022-06-15")
        summary = self.p.get_portfolio_summary({"AAPL": 160.0})
        expected_nav = self.p.cash + 10 * 160.0
        assert summary["nav"] == pytest.approx(expected_nav)

    def test_order_history_length(self) -> None:
        self.p.place_order("AAPL", "buy", 10, 150.0, simulation_time="2022-06-15")
        self.p.place_order("AAPL", "sell", 5, 155.0, simulation_time="2022-06-16")
        assert len(self.p.order_history) == 2

    def test_reset(self) -> None:
        self.p.place_order("AAPL", "buy", 10, 150.0, simulation_time="2022-06-15")
        self.p.reset(initial_cash=50_000.0)
        assert self.p.cash == 50_000.0
        assert self.p.positions == {}
        assert self.p.order_history == []

    def test_record_nav(self) -> None:
        self.p.place_order("AAPL", "buy", 10, 150.0, simulation_time="2022-06-15")
        nav = self.p.record_nav({"AAPL": 160.0}, timestamp="2022-06-15")
        assert nav == pytest.approx(self.p.cash + 10 * 160.0)
        assert len(self.p.nav_history) == 1
        assert self.p.nav_history[0]["nav"] == nav


class TestTransactionCosts:
    """Transaction cost symmetry tests — blocker §2."""

    def test_zero_commission_is_legacy_default(self) -> None:
        """Portfolio() with no args has commission_bps=0 (backward compat)."""
        p = Portfolio()
        fill = p.place_order("AAPL", "buy", 10, 100.0, simulation_time="2022-01-03")
        assert fill.commission == pytest.approx(0.0)
        assert p.total_commission == pytest.approx(0.0)
        assert p.cash == pytest.approx(100_000.0 - 10 * 100.0)

    def test_buy_commission_deducted_from_cash(self) -> None:
        """10 bps on a $1000 buy = $0.10 commission."""
        p = Portfolio(commission_bps=10.0)
        notional = 10 * 100.0  # = 1000
        commission = notional * 10.0 / 10_000.0  # = 0.10 bps * notional = 1.0
        fill = p.place_order("AAPL", "buy", 10, 100.0, simulation_time="2022-01-03")
        assert fill.commission == pytest.approx(commission)
        assert p.cash == pytest.approx(100_000.0 - notional - commission)
        assert p.total_commission == pytest.approx(commission)
        assert p.total_turnover == pytest.approx(notional)

    def test_sell_commission_deducted_from_proceeds(self) -> None:
        """Commission on sell reduces cash received."""
        p = Portfolio(commission_bps=10.0)
        p.place_order("AAPL", "buy", 10, 100.0, simulation_time="2022-01-03")
        cash_after_buy = p.cash
        sell_notional = 10 * 100.0
        sell_commission = sell_notional * 10.0 / 10_000.0
        p.place_order("AAPL", "sell", 10, 100.0, simulation_time="2022-01-04")
        expected_cash = cash_after_buy + sell_notional - sell_commission
        assert p.cash == pytest.approx(expected_cash)

    def test_turnover_zero_means_zero_commission(self) -> None:
        """No trades → zero commission, zero turnover."""
        p = Portfolio(commission_bps=10.0)
        assert p.total_commission == pytest.approx(0.0)
        assert p.total_turnover == pytest.approx(0.0)

    def test_commission_accumulates_across_trades(self) -> None:
        """total_commission accumulates across multiple buys."""
        p = Portfolio(commission_bps=10.0)
        p.place_order("AAPL", "buy", 10, 100.0, simulation_time="2022-01-03")
        p.place_order("MSFT", "buy", 5, 200.0, simulation_time="2022-01-03")
        expected_comm = (10 * 100.0 + 5 * 200.0) * 10.0 / 10_000.0
        assert p.total_commission == pytest.approx(expected_comm)

    def test_10bps_matches_baselines_cost_per_trade(self) -> None:
        """The default settings.transaction_cost_bps must equal baselines COST_PER_TRADE."""
        from mcp_quant_agent.backtest.baselines import (
            COMMISSION_BPS,
            COST_PER_TRADE,
            SLIPPAGE_BPS,
        )
        from mcp_quant_agent.config import settings

        assert settings.transaction_cost_bps == pytest.approx(
            COMMISSION_BPS + SLIPPAGE_BPS
        ), (
            "settings.transaction_cost_bps must equal COMMISSION_BPS + SLIPPAGE_BPS "
            "for symmetric agent vs baseline comparison."
        )
        assert pytest.approx(settings.transaction_cost_bps / 10_000.0) == COST_PER_TRADE
