"""Tests for financial performance metrics."""

from __future__ import annotations

import math

import pytest

from mcp_quant_agent.eval.financial import (
    annualised_return,
    bootstrap_sharpe_ci,
    calmar_ratio,
    compute_all_metrics,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def _flat_nav(n: int = 252, start: float = 100_000.0) -> list[float]:
    """NAV series with zero daily return."""
    return [start] * n


def _trending_nav(n: int = 252, daily_return: float = 0.001) -> list[float]:
    """NAV series with constant daily return."""
    nav = [100_000.0]
    for _ in range(n - 1):
        nav.append(nav[-1] * (1 + daily_return))
    return nav


def _volatile_nav(n: int = 252, seed: int = 42) -> list[float]:
    """NAV series with normally distributed daily returns."""
    import random

    rng = random.Random(seed)
    nav = [100_000.0]
    for _ in range(n - 1):
        nav.append(nav[-1] * (1 + rng.gauss(0.0, 0.01)))
    return nav


class TestAnnualisedReturn:
    def test_zero_return_flat_nav(self) -> None:
        assert annualised_return(_flat_nav()) == pytest.approx(0.0, abs=1e-9)

    def test_positive_trending(self) -> None:
        nav = _trending_nav(daily_return=0.001)
        ann = annualised_return(nav)
        assert ann > 0

    def test_approximate_value(self) -> None:
        # 252 days of +0.1% → total return ≈ (1.001)^252 - 1 ≈ 28.3%
        nav = _trending_nav(252, 0.001)
        ann = annualised_return(nav, trading_days=252)
        expected = (1.001**252) - 1.0
        assert abs(ann - expected) < 0.01

    def test_empty_nav_returns_zero(self) -> None:
        assert annualised_return([]) == 0.0

    def test_single_nav_returns_zero(self) -> None:
        assert annualised_return([100_000.0]) == 0.0


class TestMaxDrawdown:
    def test_flat_nav_zero_dd(self) -> None:
        assert max_drawdown(_flat_nav()) == pytest.approx(0.0)

    def test_known_drawdown(self) -> None:
        # 100 → 80 → peak 80 → drawdown = 20/100 = 0.20
        nav = [100.0, 90.0, 80.0, 85.0, 90.0]
        dd = max_drawdown(nav)
        assert abs(dd - 0.20) < 1e-9

    def test_always_increases(self) -> None:
        nav = _trending_nav(100, daily_return=0.005)
        assert max_drawdown(nav) == pytest.approx(0.0, abs=1e-6)

    def test_empty_returns_zero(self) -> None:
        assert max_drawdown([]) == 0.0


class TestSharpeRatio:
    def test_flat_nav_zero_sharpe(self) -> None:
        assert sharpe_ratio(_flat_nav()) == 0.0

    def test_trending_positive_sharpe(self) -> None:
        nav = _trending_nav(252, daily_return=0.001)
        assert sharpe_ratio(nav) > 0

    def test_volatile_finite_sharpe(self) -> None:
        nav = _volatile_nav()
        s = sharpe_ratio(nav)
        assert math.isfinite(s)


class TestSortinoRatio:
    def test_trending_positive_sortino(self) -> None:
        nav = _trending_nav(252, daily_return=0.001)
        assert sortino_ratio(nav) > 0

    def test_all_positive_returns_infinite(self) -> None:
        # No negative returns → infinite Sortino
        nav = _trending_nav(daily_return=0.001)
        s = sortino_ratio(nav)
        assert s == float("inf") or s > 0


class TestCalmarRatio:
    def test_positive_return_no_drawdown(self) -> None:
        nav = _trending_nav(daily_return=0.001)
        c = calmar_ratio(nav)
        assert c == float("inf") or c > 0

    def test_zero_return_zero_calmar(self) -> None:
        nav = _flat_nav()
        assert calmar_ratio(nav) == 0.0


class TestHitRate:
    def test_trending_high_hit_rate(self) -> None:
        nav = _trending_nav(252, daily_return=0.001)
        assert hit_rate(nav) == pytest.approx(1.0, abs=1e-9)

    def test_volatile_hit_rate_near_half(self) -> None:
        nav = _volatile_nav(500)
        hr = hit_rate(nav)
        assert 0.3 < hr < 0.7  # should be near 50% for zero-drift


class TestBootstrapSharpeCI:
    def test_ci_contains_point_estimate(self) -> None:
        nav = _volatile_nav(252)
        result = bootstrap_sharpe_ci(nav, n_bootstrap=500)
        assert result["ci_lower"] <= result["sharpe"] <= result["ci_upper"]

    def test_ci_width_positive(self) -> None:
        nav = _volatile_nav(252)
        result = bootstrap_sharpe_ci(nav, n_bootstrap=500)
        assert result["ci_upper"] > result["ci_lower"]


class TestComputeAllMetrics:
    def test_returns_all_keys(self) -> None:
        nav = _volatile_nav(252)
        result = compute_all_metrics(nav)
        for key in (
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "hit_rate",
            "annualised_return",
        ):
            assert key in result

    def test_metrics_finite(self) -> None:
        nav = _volatile_nav(252)
        result = compute_all_metrics(nav)
        for key, val in result.items():
            if key != "calmar":  # calmar can be inf
                assert math.isfinite(val), f"{key} is not finite"
