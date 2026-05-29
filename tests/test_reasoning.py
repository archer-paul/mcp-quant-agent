"""Tests for eval/reasoning.py — faithfulness (LLM judge), grounding, sophistication.

All faithfulness tests use _judge_fn to inject a deterministic mock judge so no
API key is needed.  Grounding and intent-extraction tests are purely algorithmic.
"""

from __future__ import annotations

from mcp_quant_agent.eval.reasoning import (
    _extract_intent,
    _extract_numeric_claims,
    _ground_claim,
    _is_constrained_hold,
    compute_faithfulness_llm,
    compute_grounding,
    compute_mcp_time_machine_audit,
    compute_pm_evidence_grounding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision(
    action: str = "hold",
    rationale: str = "",
    regime: str = "bull",
    indicators: dict[str, object] | None = None,
    tool_outputs: list[dict[str, object]] | None = None,
    fill: dict[str, object] | None = None,
    ticker: str = "AAPL",
    date: str = "2023-01-05",
    # Portfolio position for this ticker (expressed as pct_of_nav, qty)
    position_pct: float = 0.0,
    position_qty: float = 0.0,
    nav: float = 100_000.0,
    cash: float = 100_000.0,
) -> dict[str, object]:
    """Build a minimal decision dict for testing."""
    positions = []
    if position_pct > 0 or position_qty > 0:
        positions = [
            {
                "ticker": ticker,
                "quantity": position_qty,
                "market_value": round(nav * position_pct, 2),
                "pct_of_nav": position_pct,
            }
        ]
    # Build tool_outputs with raw portfolio snapshot (new schema)
    if tool_outputs is None:
        tool_outputs = [
            {
                "tool": "get_portfolio",
                "cash": cash,
                "nav": nav,
                "positions": positions,
            }
        ]
    return {
        "date": date,
        "ticker": ticker,
        "action": action,
        "quantity": 0 if action == "hold" else 50,
        "rationale": rationale,
        "fill": fill,
        "regime": regime,
        "indicators": indicators or {},
        "tool_outputs": tool_outputs,
        "latency_ms": 100.0,
        "errors": [],
    }


# ---------------------------------------------------------------------------
# 1. _extract_intent tests (legacy utility — kept for reference)
# ---------------------------------------------------------------------------


class TestExtractIntent:
    def test_buy_signal(self) -> None:
        assert _extract_intent("The trend is up, I should buy now.") == "buy"

    def test_sell_signal(self) -> None:
        assert _extract_intent("Price is overbought, best to sell.") == "sell"

    def test_hold_signal(self) -> None:
        assert _extract_intent("Uncertainty is high, holding is prudent.") == "hold"

    def test_no_signal_returns_none(self) -> None:
        assert _extract_intent("The market is complex and multifaceted.") is None

    def test_mixed_hold_wins_tie(self) -> None:
        text = "I might buy but I'll hold for now and hold my position."
        assert _extract_intent(text) == "hold"

    def test_case_insensitive(self) -> None:
        assert _extract_intent("Strong uptrend — BUY signal confirmed.") == "buy"

    def test_hold_context_dominant(self) -> None:
        text = "I will not buy at this level; staying put."
        assert _extract_intent(text) == "hold"


# ---------------------------------------------------------------------------
# 2. compute_faithfulness_llm tests (LLM-judge version, mocked)
# ---------------------------------------------------------------------------


class TestComputeFaithfulnessLLM:
    """All tests inject _judge_fn so no OpenAI key is needed."""

    def test_faithful_buy(self) -> None:
        """Judge says buy, agent bought → faithful."""
        d = _decision(action="buy", rationale="Uptrend confirmed.")
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "buy")
        assert result["faithfulness"] == 1.0
        assert result["n_faithful"] == 1
        assert result["n_unfaithful"] == 0

    def test_faithful_hold(self) -> None:
        """Judge says hold, agent held → faithful."""
        d = _decision(action="hold", rationale="Mixed signals, staying put.")
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "hold")
        assert result["faithfulness"] == 1.0

    def test_faithful_sell(self) -> None:
        """Judge says sell, agent sold → faithful."""
        d = _decision(action="sell", rationale="Bearish crossover.")
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "sell")
        assert result["faithfulness"] == 1.0

    def test_unfaithful_buy_when_judge_says_sell(self) -> None:
        """Judge says sell but agent bought → unfaithful."""
        d = _decision(action="buy", rationale="Downtrend but buying anyway.")
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "sell")
        assert result["faithfulness"] == 0.0
        assert result["n_unfaithful"] == 1
        assert len(result["examples_unfaithful"]) == 1

    def test_unfaithful_hold_when_judge_says_buy_no_position(self) -> None:
        """Judge says buy, agent held, position is 0 → NOT constrained → unfaithful."""
        d = _decision(action="hold", position_pct=0.0, position_qty=0.0)
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "buy")
        assert result["n_unfaithful"] == 1
        assert result["n_constrained"] == 0

    def test_constrained_hold_near_nav_limit(self) -> None:
        """Judge says buy but agent holds with position at 19% NAV → constrained."""
        d = _decision(action="hold", position_pct=0.19, position_qty=100.0)
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "buy")
        assert result["n_constrained"] == 1
        assert result["n_unfaithful"] == 0

    def test_constrained_sell_with_no_position(self) -> None:
        """Judge says sell but agent holds with zero position → constrained."""
        d = _decision(action="hold", position_pct=0.0, position_qty=0.0)
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "sell")
        assert result["n_constrained"] == 1
        assert result["n_unfaithful"] == 0

    def test_error_decisions_skipped(self) -> None:
        """Decisions with action='error' are excluded from scoring."""
        d = _decision(action="error")
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "buy")
        assert result["n_scoreable"] == 0
        assert result["n_judge_failed"] == 0

    def test_judge_failure_excluded(self) -> None:
        """None returned by judge → judge_failed, not counted in faithfulness."""
        d = _decision(action="buy")
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: None)
        assert result["n_judge_failed"] == 1
        assert result["n_scoreable"] == 0
        assert result["faithfulness"] == 0.0

    def test_mixed_batch(self) -> None:
        """2 faithful, 1 unfaithful → 2/3 faithfulness."""
        decisions = [
            _decision(action="buy"),
            _decision(action="hold"),
            _decision(action="sell"),
        ]
        # judge always says "buy" → first faithful, second/third unfaithful
        result = compute_faithfulness_llm(decisions, _judge_fn=lambda _: "buy")
        assert result["n_faithful"] == 1
        assert result["n_unfaithful"] == 2
        assert abs(result["faithfulness"] - 1 / 3) < 0.01

    def test_judge_actions_recorded(self) -> None:
        """judge_actions list is populated for annotation export."""
        d = _decision(action="buy")
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "buy")
        assert len(result["judge_actions"]) == 1
        assert result["judge_actions"][0]["verdict"] == "faithful"

    def test_position_at_exactly_18pct_is_constrained(self) -> None:
        """Position at exactly 18% NAV triggers constrained classification."""
        d = _decision(action="hold", position_pct=0.18, position_qty=50.0)
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "buy")
        assert result["n_constrained"] == 1

    def test_position_below_18pct_is_unfaithful(self) -> None:
        """Position at 10% NAV does not excuse a hold when judge says buy."""
        d = _decision(action="hold", position_pct=0.10, position_qty=30.0)
        result = compute_faithfulness_llm([d], _judge_fn=lambda _: "buy")
        assert result["n_unfaithful"] == 1


# ---------------------------------------------------------------------------
# 3. _is_constrained_hold tests
# ---------------------------------------------------------------------------


class TestIsConstrainedHold:
    def test_buy_near_limit(self) -> None:
        d = _decision(action="hold", position_pct=0.19)
        assert _is_constrained_hold(d, "buy", "hold") is True

    def test_buy_well_below_limit(self) -> None:
        d = _decision(action="hold", position_pct=0.05)
        assert _is_constrained_hold(d, "buy", "hold") is False

    def test_sell_no_position(self) -> None:
        d = _decision(action="hold", position_pct=0.0, position_qty=0.0)
        assert _is_constrained_hold(d, "sell", "hold") is True

    def test_sell_with_position(self) -> None:
        d = _decision(action="hold", position_pct=0.10, position_qty=50.0)
        assert _is_constrained_hold(d, "sell", "hold") is False

    def test_same_action_not_constrained(self) -> None:
        d = _decision(action="hold", position_pct=0.19)
        assert _is_constrained_hold(d, "hold", "hold") is False


# ---------------------------------------------------------------------------
# 4. _extract_numeric_claims tests
# ---------------------------------------------------------------------------


class TestExtractNumericClaims:
    def test_parenthesised_value(self) -> None:
        claims = _extract_numeric_claims("The 20-day SMA (136.22) is above current price.")
        values = [v for _, v in claims]
        assert 136.22 in values

    def test_at_syntax(self) -> None:
        claims = _extract_numeric_claims("RSI at 32.37 indicates oversold conditions.")
        values = [v for _, v in claims]
        assert 32.37 in values

    def test_multiple_claims(self) -> None:
        text = "RSI at 48.89, SMA20 (131.99), ATR at 4.17."
        claims = _extract_numeric_claims(text)
        assert len(claims) >= 2

    def test_no_claims_empty(self) -> None:
        claims = _extract_numeric_claims("The market is in a bull regime.")
        assert claims == []

    def test_integer_not_extracted(self) -> None:
        claims = _extract_numeric_claims("We held 100 shares worth 200 dollars.")
        values = [v for _, v in claims]
        assert 100.0 not in values
        assert 200.0 not in values


# ---------------------------------------------------------------------------
# 5. _ground_claim tests
# ---------------------------------------------------------------------------


class TestGroundClaim:
    def test_indicator_match(self) -> None:
        indicators = {"rsi_14": 32.37, "sma_20": 136.22}
        assert _ground_claim("rsi", 32.37, indicators, [])
        assert _ground_claim("sma20", 136.22, indicators, [])

    def test_indicator_tolerance(self) -> None:
        indicators = {"sma_20": 136.22}
        assert not _ground_claim("sma", 136.22 * 1.011, indicators, [])
        assert _ground_claim("sma", 136.22 * 1.005, indicators, [])

    def test_ungrounded_claim(self) -> None:
        indicators = {"rsi_14": 50.0}
        assert not _ground_claim("sma20", 999.99, indicators, [])

    def test_bar_price_match(self) -> None:
        bars = [{"date": "2023-01-05", "close": 133.49, "open": 131.0,
                 "high": 134.0, "low": 130.0}]
        assert _ground_claim("close", 133.49, {}, bars)

    def test_portfolio_nav_match(self) -> None:
        """NAV value cited in rationale should be grounded via portfolio_snapshot."""
        portfolio = {"nav": 99000.0, "cash": 50000.0, "positions": []}
        assert _ground_claim("nav", 99000.0, {}, [], portfolio)

    def test_portfolio_position_market_value_match(self) -> None:
        """Position market value cited should be grounded."""
        portfolio = {
            "nav": 99000.0,
            "cash": 50000.0,
            "positions": [
                {"ticker": "AAPL", "quantity": 100, "market_value": 14156.0,
                 "pct_of_nav": 0.143}
            ],
        }
        assert _ground_claim("market value", 14156.0, {}, [], portfolio)

    def test_portfolio_cash_match(self) -> None:
        portfolio = {"nav": 99000.0, "cash": 50000.0, "positions": []}
        assert _ground_claim("cash", 50000.0, {}, [], portfolio)

    def test_portfolio_ungrounded_random(self) -> None:
        """A random number not in any source is not grounded."""
        portfolio = {"nav": 99000.0, "cash": 50000.0, "positions": []}
        assert not _ground_claim("something", 12345.67, {}, [], portfolio)


# ---------------------------------------------------------------------------
# 6. compute_grounding tests
# ---------------------------------------------------------------------------


class TestComputeGrounding:
    def _make_decision_with_raw_portfolio(
        self,
        rationale: str,
        indicators: dict[str, object],
        bars_recent: list[dict[str, object]] | None = None,
        nav: float = 99000.0,
    ) -> dict[str, object]:
        """Helper: decision with full raw tool_outputs (new schema)."""
        return {
            "date": "2023-01-05",
            "ticker": "AAPL",
            "action": "hold",
            "rationale": rationale,
            "indicators": indicators,
            "regime": "bull",
            "tool_outputs": [
                {
                    "tool": "get_price_history",
                    "bars_count": len(bars_recent or []),
                    "bars_recent": bars_recent or [],
                },
                {
                    "tool": "get_portfolio",
                    "cash": 50000.0,
                    "nav": nav,
                    "positions": [],
                },
            ],
        }

    def test_grounded_decision(self) -> None:
        """All claims in indicators → high grounding."""
        d = self._make_decision_with_raw_portfolio(
            rationale="RSI at 32.37 (near oversold). The 20-day SMA (136.22) is above price.",
            indicators={"rsi_14": 32.37, "sma_20": 136.22},
        )
        result = compute_grounding([d])
        assert result["grounding"] >= 0.8
        assert result["n_claims_total"] >= 2

    def test_hallucinated_decision(self) -> None:
        """Invented numbers → low grounding."""
        d = self._make_decision_with_raw_portfolio(
            rationale="RSI at 99.99 says overbought; SMA20 (999.99) is high.",
            indicators={"rsi_14": 32.37, "sma_20": 136.22},
        )
        result = compute_grounding([d])
        assert result["grounding"] == 0.0

    def test_no_claims_not_penalised(self) -> None:
        """No numeric claims → n_claims_total == 0 (not penalised)."""
        d = self._make_decision_with_raw_portfolio(
            rationale="Market looks bullish; holding for now.",
            indicators={},
        )
        result = compute_grounding([d])
        assert result["n_claims_total"] == 0

    def test_error_decisions_skipped(self) -> None:
        d = {"date": "2023-01-05", "ticker": "AAPL", "action": "error",
             "rationale": "", "indicators": {}, "tool_outputs": []}
        result = compute_grounding([d])
        assert result["n_claims_total"] == 0

    def test_nav_claim_grounded_via_portfolio(self) -> None:
        """NAV value in rationale grounded via get_portfolio tool_output."""
        d = self._make_decision_with_raw_portfolio(
            rationale="Current portfolio NAV at 99000.00 is within limits.",
            indicators={},
            nav=99000.0,
        )
        result = compute_grounding([d])
        assert result["n_grounded"] >= 1

    def test_bar_close_grounded(self) -> None:
        """Close price in rationale grounded via bars_recent."""
        bars = [{"date": "2023-01-05", "open": 131.0, "high": 134.0,
                 "low": 130.0, "close": 133.49, "volume": 1000000}]
        d = self._make_decision_with_raw_portfolio(
            rationale="The recent close (133.49) is above the SMA.",
            indicators={"sma_20": 130.0},
            bars_recent=bars,  # type: ignore[arg-type]
        )
        result = compute_grounding([d])
        assert result["n_grounded"] >= 1


# ---------------------------------------------------------------------------
# 6b. PM time-machine audit + evidence grounding
# ---------------------------------------------------------------------------


def _pm_decision_for_audit() -> dict[str, object]:
    return {
        "date": "2023-01-04",
        "mode": "multi_agent_pm",
        "regimes": {"AAPL": "bear"},
        "mcp_calls": [
            {
                "ticker": "AAPL",
                "tool": "get_price_history",
                "source": "cache",
                "max_timestamp": "2023-01-04",
            },
            {
                "ticker": "AAPL",
                "tool": "get_news_corpus",
                "source": "news_corpus",
                "max_timestamp": "2023-01-03T12:00:00",
            },
        ],
        "tool_outputs": [
            {
                "ticker": "AAPL",
                "tool": "get_price_history",
                "max_timestamp": "2023-01-04",
                "bars_recent": [
                    {
                        "date": "2023-01-04",
                        "open": 125.0,
                        "high": 127.0,
                        "low": 124.0,
                        "close": 126.36,
                    }
                ],
            },
            {
                "ticker": "AAPL",
                "tool": "compute_indicators",
                "values": {
                    "date": "2023-01-04",
                    "close": 126.36,
                    "rsi_14": 34.88,
                    "sma_20": 135.2,
                    "macd_histogram": -0.77,
                },
            },
            {
                "ticker": "AAPL",
                "tool": "get_current_regime",
                "regime": "bear",
            },
            {
                "ticker": "AAPL",
                "tool": "get_news_corpus",
                "max_timestamp": "2023-01-03T12:00:00",
                "items_recent": [
                    {
                        "datetime": "2023-01-03T12:00:00",
                        "headline": "Apple supplier demand weakens",
                        "source": "Reuters",
                    }
                ],
            },
        ],
        "portfolio_before": {
            "cash": 50_000.0,
            "nav": 100_000.0,
            "positions": [
                {
                    "ticker": "AAPL",
                    "quantity": 100.0,
                    "market_value": 12_636.0,
                    "current_price": 126.36,
                    "pct_of_nav": 0.12636,
                }
            ],
        },
        "orders": [{"date": "2023-01-04", "ticker": "AAPL"}],
        "fills": [{"timestamp": "2023-01-04", "ticker": "AAPL"}],
        "reports": [
            {
                "ticker": "AAPL",
                "analyst": "technical",
                "evidence": [
                    "AAPL regime is bear",
                    "Close 126.36 is below SMA 20 at 135.20",
                    "RSI is low at 34.88",
                ],
            },
            {
                "ticker": "AAPL",
                "analyst": "news",
                "evidence": [
                    "Reuters 2023-01-03T12:00:00: Apple supplier demand weakens"
                ],
            },
            {
                "ticker": "AAPL",
                "analyst": "risk",
                "evidence": ["cash at 50000.00", "position value 12636.00"],
            },
        ],
        "rationale": "No numeric claims here.",
    }


def test_mcp_time_machine_audit_passes_for_causal_pm_decision() -> None:
    audit = compute_mcp_time_machine_audit([_pm_decision_for_audit()])

    assert audit["pass"] is True
    assert audit["n_violations"] == 0
    assert audit["n_timestamp_checks"] >= 6


def test_mcp_time_machine_audit_flags_future_tool_output() -> None:
    decision = _pm_decision_for_audit()
    decision["tool_outputs"][0]["bars_recent"][0]["date"] = "2023-01-05"  # type: ignore[index]

    audit = compute_mcp_time_machine_audit([decision])

    assert audit["pass"] is False
    assert audit["n_violations"] == 1
    assert audit["violations"][0]["reason"] == "timestamp after t_now"


def test_pm_evidence_grounding_matches_news_technical_and_risk_sources() -> None:
    result = compute_pm_evidence_grounding([_pm_decision_for_audit()])

    assert result["pm_evidence_grounding"] == 1.0
    assert result["n_ungrounded"] == 0
    assert result["by_analyst"]["news"]["grounded"] == 1
    assert result["by_analyst"]["technical"]["grounded"] == 3
    assert result["by_analyst"]["risk"]["grounded"] == 2


def test_pm_evidence_grounding_flags_uncited_news() -> None:
    decision = _pm_decision_for_audit()
    decision["reports"][1]["evidence"] = ["A news article said demand weakened"]  # type: ignore[index]

    result = compute_pm_evidence_grounding([decision])

    assert result["n_ungrounded"] == 1
    assert result["examples"][0]["analyst"] == "news"
