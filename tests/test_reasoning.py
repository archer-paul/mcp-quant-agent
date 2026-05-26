"""Tests for eval/reasoning.py — faithfulness, grounding, sophistication.

Tests cover known-good and known-bad cases to verify that each metric
behaves correctly before applying to real thesis data.
"""

from __future__ import annotations

from mcp_quant_agent.eval.reasoning import (
    _extract_intent,
    _extract_numeric_claims,
    _ground_claim,
    compute_faithfulness,
    compute_grounding,
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
) -> dict[str, object]:
    return {
        "date": "2023-01-05",
        "ticker": "AAPL",
        "action": action,
        "quantity": 0 if action == "hold" else 50,
        "rationale": rationale,
        "fill": fill,
        "regime": regime,
        "indicators": indicators or {},
        "tool_outputs": tool_outputs or [],
        "latency_ms": 100.0,
        "errors": [],
    }


# ---------------------------------------------------------------------------
# 1. _extract_intent tests
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
        # "buy" appears once, "hold" appears twice → hold wins
        text = "I might buy but I'll hold for now and hold my position."
        result = _extract_intent(text)
        assert result == "hold"

    def test_case_insensitive(self) -> None:
        assert _extract_intent("Strong uptrend — BUY signal confirmed.") == "buy"

    def test_hold_context_dominant(self) -> None:
        # "not buy" → should map to hold
        text = "I will not buy at this level; staying put."
        result = _extract_intent(text)
        assert result == "hold"


# ---------------------------------------------------------------------------
# 2. compute_faithfulness tests
# ---------------------------------------------------------------------------


class TestComputeFaithfulness:
    def test_faithful_buy(self) -> None:
        """action=buy + rationale signals buy → faithful."""
        d = _decision(
            action="buy",
            rationale="RSI is neutral and trend is up; I should buy to capture momentum.",
        )
        result = compute_faithfulness([d])
        assert result["faithfulness"] == 1.0
        assert result["n_faithful"] == 1

    def test_faithful_hold(self) -> None:
        """action=hold + rationale signals hold → faithful."""
        d = _decision(
            action="hold",
            rationale="Market is uncertain; holding is prudent until clearer signals.",
        )
        result = compute_faithfulness([d])
        assert result["faithfulness"] == 1.0

    def test_unfaithful_buy_with_hold_rationale(self) -> None:
        """action=buy but rationale says hold → unfaithful."""
        d = _decision(
            action="buy",
            rationale="No clear signal; I will hold and wait for confirmation.",
        )
        result = compute_faithfulness([d])
        assert result["faithfulness"] == 0.0
        assert result["n_unfaithful"] == 1
        assert len(result["examples_unfaithful"]) == 1

    def test_unfaithful_hold_with_sell_rationale(self) -> None:
        """action=hold but rationale says sell → unfaithful."""
        d = _decision(
            action="hold",
            rationale="RSI overbought, best to sell and exit position.",
        )
        result = compute_faithfulness([d])
        assert result["faithfulness"] == 0.0

    def test_no_signal_excluded(self) -> None:
        """Rationale with no intent signal is excluded (not penalised)."""
        d = _decision(
            action="buy",
            rationale="The market has complex dynamics and multifactorial drivers.",
        )
        result = compute_faithfulness([d])
        assert result["n_no_signal"] == 1
        assert result["n_scoreable"] == 0
        # faithfulness is 0.0 when no scoreable decisions (0/0 → 0.0)
        assert result["faithfulness"] == 0.0

    def test_error_decisions_skipped(self) -> None:
        """Decisions with action='error' are excluded."""
        d = _decision(action="error", rationale="")
        result = compute_faithfulness([d])
        assert result["n_scoreable"] == 0

    def test_mixed_batch(self) -> None:
        """Batch with 2 faithful, 1 unfaithful → 2/3 faithfulness."""
        decisions = [
            _decision(action="buy", rationale="uptrend confirmed, buy signal strong"),
            _decision(action="hold", rationale="It is prudent to hold and stay put."),
            _decision(action="hold", rationale="RSI is oversold; I should sell soon."),
        ]
        result = compute_faithfulness(decisions)
        assert result["n_faithful"] == 2
        assert result["n_unfaithful"] == 1
        assert abs(result["faithfulness"] - 2 / 3) < 0.01


# ---------------------------------------------------------------------------
# 3. _extract_numeric_claims tests
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
        """Integers without decimal point are not extracted (noise reduction)."""
        claims = _extract_numeric_claims("We held 100 shares worth 200 dollars.")
        # 100 and 200 have no decimal, should not be extracted
        values = [v for _, v in claims]
        assert 100.0 not in values
        assert 200.0 not in values


# ---------------------------------------------------------------------------
# 4. _ground_claim tests
# ---------------------------------------------------------------------------


class TestGroundClaim:
    def test_indicator_match(self) -> None:
        """Claim is grounded when indicator value matches within 1%."""
        indicators = {"rsi_14": 32.37, "sma_20": 136.22}
        assert _ground_claim("rsi", 32.37, indicators, [])
        assert _ground_claim("sma20", 136.22, indicators, [])

    def test_indicator_tolerance(self) -> None:
        """Match within 1% tolerance."""
        indicators = {"sma_20": 136.22}
        # 136.22 * 1.009 = 137.59 → 1.1% off, should not match
        assert not _ground_claim("sma", 136.22 * 1.011, indicators, [])
        # 136.22 * 1.005 → 0.5% off, should match
        assert _ground_claim("sma", 136.22 * 1.005, indicators, [])

    def test_ungrounded_claim(self) -> None:
        """Claim not in indicators or bars → not grounded."""
        indicators = {"rsi_14": 50.0}
        assert not _ground_claim("sma20", 999.99, indicators, [])

    def test_bar_price_match(self) -> None:
        """Claim matches a bar close price."""
        bars = [{"date": "2023-01-05", "close": 133.49, "open": 131.0, "high": 134.0, "low": 130.0}]
        assert _ground_claim("close", 133.49, {}, bars)


# ---------------------------------------------------------------------------
# 5. compute_grounding tests
# ---------------------------------------------------------------------------


class TestComputeGrounding:
    def test_grounded_decision(self) -> None:
        """Decision where all claims are in indicators → high grounding."""
        d = _decision(
            action="hold",
            rationale=(
                "RSI at 32.37 (near oversold). The 20-day SMA (136.22) is above price."
            ),
            indicators={"rsi_14": 32.37, "sma_20": 136.22},
        )
        result = compute_grounding([d])
        assert result["grounding"] >= 0.8
        assert result["n_claims_total"] >= 2

    def test_hallucinated_decision(self) -> None:
        """Decision with invented numbers → low grounding."""
        d = _decision(
            action="buy",
            rationale="RSI at 99.99 says overbought; SMA20 (999.99) is high.",
            indicators={"rsi_14": 32.37, "sma_20": 136.22},
        )
        result = compute_grounding([d])
        assert result["grounding"] == 0.0
        assert result["n_claims_total"] >= 2

    def test_no_claims_not_penalised(self) -> None:
        """Decision with no numeric claims has n_claims_total=0 (not penalised)."""
        d = _decision(
            action="hold",
            rationale="Market looks bullish; holding for now.",
        )
        result = compute_grounding([d])
        assert result["n_claims_total"] == 0

    def test_error_decisions_skipped(self) -> None:
        d = _decision(action="error")
        result = compute_grounding([d])
        assert result["n_claims_total"] == 0
