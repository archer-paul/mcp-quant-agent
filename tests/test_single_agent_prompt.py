"""Prompt contract tests for the single-agent backbone."""

from __future__ import annotations

from mcp_quant_agent.agents.orchestrator import SYSTEM_PROMPT, OpenAIBackbone


def test_system_prompt_requires_trim_when_position_exceeds_cap() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "current_position_value > 0.20 * nav" in prompt
    assert "sell enough shares" in prompt
    assert "do not ignore an over-cap position" in prompt


def test_user_prompt_includes_position_pct_of_nav_and_no_cot_request() -> None:
    user_prompt = OpenAIBackbone._build_prompt(
        {
            "ticker": "AAPL",
            "t_now_str": "2023-01-03",
            "bars": [
                {
                    "date": "2023-01-03",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1_000_000,
                }
            ],
            "news": [],
            "indicators": {"sma_20": 90.0},
            "portfolio": {
                "cash": 75_000.0,
                "nav": 100_000.0,
                "positions": [
                    {
                        "ticker": "AAPL",
                        "quantity": 250,
                        "market_value": 25_000.0,
                    }
                ],
            },
            "reasoning": "",
            "decision": {},
            "fill": None,
            "regime": "bull",
            "errors": [],
            "tool_outputs": [],
            "latency_ms": 0.0,
        }
    )

    assert '"pct_of_nav": 0.25' in user_prompt
    assert "chain-of-thought" not in user_prompt.lower()
    assert "Output only the JSON decision" in user_prompt
