"""FastMCP analytics server — exposes indicators, regime detection, and sentiment.

Run standalone::

    python -m mcp_quant_agent.mcp_servers.analytics.server

This server is purely computational: it accepts price data (already filtered
by t_now) and returns derived quantities.  It has no internal state beyond
the shared simulation clock.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_quant_agent.mcp_servers.analytics.indicators import (
    bollinger_bands as _bollinger,
)
from mcp_quant_agent.mcp_servers.analytics.indicators import (
    compute_indicators as _compute_indicators,
)
from mcp_quant_agent.mcp_servers.analytics.indicators import (
    macd as _macd,
)
from mcp_quant_agent.mcp_servers.analytics.indicators import (
    rsi as _rsi,
)
from mcp_quant_agent.mcp_servers.analytics.regime import (
    get_current_regime as _get_regime,
)
from mcp_quant_agent.mcp_servers.analytics.regime import (
    label_regimes as _label_regimes,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("mcp-quant-analytics")


# ---------------------------------------------------------------------------
# Indicator tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def compute_indicators(ticker: str, bars_json: str) -> str:
    """Compute all technical indicators for the provided OHLCV bars.

    Args:
        ticker: Ticker symbol (used for error messages only).
        bars_json: JSON array of OHLCV bars (date, open, high, low, close, volume).
                   Must already be filtered to t_now by the caller.

    Returns:
        JSON object with latest indicator values:
        {date, close, rsi_14, macd, macd_signal, macd_histogram,
         bollinger_upper, bollinger_middle, bollinger_lower,
         sma_20, sma_50, atr_14}.
    """
    bars: list[dict[str, Any]] = json.loads(bars_json)
    try:
        result = _compute_indicators(bars)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_rsi(bars_json: str, window: int = 14) -> str:
    """Compute RSI for each bar in the provided series.

    Args:
        bars_json: JSON array of OHLCV bars.
        window: RSI window (default 14).

    Returns:
        JSON array of {date, rsi} objects.
    """
    bars: list[dict[str, Any]] = json.loads(bars_json)
    return json.dumps(_rsi(bars, window=window), indent=2)


@mcp.tool()
async def get_macd(
    bars_json: str, fast: int = 12, slow: int = 26, signal: int = 9
) -> str:
    """Compute MACD for each bar.

    Args:
        bars_json: JSON array of OHLCV bars.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        JSON array of {date, macd, signal, histogram} objects.
    """
    bars: list[dict[str, Any]] = json.loads(bars_json)
    return json.dumps(_macd(bars, fast=fast, slow=slow, signal_window=signal), indent=2)


@mcp.tool()
async def get_bollinger_bands(
    bars_json: str, window: int = 20, num_std: float = 2.0
) -> str:
    """Compute Bollinger Bands.

    Args:
        bars_json: JSON array of OHLCV bars.
        window: Rolling window (default 20).
        num_std: Number of standard deviations (default 2.0).

    Returns:
        JSON array of {date, upper, middle, lower} objects.
    """
    bars: list[dict[str, Any]] = json.loads(bars_json)
    return json.dumps(_bollinger(bars, window=window, num_std=num_std), indent=2)


# ---------------------------------------------------------------------------
# Regime detection tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_current_regime(bars_json: str) -> str:
    """Detect the current market regime for the most recent bar.

    Regimes: "bull", "bear", "range", "high_vol".

    Args:
        bars_json: JSON array of OHLCV bars (≥ 80 bars recommended for
                   stable regime detection; need 60-bar trend window).

    Returns:
        JSON object {"regime": "bull" | "bear" | "range" | "high_vol" | null,
                     "date": "2022-06-15"}.
    """
    bars: list[dict[str, Any]] = json.loads(bars_json)
    if not bars:
        return json.dumps({"regime": None, "date": None})
    regime = _get_regime(bars)
    return json.dumps({"regime": regime, "date": bars[-1]["date"]})


@mcp.tool()
async def label_regimes(bars_json: str) -> str:
    """Label each bar in the series with its market regime.

    Args:
        bars_json: JSON array of OHLCV bars.

    Returns:
        JSON array of {date, close, rolling_vol_ann, momentum_60d, regime} objects.
    """
    bars: list[dict[str, Any]] = json.loads(bars_json)
    return json.dumps(_label_regimes(bars), indent=2)


# ---------------------------------------------------------------------------
# Sentiment analysis tool (FinBERT — requires transformers + torch)
# ---------------------------------------------------------------------------


@mcp.tool()
async def analyze_sentiment(headlines_json: str) -> str:
    """Run FinBERT sentiment analysis on a JSON array of headline strings.

    Requires the 'sentiment' optional extra (``uv pip install 'mcp-quant-agent[sentiment]'``).

    Args:
        headlines_json: JSON array of headline strings.

    Returns:
        JSON array of {text, label, score} objects.
        label is one of "positive", "negative", "neutral".
    """
    headlines: list[str] = json.loads(headlines_json)
    try:
        from mcp_quant_agent.mcp_servers.analytics.sentiment import (
            analyze_sentiment as _analyze,
        )

        results = _analyze(headlines)
    except ImportError:
        return json.dumps(
            {
                "error": (
                    "FinBERT not available. "
                    "Install: uv pip install 'mcp-quant-agent[sentiment]' torch"
                )
            }
        )
    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
