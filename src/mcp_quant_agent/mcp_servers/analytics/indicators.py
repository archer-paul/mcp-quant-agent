"""Technical indicators — adapted from Salvi reference code.

Source: ``gen-ai-imperial/code/week3_agent/indicators.py`` (read-only reference).
Adaptations:
- Full type annotations (``list[dict[str, Any]]``).
- Docstrings on every public function.
- ``compute_indicators`` hardened: raises ``ValueError`` instead of returning
  an error dict (fail loudly convention).

All functions operate on OHLCV rows that have already been filtered by t_now
(the caller, i.e. the MCP data wrapper, has already applied ``clock.filter_rows``).
No additional temporal filtering is needed here.

Formulae are unchanged from the reference so results are directly comparable.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _closes(rows: list[dict[str, Any]]) -> list[float]:
    """Extract the ``close`` column from a list of OHLCV row dicts."""
    return [float(r["close"]) for r in rows]


def _ema_series(closes: list[float], window: int) -> list[float]:
    """Compute an Exponential Moving Average series.

    Uses the standard smoothing factor k = 2 / (window + 1).

    Parameters
    ----------
    closes:
        List of closing prices (oldest first).
    window:
        EMA window length (e.g. 12 for MACD fast, 26 for MACD slow).

    Returns
    -------
    list[float]
        EMA values, same length as *closes*.  First value is seeded with
        ``closes[0]`` (no warm-up period — matches the reference implementation).
    """
    k = 2.0 / (window + 1)
    vals: list[float] = [closes[0]]
    for c in closes[1:]:
        vals.append(c * k + vals[-1] * (1.0 - k))
    return vals


# ---------------------------------------------------------------------------
# Public indicator functions
# ---------------------------------------------------------------------------


def sma(rows: list[dict[str, Any]], window: int = 20) -> list[dict[str, Any]]:
    """Simple Moving Average over *window* periods.

    Parameters
    ----------
    rows:
        OHLCV bars, oldest first.  Must contain ``"date"`` and ``"close"`` keys.
    window:
        Lookback window (default 20 days).

    Returns
    -------
    list[dict[str, Any]]
        One dict per input row: ``{"date": ..., "sma": float | None}``.
        ``sma`` is ``None`` for the first ``window - 1`` rows (insufficient data).

    Examples
    --------
    >>> rows = [{"date": f"2022-01-0{i}", "close": float(i*10)} for i in range(1, 6)]
    >>> sma(rows, window=3)[-1]["sma"]
    40.0
    """
    closes = _closes(rows)
    result: list[dict[str, Any]] = []
    for i in range(len(closes)):
        if i < window - 1:
            result.append({"date": rows[i]["date"], "sma": None})
        else:
            avg = sum(closes[i - window + 1 : i + 1]) / window
            result.append({"date": rows[i]["date"], "sma": round(avg, 4)})
    return result


def rsi(rows: list[dict[str, Any]], window: int = 14) -> list[dict[str, Any]]:
    """Relative Strength Index (Wilder smoothing).

    Parameters
    ----------
    rows:
        OHLCV bars, oldest first.
    window:
        RSI period (default 14).

    Returns
    -------
    list[dict[str, Any]]
        One dict per row: ``{"date": ..., "rsi": float | None}``.
        ``rsi`` is ``None`` for the first ``window`` rows.
    """
    closes = _closes(rows)
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    result: list[dict[str, Any]] = [{"date": rows[0]["date"], "rsi": None}]

    avg_gain: float = 0.0
    avg_loss: float = 0.0
    gains: list[float] = []
    losses: list[float] = []

    for i, d in enumerate(deltas):
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
        if i < window - 1:
            result.append({"date": rows[i + 1]["date"], "rsi": None})
        elif i == window - 1:
            avg_gain = sum(gains[-window:]) / window
            avg_loss = sum(losses[-window:]) / window
            rs = avg_gain / avg_loss if avg_loss != 0 else 100.0
            result.append(
                {
                    "date": rows[i + 1]["date"],
                    "rsi": round(100.0 - 100.0 / (1.0 + rs), 2),
                }
            )
        else:
            avg_gain = (avg_gain * (window - 1) + gains[-1]) / window
            avg_loss = (avg_loss * (window - 1) + losses[-1]) / window
            rs = avg_gain / avg_loss if avg_loss != 0 else 100.0
            result.append(
                {
                    "date": rows[i + 1]["date"],
                    "rsi": round(100.0 - 100.0 / (1.0 + rs), 2),
                }
            )
    return result


def macd(
    rows: list[dict[str, Any]],
    fast: int = 12,
    slow: int = 26,
    signal_window: int = 9,
) -> list[dict[str, Any]]:
    """MACD line, signal line, and histogram.

    Parameters
    ----------
    rows:
        OHLCV bars, oldest first.
    fast, slow:
        EMA periods for the MACD line (default 12 and 26).
    signal_window:
        EMA period for the signal line (default 9).

    Returns
    -------
    list[dict[str, Any]]
        One dict per row: ``{"date", "macd", "signal", "histogram"}``.
    """
    closes = _closes(rows)
    ema_fast = _ema_series(closes, fast)
    ema_slow = _ema_series(closes, slow)
    macd_line = [round(f - s, 6) for f, s in zip(ema_fast, ema_slow, strict=False)]
    signal_line = _ema_series(macd_line, signal_window)
    return [
        {
            "date": rows[i]["date"],
            "macd": round(macd_line[i], 6),
            "signal": round(signal_line[i], 6),
            "histogram": round(macd_line[i] - signal_line[i], 6),
        }
        for i in range(len(rows))
    ]


def bollinger_bands(
    rows: list[dict[str, Any]],
    window: int = 20,
    num_std: float = 2.0,
) -> list[dict[str, Any]]:
    """Bollinger Bands (middle SMA ± num_std × rolling standard deviation).

    Parameters
    ----------
    rows:
        OHLCV bars, oldest first.
    window:
        Rolling window (default 20).
    num_std:
        Number of standard deviations (default 2.0).

    Returns
    -------
    list[dict[str, Any]]
        One dict per row: ``{"date", "upper", "middle", "lower"}``.
        Values are ``None`` for the first ``window - 1`` rows.
    """
    closes = _closes(rows)
    result: list[dict[str, Any]] = []
    for i in range(len(closes)):
        if i < window - 1:
            result.append(
                {"date": rows[i]["date"], "upper": None, "middle": None, "lower": None}
            )
        else:
            window_slice = closes[i - window + 1 : i + 1]
            mid = sum(window_slice) / window
            variance = sum((x - mid) ** 2 for x in window_slice) / window
            std = variance**0.5
            result.append(
                {
                    "date": rows[i]["date"],
                    "upper": round(mid + num_std * std, 4),
                    "middle": round(mid, 4),
                    "lower": round(mid - num_std * std, 4),
                }
            )
    return result


def atr(rows: list[dict[str, Any]], window: int = 14) -> list[dict[str, Any]]:
    """Average True Range — a measure of volatility.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    ATR is the Wilder-smoothed rolling mean of True Range.

    Parameters
    ----------
    rows:
        OHLCV bars, oldest first.  Must contain ``high``, ``low``, ``close``.
    window:
        ATR period (default 14).

    Returns
    -------
    list[dict[str, Any]]
        One dict per row: ``{"date", "atr": float | None}``.
    """
    result: list[dict[str, Any]] = [{"date": rows[0]["date"], "atr": None}]
    tr_values: list[float] = []

    for i in range(1, len(rows)):
        high = float(rows[i]["high"])
        low = float(rows[i]["low"])
        prev_close = float(rows[i - 1]["close"])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

        if i < window:
            result.append({"date": rows[i]["date"], "atr": None})
        elif i == window:
            atr_val = sum(tr_values[:window]) / window
            result.append({"date": rows[i]["date"], "atr": round(atr_val, 4)})
        else:
            prev_atr = (
                float(result[-1]["atr"]) if result[-1]["atr"] is not None else 0.0
            )
            atr_val = (prev_atr * (window - 1) + tr) / window
            result.append({"date": rows[i]["date"], "atr": round(atr_val, 4)})
    return result


def compute_indicators(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all indicators and return the most recent values.

    This is the main entry point called by the analytics MCP server.

    Parameters
    ----------
    rows:
        OHLCV bars, oldest first.  Must contain at least 26 rows for MACD.
        All rows must have already been filtered by t_now (the data wrapper
        is responsible for that).

    Returns
    -------
    dict[str, Any]
        Latest indicator values: ``date``, ``close``, ``rsi_14``, ``macd``,
        ``macd_signal``, ``macd_histogram``, ``bollinger_upper/middle/lower``,
        ``sma_20``, ``sma_50`` (``None`` if < 50 bars), ``atr_14``.

    Raises
    ------
    ValueError
        If fewer than 26 rows are provided (insufficient for MACD).
    """
    if len(rows) < 26:
        raise ValueError(
            f"compute_indicators requires at least 26 OHLCV rows; "
            f"got {len(rows)}. Fetch a longer price history."
        )

    rsi_vals = rsi(rows)
    macd_vals = macd(rows)
    bb_vals = bollinger_bands(rows)
    sma20_vals = sma(rows, window=20)
    sma50_vals = sma(rows, window=50) if len(rows) >= 50 else None
    atr_vals = atr(rows)

    latest = rows[-1]
    return {
        "date": latest["date"],
        "close": float(latest["close"]),
        "rsi_14": rsi_vals[-1]["rsi"],
        "macd": macd_vals[-1]["macd"],
        "macd_signal": macd_vals[-1]["signal"],
        "macd_histogram": macd_vals[-1]["histogram"],
        "bollinger_upper": bb_vals[-1]["upper"],
        "bollinger_middle": bb_vals[-1]["middle"],
        "bollinger_lower": bb_vals[-1]["lower"],
        "sma_20": sma20_vals[-1]["sma"],
        "sma_50": sma50_vals[-1]["sma"] if sma50_vals else None,
        "atr_14": atr_vals[-1]["atr"],
    }
