"""Regime-conditioned metric computation.

Takes regime labels (from ``mcp_servers/analytics/regime.py``) and segments
financial / reasoning metrics by regime.

This is the key evaluation contribution of the thesis:
reporting ALL metrics (Sharpe, Sortino, faithfulness, grounding) per market
regime (bull / bear / range / high_vol) — not just overall averages.

Reference: ATLAS (arXiv:2510.15949) and LiveTradeBench show that agent
performance and CoT quality vary significantly across regimes.
"""

from __future__ import annotations

from typing import Any

from mcp_quant_agent.eval.financial import compute_all_metrics


def segment_by_regime(
    nav_series: list[dict[str, Any]],
    regime_labels: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Split a NAV series into regime-labelled segments.

    Parameters
    ----------
    nav_series:
        List of ``{"timestamp": ISO-8601, "nav": float}`` dicts.
    regime_labels:
        List of ``{"date": ISO-8601, "regime": str | None}`` dicts from
        ``label_regimes()`` or from the analytics MCP server.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Mapping of regime label → list of NAV dicts in that regime.
        Includes a special ``"all"`` key with the full series.
    """
    # Build a date → regime mapping
    date_to_regime: dict[str, str] = {}
    for entry in regime_labels:
        if entry.get("regime") is not None:
            date_to_regime[str(entry["date"])[:10]] = str(entry["regime"])

    segments: dict[str, list[dict[str, Any]]] = {
        "all": list(nav_series),
        "bull": [],
        "bear": [],
        "range": [],
        "high_vol": [],
        "unlabelled": [],
    }

    for nav_entry in nav_series:
        ts = str(nav_entry.get("timestamp", ""))[:10]
        regime = date_to_regime.get(ts)
        if regime in segments:
            segments[regime].append(nav_entry)
        else:
            segments["unlabelled"].append(nav_entry)

    return segments


def metrics_by_regime(
    nav_series: list[dict[str, Any]],
    regime_labels: list[dict[str, Any]],
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict[str, dict[str, float]]:
    """Compute financial metrics for each market regime.

    Parameters
    ----------
    nav_series:
        Full NAV time series.
    regime_labels:
        Regime labels aligned with the NAV series.
    risk_free_rate, trading_days:
        Forwarded to metric functions.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of regime label → metrics dict (Sharpe, Sortino, Calmar,
        max_drawdown, hit_rate, annualised_return).
        Each regime must have ≥ 2 NAV points to compute metrics; otherwise
        an empty dict is returned for that regime.
    """
    segments = segment_by_regime(nav_series, regime_labels)
    result: dict[str, dict[str, float]] = {}

    for regime, nav_entries in segments.items():
        if len(nav_entries) < 2:
            result[regime] = {}
            continue
        nav_values = [float(e["nav"]) for e in nav_entries]
        result[regime] = compute_all_metrics(nav_values, risk_free_rate, trading_days)

    return result
