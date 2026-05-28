"""Schemas for the portfolio-manager multi-agent path.

These models are deliberately separate from the single-agent decision schema.
The PM emits portfolio-level target weights, then a deterministic rebalancer
turns those weights into executable orders.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AnalystRole = Literal["technical", "news", "risk"]
Signal = Literal["bullish", "bearish", "neutral"]
OrderSide = Literal["buy", "sell", "hold"]


def _validate_iso_date(value: str) -> str:
    """Validate an ISO date or datetime string and return it unchanged."""
    dt.date.fromisoformat(str(value)[:10])
    return str(value)


def _normalise_ticker(value: str) -> str:
    ticker = str(value).strip().upper()
    if not ticker:
        raise ValueError("ticker must be non-empty")
    if ticker == "CASH":
        raise ValueError("cash must be represented by cash_weight, not a ticker")
    return ticker


class AnalystReport(BaseModel):
    """Compact analyst report passed to the global Portfolio Manager."""

    model_config = ConfigDict(extra="forbid")

    date: str
    ticker: str
    analyst: AnalystRole
    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("date")
    @classmethod
    def _date_is_iso(cls, value: str) -> str:
        return _validate_iso_date(value)

    @field_validator("ticker")
    @classmethod
    def _ticker_is_clean(cls, value: str) -> str:
        return _normalise_ticker(value)

    @field_validator("evidence")
    @classmethod
    def _evidence_is_bounded(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        for item in cleaned:
            if len(item) > 200:
                raise ValueError("evidence entries must be <= 200 characters")
        return cleaned


class PMTargetWeights(BaseModel):
    """Portfolio-level target weights emitted by the PM.

    The model enforces long-only cash-only allocation:
    ticker weights are non-negative, cash is non-negative, and the total is not
    above 1.0. There is intentionally no per-ticker 20% cap here.
    """

    model_config = ConfigDict(extra="forbid")

    date: str
    weights: dict[str, float] = Field(default_factory=dict)
    cash_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=1000)
    adjustments: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("date")
    @classmethod
    def _date_is_iso(cls, value: str) -> str:
        return _validate_iso_date(value)

    @field_validator("weights", mode="before")
    @classmethod
    def _weights_are_long_only(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("weights must be a mapping of ticker -> weight")

        cleaned: dict[str, float] = {}
        for raw_ticker, raw_weight in value.items():
            ticker = _normalise_ticker(str(raw_ticker))
            if ticker in cleaned:
                raise ValueError(f"duplicate ticker after normalisation: {ticker}")
            weight = float(raw_weight)
            if weight < 0.0:
                raise ValueError(f"negative target weight for {ticker}: {weight}")
            if weight > 1.0:
                raise ValueError(f"target weight for {ticker} exceeds 1.0: {weight}")
            cleaned[ticker] = weight
        return dict(sorted(cleaned.items()))

    @model_validator(mode="after")
    def _total_weight_is_not_levered(self) -> PMTargetWeights:
        total = sum(self.weights.values()) + self.cash_weight
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"target weights plus cash exceed 1.0: {total:.6f}"
            )
        return self

    @property
    def gross_weight(self) -> float:
        """Ticker-only gross exposure."""
        return sum(self.weights.values())

    @property
    def effective_cash_weight(self) -> float:
        """Cash left after applying ticker targets."""
        return max(0.0, 1.0 - self.gross_weight)


class RebalanceOrder(BaseModel):
    """One deterministic order generated from PM target weights."""

    model_config = ConfigDict(extra="forbid")

    date: str
    ticker: str
    side: OrderSide
    quantity: int = Field(ge=0)
    price: float = Field(gt=0.0)
    notional: float = Field(ge=0.0)
    current_weight: float = Field(ge=0.0)
    target_weight: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=500)

    @field_validator("date")
    @classmethod
    def _date_is_iso(cls, value: str) -> str:
        return _validate_iso_date(value)

    @field_validator("ticker")
    @classmethod
    def _ticker_is_clean(cls, value: str) -> str:
        return _normalise_ticker(value)

    @model_validator(mode="after")
    def _side_quantity_consistent(self) -> RebalanceOrder:
        if self.side == "hold":
            if self.quantity != 0:
                raise ValueError("hold orders must have quantity=0")
            if abs(self.notional) > 1e-9:
                raise ValueError("hold orders must have notional=0")
            return self

        if self.quantity <= 0:
            raise ValueError("buy/sell orders must have positive quantity")
        expected = round(self.quantity * self.price, 4)
        if abs(self.notional - expected) > max(0.01, expected * 1e-6):
            raise ValueError(
                f"notional must equal quantity * price: {self.notional} != {expected}"
            )
        return self


class RebalanceResult(BaseModel):
    """Full deterministic rebalance result for one PM decision date."""

    model_config = ConfigDict(extra="forbid")

    date: str
    targets: PMTargetWeights
    orders: list[RebalanceOrder]
    adjustments: list[str] = Field(default_factory=list)
    nav: float = Field(ge=0.0)
    cash_before: float = Field(ge=0.0)
    cash_after_estimate: float = Field(ge=0.0)

    @field_validator("date")
    @classmethod
    def _date_is_iso(cls, value: str) -> str:
        return _validate_iso_date(value)


class DailyPMDecision(BaseModel):
    """Serializable daily PM decision written to decisions.jsonl."""

    model_config = ConfigDict(extra="forbid")

    date: str
    ticker: str = "PORTFOLIO"
    mode: Literal["multi_agent_pm"] = "multi_agent_pm"
    tickers: list[str]
    reports: list[AnalystReport]
    discussion: list[dict[str, Any]] = Field(default_factory=list)
    targets: PMTargetWeights
    rationale: str = Field(default="", max_length=1000)
    orders: list[RebalanceOrder]
    fills: list[dict[str, Any]] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    mcp_calls: list[dict[str, Any]] = Field(default_factory=list)
    indicators: dict[str, Any] = Field(default_factory=dict)
    regimes: dict[str, str | None] = Field(default_factory=dict)
    portfolio_before: dict[str, Any]
    portfolio_after: dict[str, Any]
    nav_before: float = Field(ge=0.0)
    nav_after: float = Field(ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    errors: list[str] = Field(default_factory=list)

    @field_validator("date")
    @classmethod
    def _date_is_iso(cls, value: str) -> str:
        return _validate_iso_date(value)

    @field_validator("tickers")
    @classmethod
    def _tickers_are_clean(cls, value: list[str]) -> list[str]:
        return sorted(_normalise_ticker(ticker) for ticker in value)

    @model_validator(mode="after")
    def _children_match_date(self) -> DailyPMDecision:
        report_dates = {report.date[:10] for report in self.reports}
        order_dates = {order.date[:10] for order in self.orders}
        expected = self.date[:10]
        if report_dates and report_dates != {expected}:
            raise ValueError(f"report dates do not match decision date: {report_dates}")
        if order_dates and order_dates != {expected}:
            raise ValueError(f"order dates do not match decision date: {order_dates}")
        if self.targets.date[:10] != expected:
            raise ValueError("target date does not match decision date")
        return self
