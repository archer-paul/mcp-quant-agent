"""OpenAI-backed TradingAgents-style Portfolio Manager chain.

The real PM path mirrors the communication topology used in the local
``reference/TradingAgents`` repo: analyst reports, a bounded Bull/Bear research
debate, Research Manager synthesis, Trader proposal, bounded risk debate, and a
final Portfolio Manager allocation.  Each node passes a compact JSON artifact,
not an unbounded chat transcript, so smoke runs remain replayable and resistant
to context-window drift.

Disk caching follows the same prompt-hash pattern as the single-agent
``OpenAIBackbone`` so small smoke reruns are cheap and replayable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from mcp_quant_agent.agents.orchestrator import _cache_key, _load_cache, _save_cache
from mcp_quant_agent.agents.pm_schemas import AnalystReport, PMTargetWeights

logger = logging.getLogger(__name__)


ANALYST_ROLES = ("technical", "news", "risk")
INVESTMENT_DEBATE_SEQUENCE = ("bull_researcher", "bear_researcher")
RISK_DEBATE_SEQUENCE = ("aggressive_risk", "conservative_risk", "neutral_risk")
MAX_DEBATE_ROUNDS = 1
MAX_RISK_DISCUSS_ROUNDS = 1
TRADINGAGENTS_STYLE_LLM_CALLS = (
    len(ANALYST_ROLES)
    + len(INVESTMENT_DEBATE_SEQUENCE) * MAX_DEBATE_ROUNDS
    + 1  # research manager
    + 1  # trader
    + len(RISK_DEBATE_SEQUENCE) * MAX_RISK_DISCUSS_ROUNDS
    + 1  # portfolio manager
)

ANALYST_SYSTEM_PROMPT = """You are the {role} analyst in a multi-agent trading team.
Use only the provided MCP tool outputs. Do not use outside knowledge or future events.

Return ONLY valid JSON with exactly this shape:
{{"reports": [
  {{"ticker": "TICKER", "signal": "bullish"|"bearish"|"neutral", "confidence": <float 0..1>, "summary": "<one concise sentence>", "evidence": ["<bounded evidence item>", "..."]}}
]}}

Rules:
- Include one report for each allowed ticker.
- Evidence must quote or paraphrase only provided tool outputs.
- Keep summaries concise. No markdown. No prose outside JSON.
"""

INVESTMENT_DEBATE_SYSTEM_PROMPT = """You are the {speaker} in a TradingAgents-style investment debate.
Use only the compact analyst reports and bounded MCP summaries provided.

Return ONLY valid JSON:
{{"speaker": "{speaker}", "stance": "bullish"|"bearish"|"neutral", "message": "<2-4 concise sentences>", "referenced_tickers": ["TICKER", "..."], "key_points": ["<short point>", "..."]}}

Rules:
- Bull researcher builds the pro-risk case and directly answers the latest bear point.
- Bear researcher builds the anti-risk case and directly answers the latest bull point.
- Keep the message concise. No markdown. No chain-of-thought. No outside or future information.
"""

RESEARCH_MANAGER_SYSTEM_PROMPT = """You are the Research Manager in a TradingAgents-style team.
Synthesize the bounded Bull/Bear debate into one actionable investment plan.

Return ONLY valid JSON:
{{"speaker": "research_manager", "rating": "Buy"|"Overweight"|"Hold"|"Underweight"|"Sell", "investment_plan": "<3-5 concise sentences>", "referenced_tickers": ["TICKER", "..."], "key_evidence": ["<short evidence item>", "..."]}}

No markdown. No chain-of-thought. No outside or future information.
"""

TRADER_SYSTEM_PROMPT = """You are the Trader in a TradingAgents-style team.
Turn the Research Manager's plan into a concrete transaction proposal for a
long-only paper portfolio.

Return ONLY valid JSON:
{{"speaker": "trader", "action": "buy"|"sell"|"hold"|"rebalance", "proposal": "<2-4 concise sentences>", "target_tickers": ["TICKER", "..."], "risk_notes": ["<short note>", "..."]}}

No markdown. No chain-of-thought. No outside or future information.
"""

RISK_DEBATE_SYSTEM_PROMPT = """You are the {speaker} analyst in the TradingAgents-style risk debate.
Use the trader proposal, initial reports, and previous risk debate state.

Return ONLY valid JSON:
{{"speaker": "{speaker}", "risk_posture": "increase"|"maintain"|"decrease", "message": "<2-4 concise sentences>", "referenced_tickers": ["TICKER", "..."], "concerns": ["<short concern>", "..."]}}

Rules:
- aggressive_risk argues for accepting justified upside.
- conservative_risk argues for capital preservation and drawdown control.
- neutral_risk reconciles both sides into a balanced risk view.
- Respond to prior risk speakers when present.
- No markdown. No chain-of-thought. No outside or future information.
"""

PM_SYSTEM_PROMPT = """You are a portfolio manager allocating a long-only paper portfolio.
You receive a TradingAgents-style structured state: analyst reports, Bull/Bear
debate state, Research Manager plan, Trader proposal, risk debate state, current
prices, portfolio state, market regimes, and MCP tool-call summaries. Use only
data at or before T_NOW.

Return ONLY one JSON object with exactly these keys:
{"weights": {"TICKER": <float 0..1>}, "cash_weight": <float 0..1>, "rationale": "<3-5 concise sentences>"}

Rules:
- Use only tickers listed in the prompt.
- weights are target portfolio weights, not share quantities.
- Long-only: no negative weights, no shorts.
- No leverage: sum(weights) + cash_weight MUST be <= 1.0.
- cash_weight MUST be explicit.
- If evidence is insufficient or parsing would be uncertain, allocate all cash:
  {"weights": {}, "cash_weight": 1.0, "rationale": "insufficient evidence for risk deployment"}.
- Do not include markdown, prose outside JSON, or chain-of-thought.
"""


@dataclass(frozen=True)
class PMDiscussionResult:
    """Full result from the LLM multi-agent PM chain."""

    reports: list[AnalystReport]
    discussion: list[dict[str, Any]]
    targets: PMTargetWeights


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _truncate_text(value: Any, limit: int = 1000) -> str:
    return str(value or "").strip()[:limit]


def _bounded_str_list(value: Any, *, limit: int, item_limit: int = 200) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_truncate_text(item, item_limit) for item in value if _truncate_text(item)][:limit]


def _parse_json_object(raw: str, *, label: str) -> dict[str, Any]:
    text = _strip_markdown_fences(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} response must be a JSON object")
    return payload


def _allowed_refs(payload: dict[str, Any], allowed_tickers: list[str]) -> list[str]:
    allowed = {ticker.upper() for ticker in allowed_tickers}
    raw_refs = (
        payload.get("referenced_tickers")
        or payload.get("target_tickers")
        or payload.get("tickers")
        or []
    )
    refs = [
        str(ticker).strip().upper()
        for ticker in raw_refs
        if str(ticker).strip().upper() in allowed
    ]
    if not refs:
        refs = sorted(allowed)
    return sorted(set(refs))


def parse_pm_target_response(
    raw: str,
    *,
    date: str,
    allowed_tickers: list[str],
) -> PMTargetWeights:
    """Parse and validate the PM's strict JSON target-weight response.

    The parser is fail-loud.  It does not repair negative weights, unknown
    tickers, missing cash, or leverage.  The caller may decide to log an error
    and use an explicit all-cash fallback, but that fallback must be visible in
    the decision record.
    """
    text = _strip_markdown_fences(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("PM response is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("PM response must be a JSON object")
    if "cash_weight" not in payload:
        raise ValueError("PM response missing required cash_weight")

    raw_weights = payload.get("weights")
    if raw_weights is None and "target_weights" in payload:
        raw_weights = payload["target_weights"]
    if raw_weights is None:
        raw_weights = {}
    if not isinstance(raw_weights, dict):
        raise ValueError("PM weights must be a JSON object")

    allowed = {ticker.upper() for ticker in allowed_tickers}
    unexpected = sorted(
        str(ticker).strip().upper()
        for ticker in raw_weights
        if str(ticker).strip().upper() not in allowed
    )
    if unexpected:
        raise ValueError(f"PM response contains unknown tickers: {unexpected}")

    return PMTargetWeights(
        date=date,
        weights={str(ticker).upper(): float(weight) for ticker, weight in raw_weights.items()},
        cash_weight=float(payload["cash_weight"]),
        rationale=str(payload.get("rationale", "")),
    )


def parse_analyst_reports_response(
    raw: str,
    *,
    date: str,
    role: str,
    allowed_tickers: list[str],
) -> list[AnalystReport]:
    """Parse one analyst agent response into strict report schemas."""
    text = _strip_markdown_fences(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{role} analyst response is not valid JSON") from exc

    reports_raw = payload.get("reports") if isinstance(payload, dict) else None
    if not isinstance(reports_raw, list):
        raise ValueError(f"{role} analyst response missing reports list")

    allowed = {ticker.upper() for ticker in allowed_tickers}
    reports: list[AnalystReport] = []
    seen: set[str] = set()
    for item in reports_raw:
        if not isinstance(item, dict):
            raise ValueError(f"{role} analyst report must be an object")
        ticker = str(item.get("ticker", "")).strip().upper()
        if ticker not in allowed:
            raise ValueError(f"{role} analyst returned unknown ticker {ticker!r}")
        seen.add(ticker)
        reports.append(
            AnalystReport(
                date=date,
                ticker=ticker,
                analyst=role,  # type: ignore[arg-type]
                signal=str(item.get("signal", "neutral")).lower(),  # type: ignore[arg-type]
                confidence=float(item.get("confidence", 0.0)),
                summary=str(item.get("summary", ""))[:500],
                evidence=[str(evidence)[:200] for evidence in item.get("evidence", [])][:5],
            )
        )

    missing = sorted(allowed - seen)
    if missing:
        raise ValueError(f"{role} analyst omitted tickers: {missing}")
    return sorted(reports, key=lambda report: report.ticker)


def parse_discussion_response(
    raw: str,
    *,
    role: str,
    allowed_tickers: list[str],
    stage: str = "discussion",
) -> dict[str, Any]:
    """Parse one bounded communication turn.

    The function accepts the TradingAgents-style intermediate shapes used by
    research, trader, and risk nodes, while preserving the legacy
    ``speaker/message/referenced_tickers`` projection consumed by existing logs.
    """
    payload = _parse_json_object(raw, label=f"{role} discussion")
    message = (
        payload.get("message")
        or payload.get("investment_plan")
        or payload.get("proposal")
        or ""
    )
    turn: dict[str, Any] = {
        "stage": stage,
        "speaker": role,
        "message": _truncate_text(message, 1000),
        "referenced_tickers": _allowed_refs(payload, allowed_tickers),
    }
    for key in (
        "stance",
        "rating",
        "action",
        "risk_posture",
        "investment_plan",
        "proposal",
    ):
        if key in payload:
            turn[key] = _truncate_text(payload[key], 1000)
    for key in ("key_points", "key_evidence", "risk_notes", "concerns"):
        if key in payload:
            turn[key] = _bounded_str_list(payload[key], limit=5)
    return turn


def _compact_market_data_summary(
    market_data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for ticker, payload in market_data.items():
        bars = list(payload.get("bars", []))
        news = list(payload.get("news", []))
        recent_bars = [
            {
                "date": bar.get("date"),
                "open": bar.get("open"),
                "high": bar.get("high"),
                "low": bar.get("low"),
                "close": bar.get("close"),
                "volume": bar.get("volume"),
            }
            for bar in bars[-5:]
            if isinstance(bar, dict)
        ]
        summary[ticker] = {
            "bars_count": len(bars),
            "recent_bars": recent_bars,
            "latest_close": recent_bars[-1]["close"] if recent_bars else None,
            "indicators": payload.get("indicators", {}),
            "regime": payload.get("regime"),
            "news_count": len(news),
            "recent_news": [
                {
                    "datetime": item.get("datetime") or item.get("date"),
                    "headline": _truncate_text(item.get("headline", ""), 160),
                }
                for item in news[:3]
                if isinstance(item, dict)
            ],
        }
    return summary


def _reports_payload(reports: list[AnalystReport]) -> list[dict[str, Any]]:
    return [report.model_dump() for report in reports]


class _OpenAIChatMixin:
    """Shared cached JSON chat call helper for PM agents."""

    model: str
    use_cache: bool
    last_cache_hit: bool
    last_prompt_hash: str
    cache_events: list[dict[str, Any]]

    def _call_json_chat(
        self,
        *,
        label: str,
        system: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        cache_prompt = f"label={label}\nSYSTEM:\n{system}\nUSER:\n{prompt}"
        key = _cache_key(self.model, cache_prompt)
        self.last_prompt_hash = key
        self.last_cache_hit = False

        if self.use_cache:
            cached = _load_cache(key)
            if cached is not None:
                self.last_cache_hit = True
                self.cache_events.append(
                    {"label": label, "cache_hit": True, "prompt_hash": key}
                )
                return cached

        content = self._call_openai_raw(
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if self.use_cache:
            _save_cache(key, content)
        self.cache_events.append(
            {"label": label, "cache_hit": False, "prompt_hash": key}
        )
        return content

    def _call_openai_raw(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        from mcp_quant_agent.config import settings
        from mcp_quant_agent.observability.langfuse_setup import _is_langfuse_configured

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; PM API smoke cannot run.")
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        _is_langfuse_configured()

        try:
            from langfuse.openai import openai  # type: ignore[attr-defined]
        except ImportError:
            import openai

        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return str(response.choices[0].message.content or "")


class OpenAIDiscussionBackbone(_OpenAIChatMixin):
    """TradingAgents-style LLM communication chain followed by PM allocation."""

    def __init__(self, model: str, use_cache: bool = True) -> None:
        self.model = model
        self.use_cache = use_cache
        self.last_cache_hit = False
        self.last_prompt_hash = ""
        self.cache_events: list[dict[str, Any]] = []

    def decide(
        self,
        *,
        date: str,
        tickers: list[str],
        market_data: dict[str, dict[str, Any]],
        portfolio: dict[str, Any],
        current_prices: dict[str, float],
        regimes: dict[str, str | None],
        tool_outputs: list[dict[str, Any]],
        mcp_calls: list[dict[str, Any]],
        past_context: str = "",
    ) -> PMDiscussionResult:
        """Run analysts, discussion, and final PM allocation.

        Args:
            past_context: Optional formatted string from ``PMDecisionLog.get_past_context()``.
                When non-empty it is prepended to the PM and analyst prompts so
                the agents can reason about their recent allocation history.
        """
        tickers = [ticker.upper() for ticker in tickers]
        self.cache_events = []

        reports: list[AnalystReport] = []
        for role in ANALYST_ROLES:
            raw = self._call_json_chat(
                label=f"analyst:{role}:{date}:{','.join(tickers)}",
                system=ANALYST_SYSTEM_PROMPT.format(role=role),
                prompt=self._build_analyst_prompt(
                    date=date,
                    role=role,
                    tickers=tickers,
                    market_data=market_data,
                    portfolio=portfolio,
                    current_prices=current_prices,
                    regimes=regimes,
                    tool_outputs=tool_outputs,
                    mcp_calls=mcp_calls,
                    past_context=past_context,
                ),
                max_tokens=900,
            )
            reports.extend(
                parse_analyst_reports_response(
                    raw,
                    date=date,
                    role=role,
                    allowed_tickers=tickers,
                )
            )

        discussion: list[dict[str, Any]] = []
        state = self._create_tradingagents_state(
            reports=reports,
            portfolio=portfolio,
            current_prices=current_prices,
            regimes=regimes,
            mcp_calls=mcp_calls,
        )

        for round_idx in range(MAX_DEBATE_ROUNDS):
            for speaker in INVESTMENT_DEBATE_SEQUENCE:
                raw = self._call_json_chat(
                    label=(
                        f"investment_debate:{speaker}:round{round_idx + 1}:"
                        f"{date}:{','.join(tickers)}"
                    ),
                    system=INVESTMENT_DEBATE_SYSTEM_PROMPT.format(speaker=speaker),
                    prompt=self._build_investment_debate_prompt(
                        date=date,
                        tickers=tickers,
                        speaker=speaker,
                        state=state,
                    ),
                    max_tokens=650,
                    temperature=0.1,
                )
                turn = parse_discussion_response(
                    raw,
                    role=speaker,
                    allowed_tickers=tickers,
                    stage="investment_debate",
                )
                discussion.append(turn)
                self._apply_investment_turn(state, turn)

        raw_research_manager = self._call_json_chat(
            label=f"research_manager:{date}:{','.join(tickers)}",
            system=RESEARCH_MANAGER_SYSTEM_PROMPT,
            prompt=self._build_research_manager_prompt(
                date=date,
                tickers=tickers,
                state=state,
            ),
            max_tokens=750,
            temperature=0.0,
        )
        research_turn = parse_discussion_response(
            raw_research_manager,
            role="research_manager",
            allowed_tickers=tickers,
            stage="research_manager",
        )
        discussion.append(research_turn)
        investment_plan = _truncate_text(
            research_turn.get("investment_plan") or research_turn["message"],
            1500,
        )
        state["investment_plan"] = investment_plan
        state["investment_debate_state"]["judge_decision"] = investment_plan
        state["investment_debate_state"]["current_response"] = investment_plan

        raw_trader = self._call_json_chat(
            label=f"trader:{date}:{','.join(tickers)}",
            system=TRADER_SYSTEM_PROMPT,
            prompt=self._build_trader_prompt(
                date=date,
                tickers=tickers,
                state=state,
            ),
            max_tokens=650,
            temperature=0.0,
        )
        trader_turn = parse_discussion_response(
            raw_trader,
            role="trader",
            allowed_tickers=tickers,
            stage="trader",
        )
        discussion.append(trader_turn)
        trader_plan = _truncate_text(
            trader_turn.get("proposal") or trader_turn["message"],
            1500,
        )
        state["trader_investment_plan"] = trader_plan

        for round_idx in range(MAX_RISK_DISCUSS_ROUNDS):
            for speaker in RISK_DEBATE_SEQUENCE:
                raw = self._call_json_chat(
                    label=(
                        f"risk_debate:{speaker}:round{round_idx + 1}:"
                        f"{date}:{','.join(tickers)}"
                    ),
                    system=RISK_DEBATE_SYSTEM_PROMPT.format(speaker=speaker),
                    prompt=self._build_risk_debate_prompt(
                        date=date,
                        tickers=tickers,
                        speaker=speaker,
                        state=state,
                    ),
                    max_tokens=650,
                    temperature=0.1,
                )
                turn = parse_discussion_response(
                    raw,
                    role=speaker,
                    allowed_tickers=tickers,
                    stage="risk_debate",
                )
                discussion.append(turn)
                self._apply_risk_turn(state, turn)

        pm_prompt = self._build_prompt(
            date=date,
            tickers=tickers,
            reports=reports,
            discussion=discussion,
            tradingagents_state=state,
            portfolio=portfolio,
            current_prices=current_prices,
            regimes=regimes,
            mcp_calls=mcp_calls,
            past_context=past_context,
        )
        raw_targets = self._call_json_chat(
            label=f"portfolio_manager:{date}:{','.join(tickers)}",
            system=PM_SYSTEM_PROMPT,
            prompt=pm_prompt,
            max_tokens=900,
        )
        targets = parse_pm_target_response(
            raw_targets,
            date=date,
            allowed_tickers=tickers,
        )
        return PMDiscussionResult(
            reports=reports,
            discussion=discussion,
            targets=targets,
        )

    @staticmethod
    def _create_tradingagents_state(
        *,
        reports: list[AnalystReport],
        portfolio: dict[str, Any],
        current_prices: dict[str, float],
        regimes: dict[str, str | None],
        mcp_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a compact state shaped like TradingAgents' AgentState."""
        report_by_role: dict[str, list[dict[str, Any]]] = {}
        for report in reports:
            report_by_role.setdefault(report.analyst, []).append(report.model_dump())
        return {
            "market_report": report_by_role.get("technical", []),
            "news_report": report_by_role.get("news", []),
            "risk_report": report_by_role.get("risk", []),
            "portfolio_snapshot": {
                "cash": portfolio.get("cash"),
                "nav": portfolio.get("nav"),
                "positions": portfolio.get("positions", []),
                "num_trades": portfolio.get("num_trades", 0),
            },
            "current_prices": current_prices,
            "regimes": regimes,
            "mcp_call_summaries": mcp_calls,
            "investment_debate_state": {
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "current_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "investment_plan": "",
            "trader_investment_plan": "",
            "risk_debate_state": {
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            },
        }

    @staticmethod
    def _apply_investment_turn(state: dict[str, Any], turn: dict[str, Any]) -> None:
        debate_state = state["investment_debate_state"]
        speaker = str(turn["speaker"])
        argument = f"{speaker}: {turn['message']}"
        debate_state["history"] = _truncate_text(
            f"{debate_state.get('history', '')}\n{argument}",
            4000,
        )
        if speaker == "bull_researcher":
            debate_state["bull_history"] = _truncate_text(
                f"{debate_state.get('bull_history', '')}\n{argument}",
                2500,
            )
        elif speaker == "bear_researcher":
            debate_state["bear_history"] = _truncate_text(
                f"{debate_state.get('bear_history', '')}\n{argument}",
                2500,
            )
        debate_state["current_response"] = argument
        debate_state["count"] = int(debate_state.get("count", 0)) + 1

    @staticmethod
    def _apply_risk_turn(state: dict[str, Any], turn: dict[str, Any]) -> None:
        debate_state = state["risk_debate_state"]
        speaker = str(turn["speaker"])
        argument = f"{speaker}: {turn['message']}"
        debate_state["history"] = _truncate_text(
            f"{debate_state.get('history', '')}\n{argument}",
            5000,
        )
        debate_state["latest_speaker"] = speaker
        if speaker == "aggressive_risk":
            debate_state["aggressive_history"] = _truncate_text(
                f"{debate_state.get('aggressive_history', '')}\n{argument}",
                2500,
            )
            debate_state["current_aggressive_response"] = argument
        elif speaker == "conservative_risk":
            debate_state["conservative_history"] = _truncate_text(
                f"{debate_state.get('conservative_history', '')}\n{argument}",
                2500,
            )
            debate_state["current_conservative_response"] = argument
        elif speaker == "neutral_risk":
            debate_state["neutral_history"] = _truncate_text(
                f"{debate_state.get('neutral_history', '')}\n{argument}",
                2500,
            )
            debate_state["current_neutral_response"] = argument
        debate_state["count"] = int(debate_state.get("count", 0)) + 1

    @staticmethod
    def _state_for_prompt(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "market_report": state["market_report"],
            "news_report": state["news_report"],
            "risk_report": state["risk_report"],
            "portfolio_snapshot": state["portfolio_snapshot"],
            "current_prices": state["current_prices"],
            "regimes": state["regimes"],
            "investment_debate_state": state["investment_debate_state"],
            "investment_plan": state["investment_plan"],
            "trader_investment_plan": state["trader_investment_plan"],
            "risk_debate_state": state["risk_debate_state"],
            "mcp_call_summaries": state["mcp_call_summaries"],
        }

    @classmethod
    def _build_investment_debate_prompt(
        cls,
        *,
        date: str,
        tickers: list[str],
        speaker: str,
        state: dict[str, Any],
    ) -> str:
        return (
            f"T_NOW: {date}\n"
            f"Allowed tickers: {', '.join(tickers)}\n"
            f"Speaker: {speaker}\n"
            "TradingAgents state before this turn:\n"
            f"{json.dumps(cls._state_for_prompt(state), indent=2, sort_keys=True, default=str)}\n"
        )

    @classmethod
    def _build_research_manager_prompt(
        cls,
        *,
        date: str,
        tickers: list[str],
        state: dict[str, Any],
    ) -> str:
        return (
            f"T_NOW: {date}\n"
            f"Allowed tickers: {', '.join(tickers)}\n"
            "Synthesize this bounded investment debate:\n"
            f"{json.dumps(cls._state_for_prompt(state), indent=2, sort_keys=True, default=str)}\n"
        )

    @classmethod
    def _build_trader_prompt(
        cls,
        *,
        date: str,
        tickers: list[str],
        state: dict[str, Any],
    ) -> str:
        return (
            f"T_NOW: {date}\n"
            f"Allowed tickers: {', '.join(tickers)}\n"
            "Create a transaction proposal from this Research Manager plan and portfolio state:\n"
            f"{json.dumps(cls._state_for_prompt(state), indent=2, sort_keys=True, default=str)}\n"
        )

    @classmethod
    def _build_risk_debate_prompt(
        cls,
        *,
        date: str,
        tickers: list[str],
        speaker: str,
        state: dict[str, Any],
    ) -> str:
        return (
            f"T_NOW: {date}\n"
            f"Allowed tickers: {', '.join(tickers)}\n"
            f"Speaker: {speaker}\n"
            "TradingAgents state before this risk turn:\n"
            f"{json.dumps(cls._state_for_prompt(state), indent=2, sort_keys=True, default=str)}\n"
        )

    @staticmethod
    def _build_discussion_prompt(
        *,
        date: str,
        tickers: list[str],
        role: str,
        reports: list[AnalystReport],
        portfolio: dict[str, Any],
        current_prices: dict[str, float],
        regimes: dict[str, str | None],
        mcp_calls: list[dict[str, Any]],
    ) -> str:
        # Backwards-compatible helper retained for tests and older mocks; the
        # real path now uses the TradingAgents-style debate builders above.
        return (
            f"T_NOW: {date}\n"
            f"Speaker role: {role}\n"
            f"Allowed tickers: {', '.join(tickers)}\n"
            f"Current prices:\n{json.dumps(current_prices, indent=2, sort_keys=True)}\n\n"
            f"Regimes:\n{json.dumps(regimes, indent=2, sort_keys=True)}\n\n"
            f"Portfolio:\n{json.dumps(portfolio, indent=2, sort_keys=True, default=str)}\n\n"
            f"Initial reports:\n{json.dumps(_reports_payload(reports), indent=2, sort_keys=True, default=str)}\n\n"
            f"MCP call summaries:\n{json.dumps(mcp_calls, indent=2, sort_keys=True, default=str)}\n"
        )

    @staticmethod
    def _build_prompt(
        *,
        date: str,
        tickers: list[str],
        reports: list[AnalystReport],
        discussion: list[dict[str, Any]],
        tradingagents_state: dict[str, Any],
        portfolio: dict[str, Any],
        current_prices: dict[str, float],
        regimes: dict[str, str | None],
        mcp_calls: list[dict[str, Any]],
        past_context: str = "",
    ) -> str:
        reports_payload = _reports_payload(reports)
        prices_payload = {
            ticker: round(float(current_prices[ticker]), 4)
            for ticker in sorted(current_prices)
        }
        portfolio_payload = {
            "cash": portfolio.get("cash"),
            "nav": portfolio.get("nav"),
            "positions": portfolio.get("positions", []),
            "num_trades": portfolio.get("num_trades", 0),
        }
        past_block = f"{past_context.strip()}\n\n" if past_context.strip() else ""
        return (
            f"{past_block}"
            f"T_NOW: {date}\n"
            f"Allowed tickers: {', '.join(tickers)}\n"
            f"Current prices:\n{json.dumps(prices_payload, indent=2, sort_keys=True)}\n\n"
            f"Regimes:\n{json.dumps(regimes, indent=2, sort_keys=True)}\n\n"
            f"Portfolio snapshot:\n{json.dumps(portfolio_payload, indent=2, sort_keys=True, default=str)}\n\n"
            f"Analyst reports:\n{json.dumps(reports_payload, indent=2, sort_keys=True, default=str)}\n\n"
            f"Communication turns:\n{json.dumps(discussion, indent=2, sort_keys=True, default=str)}\n\n"
            "TradingAgents structured state:\n"
            f"{json.dumps(OpenAIDiscussionBackbone._state_for_prompt(tradingagents_state), indent=2, sort_keys=True, default=str)}\n\n"
            f"MCP tool-call summaries:\n{json.dumps(mcp_calls, indent=2, sort_keys=True, default=str)}\n"
        )

    @staticmethod
    def _build_analyst_prompt(
        *,
        date: str,
        role: str,
        tickers: list[str],
        market_data: dict[str, dict[str, Any]],
        portfolio: dict[str, Any],
        current_prices: dict[str, float],
        regimes: dict[str, str | None],
        tool_outputs: list[dict[str, Any]],
        mcp_calls: list[dict[str, Any]],
        past_context: str = "",
    ) -> str:
        role_tools = {
            "technical": {"get_price_history", "compute_indicators", "get_current_regime"},
            "news": {"get_news_items_cache_first"},
            "risk": {
                "get_portfolio",
                "get_current_regime",
                "compute_indicators",
                "get_price_history",
            },
        }[role]
        visible_tool_outputs = [
            output
            for output in tool_outputs
            if str(output.get("tool")) in role_tools
        ]
        past_block = f"{past_context.strip()}\n\n" if past_context.strip() else ""
        return (
            f"{past_block}"
            f"T_NOW: {date}\n"
            f"Role: {role}\n"
            f"Allowed tickers: {', '.join(tickers)}\n"
            f"Current prices:\n{json.dumps(current_prices, indent=2, sort_keys=True)}\n\n"
            f"Regimes:\n{json.dumps(regimes, indent=2, sort_keys=True)}\n\n"
            f"Portfolio:\n{json.dumps(portfolio, indent=2, sort_keys=True, default=str)}\n\n"
            f"Role-visible MCP tool outputs:\n{json.dumps(visible_tool_outputs, indent=2, sort_keys=True, default=str)}\n\n"
            f"MCP call summaries:\n{json.dumps(mcp_calls, indent=2, sort_keys=True, default=str)}\n\n"
            "Compact market data summary:\n"
            f"{json.dumps(_compact_market_data_summary(market_data), indent=2, sort_keys=True, default=str)}\n"
        )


class OpenAIPMBackbone(_OpenAIChatMixin):
    """Single-call OpenAI backbone for the global Portfolio Manager."""

    def __init__(self, model: str, use_cache: bool = True) -> None:
        self.model = model
        self.use_cache = use_cache
        self.last_cache_hit = False
        self.last_prompt_hash = ""
        self.cache_events: list[dict[str, Any]] = []

    def decide(
        self,
        *,
        date: str,
        tickers: list[str],
        reports: list[AnalystReport],
        portfolio: dict[str, Any],
        current_prices: dict[str, float],
        regimes: dict[str, str | None],
        mcp_calls: list[dict[str, Any]],
    ) -> PMTargetWeights:
        """Return validated PM target weights for one smoke decision date."""
        prompt = self._build_prompt(
            date=date,
            tickers=tickers,
            reports=reports,
            portfolio=portfolio,
            current_prices=current_prices,
            regimes=regimes,
            mcp_calls=mcp_calls,
        )
        content = self._call_json_chat(
            label=f"portfolio_manager_single:{date}:{','.join(tickers)}",
            system=PM_SYSTEM_PROMPT,
            prompt=prompt,
            max_tokens=900,
        )

        return parse_pm_target_response(
            content,
            date=date,
            allowed_tickers=tickers,
        )

    @staticmethod
    def _build_prompt(
        *,
        date: str,
        tickers: list[str],
        reports: list[AnalystReport],
        portfolio: dict[str, Any],
        current_prices: dict[str, float],
        regimes: dict[str, str | None],
        mcp_calls: list[dict[str, Any]],
    ) -> str:
        reports_payload = [report.model_dump() for report in reports]
        prices_payload = {
            ticker: round(float(current_prices[ticker]), 4)
            for ticker in sorted(current_prices)
        }
        portfolio_payload = {
            "cash": portfolio.get("cash"),
            "nav": portfolio.get("nav"),
            "positions": portfolio.get("positions", []),
            "num_trades": portfolio.get("num_trades", 0),
        }
        return (
            f"T_NOW: {date}\n"
            f"Allowed tickers: {', '.join(tickers)}\n"
            f"Current prices:\n{json.dumps(prices_payload, indent=2, sort_keys=True)}\n\n"
            f"Regimes:\n{json.dumps(regimes, indent=2, sort_keys=True)}\n\n"
            f"Portfolio snapshot:\n{json.dumps(portfolio_payload, indent=2, sort_keys=True, default=str)}\n\n"
            f"Analyst reports:\n{json.dumps(reports_payload, indent=2, sort_keys=True, default=str)}\n\n"
            f"MCP tool-call summaries:\n{json.dumps(mcp_calls, indent=2, sort_keys=True, default=str)}\n"
        )
