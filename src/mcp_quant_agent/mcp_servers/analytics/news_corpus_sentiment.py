"""Deterministic sentiment over already-filtered news corpus articles."""

from __future__ import annotations

from typing import Any, Literal

from mcp_quant_agent.mcp_servers.data.news_corpus_cache import evidence_from_article

Signal = Literal["bullish", "bearish", "neutral"]

_POSITIVE_TERMS = {
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "raises",
    "raised",
    "growth",
    "profit",
    "profits",
    "strong",
    "record",
    "surge",
    "surges",
    "rally",
    "outperform",
    "positive",
}

_NEGATIVE_TERMS = {
    "miss",
    "misses",
    "downgrade",
    "downgraded",
    "cuts",
    "cut",
    "loss",
    "losses",
    "weak",
    "probe",
    "lawsuit",
    "falls",
    "fall",
    "drop",
    "drops",
    "slump",
    "underperform",
    "negative",
}


def _score_article(article: dict[str, Any]) -> int:
    text = f"{article.get('title', '')} {article.get('body', '')}".lower()
    positive = sum(1 for term in _POSITIVE_TERMS if term in text)
    negative = sum(1 for term in _NEGATIVE_TERMS if term in text)
    return positive - negative


def analyze_causal_news_sentiment(
    ticker: str,
    articles: list[dict[str, Any]],
    *,
    max_evidence: int = 5,
) -> dict[str, Any]:
    """Return deterministic sentiment using only supplied causal articles."""
    ticker = ticker.upper()
    if not articles:
        return {
            "ticker": ticker,
            "signal": "neutral",
            "confidence": 0.0,
            "summary": "sentiment_unavailable: no causal news articles",
            "evidence": [],
        }

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for article in articles:
        score = _score_article(article)
        scored.append((abs(score), score, article))
    total_score = sum(score for _, score, _ in scored)
    if total_score > 0:
        signal: Signal = "bullish"
    elif total_score < 0:
        signal = "bearish"
    else:
        signal = "neutral"

    confidence = min(1.0, abs(total_score) / max(1, len(articles) * 2))
    scored.sort(key=lambda item: item[0], reverse=True)
    evidence: list[str] = []
    for _, _, article in scored[:max_evidence]:
        item = evidence_from_article(article)
        if item:
            evidence.append(item)

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(confidence, 4),
        "summary": f"{signal} deterministic news signal from {len(articles)} causal articles",
        "evidence": evidence,
    }


def get_causal_news_sentiment(
    ticker: str,
    *,
    limit: int | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Read the enabled corpus through the causal gate, then score sentiment."""
    from mcp_quant_agent.mcp_servers.data.news_corpus_cache import get_news_corpus

    payload = get_news_corpus(ticker, limit=limit, strict=strict)
    if payload.get("status") != "ok":
        return {
            "ticker": ticker.upper(),
            "signal": "neutral",
            "confidence": 0.0,
            "summary": f"sentiment_unavailable: {payload.get('reason', 'news unavailable')}",
            "evidence": [],
        }
    return analyze_causal_news_sentiment(ticker, list(payload.get("items", [])))
