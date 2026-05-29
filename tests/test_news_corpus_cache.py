from __future__ import annotations

import datetime as dt

import pytest

from mcp_quant_agent.clock import SimulationClock, set_clock
from mcp_quant_agent.config import settings
from mcp_quant_agent.mcp_servers.analytics.news_corpus_sentiment import (
    analyze_causal_news_sentiment,
)
from mcp_quant_agent.mcp_servers.data.news_corpus_cache import (
    NewsCorpusCache,
    NewsCorpusError,
    NewsCorpusUnavailable,
    _parse_published_at,
    get_news_corpus,
)


def _article(
    published_at: str,
    title: str,
    *,
    ticker: str = "AAPL",
    body: str = "",
    source: str = "unit",
    url: str = "https://example.test/story",
) -> dict[str, str]:
    return {
        "ticker": ticker,
        "published_at": published_at,
        "title": title,
        "body": body,
        "source": source,
        "url": url,
    }


def test_news_corpus_filters_future_and_sorts_desc(tmp_path) -> None:
    set_clock(SimulationClock("2023-01-05T12:00:00"))
    cache = NewsCorpusCache(tmp_path, max_items=10)
    cache.write_corpus(
        "AAPL",
        [
            _article("2023-01-04T09:00:00", "older positive growth"),
            _article("2023-01-05T11:00:00", "newer record profit"),
            _article("2023-01-06T09:00:00", "future should not appear"),
        ],
    )

    rows = cache.read_filtered("AAPL")

    assert [row["title"] for row in rows] == [
        "newer record profit",
        "older positive growth",
    ]
    assert all(row["published_at"] <= "2023-01-05T12:00:00" for row in rows)


def test_parse_published_at_supported_formats() -> None:
    expected = dt.datetime(2023, 1, 5, 14, 30)

    assert _parse_published_at("2023-01-05T14:30:00") == expected
    assert _parse_published_at("2023-01-05T14:30:00Z") == expected
    assert _parse_published_at("2023-01-05 14:30:00") == expected
    assert _parse_published_at("1672929000") == expected


def test_news_corpus_disabled_returns_unavailable_without_reading(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(settings, "news_corpus_enabled", False)
    payload = get_news_corpus("AAPL", cache=NewsCorpusCache(tmp_path))

    assert payload["status"] == "sentiment_unavailable"
    assert payload["items"] == []
    assert payload["reason"] == "news_corpus_enabled=False"


def test_default_news_corpus_cache_uses_settings_dir(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(settings, "news_corpus_dir", tmp_path)
    cache = NewsCorpusCache()
    assert cache.parquet_path("AAPL") == tmp_path / "AAPL.parquet"


def test_missing_corpus_raises_in_strict_enabled_mode(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(settings, "news_corpus_enabled", True)

    with pytest.raises(NewsCorpusUnavailable):
        get_news_corpus("AAPL", strict=True, cache=NewsCorpusCache(tmp_path))


def test_news_corpus_write_requires_strict_schema(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = NewsCorpusCache(tmp_path)

    with pytest.raises(NewsCorpusError, match="missing required columns"):
        cache.write_corpus("AAPL", [{"ticker": "AAPL", "published_at": "2023-01-05"}])


def test_sentiment_evidence_uses_only_filtered_articles(tmp_path) -> None:
    set_clock(SimulationClock("2023-01-05T12:00:00"))
    cache = NewsCorpusCache(tmp_path, max_items=10)
    cache.write_corpus(
        "AAPL",
        [
            _article(
                "2023-01-05T11:00:00",
                "AAPL beats expectations",
                body="Record profit and strong growth.",
                source="past-source",
            ),
            _article(
                "2023-01-06T09:00:00",
                "AAPL future lawsuit",
                body="Future negative story.",
                source="future-source",
            ),
        ],
    )

    rows = cache.read_filtered("AAPL")
    report = analyze_causal_news_sentiment("AAPL", rows)

    assert report["signal"] == "bullish"
    assert report["confidence"] > 0.0
    evidence_text = " ".join(report["evidence"])
    assert "past-source" in evidence_text
    assert "AAPL beats expectations" in evidence_text
    assert "future-source" not in evidence_text
    assert "future lawsuit" not in evidence_text


def test_single_agent_perceive_uses_enabled_news_corpus(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from mcp_quant_agent.agents import orchestrator

    def fail_if_called(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise AssertionError("Finnhub should not be called")

    set_clock(SimulationClock("2023-01-05T12:00:00"))
    cache = NewsCorpusCache(tmp_path, max_items=10)
    cache.write_corpus(
        "AAPL",
        [
            _article("2023-01-04T09:00:00", "causal corpus item"),
            _article("2023-01-06T09:00:00", "future corpus item"),
        ],
    )
    monkeypatch.setattr(settings, "news_corpus_enabled", True)
    monkeypatch.setattr(settings, "news_corpus_dir", tmp_path)
    monkeypatch.setattr(
        "mcp_quant_agent.mcp_servers.data.finnhub_source.get_news_items",
        fail_if_called,
    )
    monkeypatch.setattr(orchestrator, "NewsCorpusCache", NewsCorpusCache, raising=False)

    news = orchestrator._get_single_agent_news(
        "AAPL",
        end_str="2023-01-05",
        news_start="2023-01-01",
    )

    headlines = [item["headline"] for item in news]
    assert "causal corpus item" in headlines
    assert "future corpus item" not in headlines
