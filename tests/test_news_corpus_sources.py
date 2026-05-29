from __future__ import annotations

import datetime as dt

from mcp_quant_agent.mcp_servers.data.news_corpus_sources import (
    enrich_articles_with_firecrawl,
    fetch_alpha_vantage_news_sentiment,
    fetch_finnhub_company_news,
    fetch_firecrawl_news_search,
)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_fetch_finnhub_company_news_maps_to_corpus_schema(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    ts = int(dt.datetime(2023, 1, 5, 14, 30, tzinfo=dt.UTC).timestamp())

    def fake_get(url, params, timeout):  # type: ignore[no-untyped-def]
        assert params["symbol"] == "AAPL"
        return _Response(
            [
                {
                    "datetime": ts,
                    "headline": "Apple beats expectations",
                    "summary": "Record profit and strong growth.",
                    "source": "unit",
                    "url": "https://example.test/a",
                }
            ]
        )

    monkeypatch.setattr(
        "mcp_quant_agent.mcp_servers.data.news_corpus_sources.requests.get",
        fake_get,
    )

    rows = fetch_finnhub_company_news(
        "AAPL",
        start_date="2023-01-01",
        end_date="2023-01-31",
        api_key="test",
    )

    assert rows == [
        {
            "ticker": "AAPL",
            "published_at": "2023-01-05T14:30:00",
            "title": "Apple beats expectations",
            "body": "Record profit and strong growth.",
            "source": "unit",
            "url": "https://example.test/a",
        }
    ]


def test_fetch_alpha_vantage_news_sentiment_maps_to_corpus_schema(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake_get(url, params, timeout):  # type: ignore[no-untyped-def]
        assert params["function"] == "NEWS_SENTIMENT"
        assert params["tickers"] == "AAPL"
        assert params["time_from"] == "20230101T0000"
        assert params["time_to"] == "20230131T2359"
        return _Response(
            {
                "feed": [
                    {
                        "time_published": "20230105T143000",
                        "title": "Apple beats expectations",
                        "summary": "Record profit and strong growth.",
                        "source": "unit",
                        "url": "https://example.test/a",
                        "ticker_sentiment": [{"ticker": "AAPL"}],
                    },
                    {
                        "time_published": "20230105T153000",
                        "title": "Broad market story",
                        "summary": "Not ticker specific.",
                        "source": "unit",
                        "url": "https://example.test/b",
                        "ticker_sentiment": [{"ticker": "MSFT"}],
                    },
                ]
            }
        )

    monkeypatch.setattr(
        "mcp_quant_agent.mcp_servers.data.news_corpus_sources.requests.get",
        fake_get,
    )

    rows = fetch_alpha_vantage_news_sentiment(
        "AAPL",
        start_date="2023-01-01",
        end_date="2023-01-31",
        api_key="test",
    )

    assert rows == [
        {
            "ticker": "AAPL",
            "published_at": "2023-01-05T14:30:00",
            "title": "Apple beats expectations",
            "body": "Record profit and strong growth.",
            "source": "unit",
            "url": "https://example.test/a",
        }
    ]


def test_firecrawl_enrichment_updates_body_but_not_timestamp(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake_post(url, headers, json, timeout):  # type: ignore[no-untyped-def]
        return _Response(
            {
                "data": {
                    "markdown": "Full article body from Firecrawl.",
                    "metadata": {"publishedTime": "not authoritative"},
                }
            }
        )

    monkeypatch.setattr(
        "mcp_quant_agent.mcp_servers.data.news_corpus_sources.requests.post",
        fake_post,
    )
    articles = [
        {
            "ticker": "AAPL",
            "published_at": "2023-01-05T14:30:00",
            "title": "Title",
            "body": "Short summary",
            "source": "unit",
            "url": "https://example.test/a",
        }
    ]

    enriched, audit = enrich_articles_with_firecrawl(
        articles,
        max_articles=1,
        api_key="test",
    )

    assert enriched[0]["body"] == "Full article body from Firecrawl."
    assert enriched[0]["published_at"] == "2023-01-05T14:30:00"
    assert audit[0]["metadata"]["publishedTime"] == "not authoritative"


def test_firecrawl_news_search_accepts_publication_metadata(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake_post(url, headers, json, timeout):  # type: ignore[no-untyped-def]
        return _Response(
            {
                "data": {
                    "news": [
                        {
                            "title": "AAPL news",
                            "snippet": "Strong growth",
                            "url": "https://example.test/a",
                            "date": "2023-01-06",
                            "metadata": {
                                "article:published_time": "2023-01-05T14:30:00Z",
                            },
                        },
                        {
                            "title": "No date",
                            "url": "https://example.test/no-date",
                            "metadata": {},
                        },
                    ]
                }
            }
        )

    monkeypatch.setattr(
        "mcp_quant_agent.mcp_servers.data.news_corpus_sources.requests.post",
        fake_post,
    )

    rows, audit = fetch_firecrawl_news_search(
        "AAPL",
        start_date="2023-01-01",
        end_date="2023-01-31",
        api_key="test",
        allow_search_result_date=False,
    )

    assert rows[0]["published_at"] == "2023-01-05T14:30:00"
    assert rows[0]["source"] == "example.test:published_metadata"
    assert any(row["status"] == "rejected_no_timestamp" for row in audit)
