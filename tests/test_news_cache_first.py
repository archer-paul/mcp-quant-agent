"""Tests for cache-first Finnhub news access."""

from __future__ import annotations

import pytest

from mcp_quant_agent.clock import SimulationClock, set_clock
from mcp_quant_agent.mcp_servers.data.cache import NewsCache
from mcp_quant_agent.mcp_servers.data.finnhub_source import get_news_items_cache_first


def test_cache_first_news_filters_future_items(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_clock(SimulationClock("2022-06-15"))
    cache = NewsCache(cache_dir=tmp_path)
    cache.write(
        "AAPL",
        [
            {
                "datetime": "2022-06-14T09:00:00",
                "headline": "past item",
                "summary": "",
                "source": "test",
                "url": "past",
                "ticker": "AAPL",
            },
            {
                "datetime": "2022-06-16T09:00:00",
                "headline": "future item",
                "summary": "",
                "source": "test",
                "url": "future",
                "ticker": "AAPL",
            },
        ],
    )

    result = get_news_items_cache_first(
        "AAPL",
        "2022-06-14",
        "2022-06-16",
        offline=True,
        cache=cache,
    )

    assert [item["headline"] for item in result] == ["past item"]


def test_offline_cache_miss_raises_without_fetch(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    set_clock(SimulationClock("2022-06-15"))
    cache = NewsCache(cache_dir=tmp_path)

    def fail_fetch(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise AssertionError("offline mode must not fetch Finnhub")

    monkeypatch.setattr(
        "mcp_quant_agent.mcp_servers.data.finnhub_source._fetch_raw_news",
        fail_fetch,
    )

    with pytest.raises(RuntimeError, match="Offline mode forbids"):
        get_news_items_cache_first(
            "AAPL",
            "2022-06-14",
            "2022-06-16",
            offline=True,
            cache=cache,
        )


def test_existing_cache_with_no_rows_in_window_returns_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_clock(SimulationClock("2022-06-15"))
    cache = NewsCache(cache_dir=tmp_path)
    cache.write(
        "AAPL",
        [
            {
                "datetime": "2022-05-01T09:00:00",
                "headline": "old item",
                "summary": "",
                "source": "test",
                "url": "old",
                "ticker": "AAPL",
            }
        ],
    )

    result = get_news_items_cache_first(
        "AAPL",
        "2022-06-14",
        "2022-06-16",
        offline=True,
        cache=cache,
    )

    assert result == []
