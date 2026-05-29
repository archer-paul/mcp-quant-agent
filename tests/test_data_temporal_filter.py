"""Temporal filter tests for the MCP data wrappers.

These tests verify that the *data layer* (yfinance / Finnhub / cache) is
structurally unable to return a bar or news item dated after ``t_now``.
They use mocked data sources so no network calls are made.

TDD timeline
------------
Committed *before* the data wrappers are implemented.  Running::

    pytest tests/test_data_temporal_filter.py

at that point FAILS because the wrapper modules do not yet exist.  The
wrappers are then implemented to make every test here GREEN.  Never weaken
these tests to make code pass.

What we test
------------
1.  Price bars strictly after t_now never leak through the wrapper.
2.  Bar at exactly t_now IS returned (boundary included).
3.  Bar at t_now + 1 day IS excluded (the critical +1-bar test).
4.  t_now before all data: returns empty list, not an error.
5.  All-past data is returned unchanged.
6.  News item with datetime > t_now is excluded.
7.  News item with datetime == t_now midnight is included.
8.  All-future news returns empty list.
9.  ``yfinance.download`` called with ``auto_adjust=False`` (adjusted-price
    leakage guard).
10. Parquet cache stores raw data; t_now filter applied at read time.
11. Cache returns t_now bar after write + read cycle.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_quant_agent.clock import SimulationClock, set_clock

# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------


def _make_price_bars(start: dt.date, end: dt.date) -> list[dict[str, object]]:
    """Generate synthetic OHLCV bars from *start* to *end* inclusive (daily)."""
    bars: list[dict[str, object]] = []
    day = start
    while day <= end:
        close = 100.0 + (day - start).days * 0.5
        bars.append(
            {
                "date": day.isoformat(),
                "open": close - 0.10,
                "high": close + 0.50,
                "low": close - 0.50,
                "close": close,
                "volume": 1_000_000,
            }
        )
        day += dt.timedelta(days=1)
    return bars


def _make_news_items(datetimes: list[str]) -> list[dict[str, object]]:
    return [
        {
            "datetime": ts,
            "headline": f"News at {ts}",
            "summary": "summary",
            "source": "test",
            "url": "https://example.com",
            "ticker": "AAPL",
        }
        for ts in datetimes
    ]


# ---------------------------------------------------------------------------
# 1–5. Price history temporal filter
# ---------------------------------------------------------------------------


class TestPriceHistoryTemporalFilter:
    """get_price_history must respect t_now for every call."""

    @pytest.fixture(autouse=True)
    def _setup_clock(self) -> None:
        self.clock = SimulationClock("2022-06-15")
        set_clock(self.clock)

    def test_bars_after_t_now_are_excluded(self) -> None:
        """Core guarantee: no bar with date > t_now ever leaks through."""
        from mcp_quant_agent.mcp_servers.data.yfinance_source import get_price_history

        bars_with_future = _make_price_bars(dt.date(2022, 6, 10), dt.date(2022, 6, 20))
        with patch(
            "mcp_quant_agent.mcp_servers.data.yfinance_source._fetch_raw_bars",
            return_value=bars_with_future,
        ):
            result = get_price_history(
                "AAPL", "2022-06-10", "2022-06-20", use_cache=False
            )

        future_dates = [r["date"] for r in result if str(r["date"]) > "2022-06-15"]
        assert future_dates == [], (
            f"LOOK-AHEAD LEAK: bars after t_now returned: {future_dates}"
        )

    def test_bar_exactly_at_t_now_included(self) -> None:
        """t_now bar must be included — it is the current bar the agent sees."""
        from mcp_quant_agent.mcp_servers.data.yfinance_source import get_price_history

        bars = _make_price_bars(dt.date(2022, 6, 13), dt.date(2022, 6, 20))
        with patch(
            "mcp_quant_agent.mcp_servers.data.yfinance_source._fetch_raw_bars",
            return_value=bars,
        ):
            result = get_price_history(
                "AAPL", "2022-06-13", "2022-06-20", use_cache=False
            )

        dates = [r["date"] for r in result]
        assert "2022-06-15" in dates, "Bar at t_now was incorrectly excluded"

    def test_bar_one_day_after_t_now_excluded(self) -> None:
        """THE +1-bar test: the bar at t_now+1 must be excluded."""
        from mcp_quant_agent.mcp_servers.data.yfinance_source import get_price_history

        bars = _make_price_bars(dt.date(2022, 6, 13), dt.date(2022, 6, 20))
        with patch(
            "mcp_quant_agent.mcp_servers.data.yfinance_source._fetch_raw_bars",
            return_value=bars,
        ):
            result = get_price_history(
                "AAPL", "2022-06-13", "2022-06-20", use_cache=False
            )

        dates = [r["date"] for r in result]
        assert "2022-06-16" not in dates, "LOOK-AHEAD LEAK: t_now+1 bar was returned"

    def test_t_now_before_all_data_returns_empty(self) -> None:
        """When all fetched data is future relative to t_now, return []."""
        from mcp_quant_agent.mcp_servers.data.yfinance_source import get_price_history

        bars = _make_price_bars(dt.date(2022, 6, 20), dt.date(2022, 6, 30))
        with patch(
            "mcp_quant_agent.mcp_servers.data.yfinance_source._fetch_raw_bars",
            return_value=bars,
        ):
            result = get_price_history(
                "AAPL", "2022-06-20", "2022-06-30", use_cache=False
            )

        assert result == [], "Expected empty list when all data is future"

    def test_all_past_data_returned_intact(self) -> None:
        from mcp_quant_agent.mcp_servers.data.yfinance_source import get_price_history

        bars = _make_price_bars(dt.date(2022, 6, 1), dt.date(2022, 6, 10))
        with patch(
            "mcp_quant_agent.mcp_servers.data.yfinance_source._fetch_raw_bars",
            return_value=bars,
        ):
            # use_cache=False: test the filter behaviour only, not the cache layer
            result = get_price_history(
                "AAPL", "2022-06-01", "2022-06-10", use_cache=False
            )

        assert len(result) == len(bars)


# ---------------------------------------------------------------------------
# 6–8. News temporal filter
# ---------------------------------------------------------------------------


class TestNewsTemporalFilter:
    """get_news_items must respect t_now for every call."""

    @pytest.fixture(autouse=True)
    def _setup_clock(self) -> None:
        self.clock = SimulationClock("2022-06-15")
        set_clock(self.clock)

    def test_news_after_t_now_excluded(self) -> None:
        from mcp_quant_agent.mcp_servers.data.finnhub_source import get_news_items

        news = _make_news_items(
            [
                "2022-06-14T09:00:00",
                "2022-06-15T00:00:00",  # t_now midnight — included
                "2022-06-16T08:00:00",  # future — excluded
            ]
        )
        with patch(
            "mcp_quant_agent.mcp_servers.data.finnhub_source._fetch_raw_news",
            return_value=news,
        ):
            result = get_news_items("AAPL", "2022-06-14", "2022-06-16")

        datetimes = [n["datetime"] for n in result]
        assert "2022-06-16T08:00:00" not in datetimes, "Future news item leaked through"

    def test_news_at_t_now_midnight_included(self) -> None:
        from mcp_quant_agent.mcp_servers.data.finnhub_source import get_news_items

        news = _make_news_items(["2022-06-15T00:00:00"])
        with patch(
            "mcp_quant_agent.mcp_servers.data.finnhub_source._fetch_raw_news",
            return_value=news,
        ):
            result = get_news_items("AAPL", "2022-06-15", "2022-06-15")

        assert len(result) == 1

    def test_all_future_news_returns_empty(self) -> None:
        from mcp_quant_agent.mcp_servers.data.finnhub_source import get_news_items

        news = _make_news_items(["2022-06-20T09:00:00", "2022-06-25T14:00:00"])
        with patch(
            "mcp_quant_agent.mcp_servers.data.finnhub_source._fetch_raw_news",
            return_value=news,
        ):
            result = get_news_items("AAPL", "2022-06-20", "2022-06-25")

        assert result == []


# ---------------------------------------------------------------------------
# 9. Adjusted-price leakage guard
# ---------------------------------------------------------------------------


class TestAdjustedPriceLeakage:
    """yfinance must be called with auto_adjust=False.

    Using auto_adjust=True (yfinance default) encodes future split/dividend
    adjustment factors into historical prices — a look-ahead form.  We assert
    that our wrapper always requests unadjusted prices.

    See docs/DECISIONS.md for the full rationale.
    """

    @pytest.fixture(autouse=True)
    def _setup_clock(self) -> None:
        set_clock(SimulationClock("2022-06-15"))

    def test_yfinance_called_with_auto_adjust_false(self) -> None:
        from mcp_quant_agent.mcp_servers.data.yfinance_source import _fetch_raw_bars

        mock_df = MagicMock()
        mock_df.empty = True

        with patch("yfinance.download", return_value=mock_df) as mock_dl:
            _fetch_raw_bars("AAPL", "2022-06-01", "2022-06-15", "1d")

        call_kwargs = mock_dl.call_args.kwargs
        assert call_kwargs.get("auto_adjust") is False, (
            f"yfinance.download was NOT called with auto_adjust=False. "
            f"This risks leaking future split/dividend adjustments. "
            f"kwargs received: {call_kwargs}"
        )

    def test_get_price_history_fetches_day_after_inclusive_daily_end(self) -> None:
        from mcp_quant_agent.mcp_servers.data.yfinance_source import get_price_history

        with patch(
            "mcp_quant_agent.mcp_servers.data.yfinance_source._fetch_raw_bars",
            return_value=[],
        ) as mock_fetch:
            get_price_history(
                "AAPL",
                "2022-06-01",
                "2022-06-15",
                interval="1d",
                use_cache=False,
            )

        mock_fetch.assert_called_once_with("AAPL", "2022-06-01", "2022-06-16", "1d")


# ---------------------------------------------------------------------------
# 10–11. Parquet cache temporal filter
# ---------------------------------------------------------------------------


class TestCacheTemporalFilter:
    """The cache stores raw data; t_now filter is applied at read time.

    This test verifies that even if the parquet file on disk contains future
    data (e.g., because it was written during a previous session at a later
    t_now), the wrapper strips it at read time.
    """

    @pytest.fixture(autouse=True)
    def _setup_clock(self) -> None:
        self.clock = SimulationClock("2022-06-15")
        set_clock(self.clock)

    def test_cached_future_data_stripped_at_read(self, tmp_path: Path) -> None:
        from mcp_quant_agent.mcp_servers.data.cache import PriceCache

        bars_with_future = _make_price_bars(dt.date(2022, 6, 10), dt.date(2022, 6, 25))
        cache = PriceCache(cache_dir=tmp_path)
        cache.write("AAPL", "1d", bars_with_future)

        result = cache.read_filtered("AAPL", "1d")
        future_dates = [r["date"] for r in result if str(r["date"]) > "2022-06-15"]
        assert future_dates == [], (
            f"Cache leaked future data after read_filtered: {future_dates}"
        )

    def test_cache_includes_t_now_bar(self, tmp_path: Path) -> None:
        from mcp_quant_agent.mcp_servers.data.cache import PriceCache

        bars = _make_price_bars(dt.date(2022, 6, 10), dt.date(2022, 6, 20))
        cache = PriceCache(cache_dir=tmp_path)
        cache.write("AAPL", "1d", bars)

        result = cache.read_filtered("AAPL", "1d")
        dates = [r["date"] for r in result]
        assert "2022-06-15" in dates

    def test_cache_returns_empty_for_nonexistent_ticker(self, tmp_path: Path) -> None:
        from mcp_quant_agent.mcp_servers.data.cache import PriceCache

        cache = PriceCache(cache_dir=tmp_path)
        assert cache.read_filtered("NONEXISTENT", "1d") == []

    def test_cache_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        from mcp_quant_agent.mcp_servers.data.cache import PriceCache

        bars = _make_price_bars(dt.date(2022, 6, 1), dt.date(2022, 6, 10))
        cache = PriceCache(cache_dir=tmp_path)
        cache.write("AAPL", "1d", bars)

        # All bars are past/present → all should survive the filter
        result = cache.read_filtered("AAPL", "1d")
        assert len(result) == len(bars)
        # Spot-check a value
        assert result[0]["date"] == bars[0]["date"]
