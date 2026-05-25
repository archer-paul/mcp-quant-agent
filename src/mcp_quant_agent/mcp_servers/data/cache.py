"""Disk-based parquet cache for market data.

Design
------
The cache stores **raw data without any t_now filter**.  This is intentional:
the cache is a persistent mirror of the API data, reusable across many backtest
runs with different ``t_now`` values.  The t_now filter is applied at *read time*
by :meth:`PriceCache.read_filtered` (which calls ``clock.filter_rows``).

Cache layout (all under ``CACHE_DIR``)::

    prices/{ticker}_{interval}.parquet   — OHLCV bars, "date" column
    news/{ticker}_news.parquet           — news items, "datetime" column

Parquet is used because it is columnar (fast date-range queries), self-describing
(preserves dtypes), and natively supported by pandas + pyarrow.

Re-running the backtest offline
--------------------------------
Once data is cached for a given date range, the backtest can run with no
network access.  The cache is gitignored (``data/`` is in ``.gitignore``).

Updating the cache (incremental fetch)
----------------------------------------
:meth:`PriceCache.update` fetches only the missing tail (from ``latest_cached_date``
to ``end_date``) and merges with the existing cache.  This avoids re-hitting the
API for data already on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mcp_quant_agent.clock import get_clock

logger = logging.getLogger(__name__)


class PriceCache:
    """Parquet-backed cache for OHLCV price bars.

    Parameters
    ----------
    cache_dir:
        Root cache directory.  A ``prices/`` sub-directory is created
        automatically.  Defaults to ``./data/cache`` (matches ``.env.example``).

    Examples
    --------
    >>> from pathlib import Path
    >>> cache = PriceCache(cache_dir=Path("/tmp/test_cache"))
    >>> bars = [{"date": "2022-01-03", "open": 100.0, "high": 101.0,
    ...          "low": 99.0, "close": 100.5, "volume": 1_000_000}]
    >>> cache.write("AAPL", "1d", bars)
    >>> cached = cache.read_all("AAPL", "1d")
    >>> cached[0]["date"]
    '2022-01-03'
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        from mcp_quant_agent.config import settings

        root = cache_dir if cache_dir is not None else settings.cache_dir
        self._prices_dir: Path = Path(root) / "prices"
        self._prices_dir.mkdir(parents=True, exist_ok=True)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _path(self, ticker: str, interval: str) -> Path:
        """Return the parquet file path for (ticker, interval)."""
        # Sanitise ticker to avoid path-separator issues (e.g. BRK.B → BRK_B)
        safe_ticker = ticker.replace(".", "_").replace("/", "_")
        return self._prices_dir / f"{safe_ticker}_{interval}.parquet"

    # ── write ──────────────────────────────────────────────────────────────────

    def write(self, ticker: str, interval: str, bars: list[dict[str, Any]]) -> None:
        """Persist *bars* to the parquet cache (overwrites existing file).

        Parameters
        ----------
        ticker:
            Equity ticker symbol (e.g. ``"AAPL"``).
        interval:
            Bar interval string (e.g. ``"1d"``, ``"1h"``).
        bars:
            List of OHLCV bar dicts.  Each dict must contain a ``"date"`` key.
        """
        if not bars:
            logger.debug("write: empty bars for %s %s — skipping", ticker, interval)
            return
        df = pd.DataFrame(bars)
        # Ensure the date column is sorted ascending for efficient range queries
        df = df.sort_values("date").reset_index(drop=True)
        path = self._path(ticker, interval)
        df.to_parquet(path, index=False, compression="snappy")
        logger.debug("cache.write: %s %s → %d rows → %s", ticker, interval, len(df), path)

    def merge_and_write(
        self, ticker: str, interval: str, new_bars: list[dict[str, Any]]
    ) -> None:
        """Merge *new_bars* with any existing cache and write back.

        Duplicate dates are resolved by keeping the new row (data refresh).
        """
        existing = self.read_all(ticker, interval)
        if not existing:
            self.write(ticker, interval, new_bars)
            return
        df_old = pd.DataFrame(existing)
        df_new = pd.DataFrame(new_bars)
        merged = (
            pd.concat([df_old, df_new], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        path = self._path(ticker, interval)
        merged.to_parquet(path, index=False, compression="snappy")
        logger.debug(
            "cache.merge_and_write: %s %s → %d rows (was %d)",
            ticker, interval, len(merged), len(df_old),
        )

    # ── read ───────────────────────────────────────────────────────────────────

    def read_all(self, ticker: str, interval: str) -> list[dict[str, Any]]:
        """Return all cached bars for (ticker, interval) without any t_now filter.

        Returns an empty list if no cache file exists.
        """
        path = self._path(ticker, interval)
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        # Ensure date is stored as plain string (parquet may read it as timestamp)
        if "date" in df.columns and not pd.api.types.is_string_dtype(df["date"]):
            df["date"] = df["date"].astype(str).str[:10]
        return df.to_dict(orient="records")  # type: ignore[return-value]

    def read_filtered(
        self, ticker: str, interval: str
    ) -> list[dict[str, Any]]:
        """Return cached bars filtered to rows with date ≤ t_now.

        This is the main read method used by all data tools.  It calls
        ``get_clock().filter_rows(...)`` to enforce the anti-look-ahead
        guarantee — even if the parquet file contains future bars, they
        are stripped here.

        Raises
        ------
        RuntimeError
            If no simulation clock has been set (see ``clock.set_clock``).
        """
        bars = self.read_all(ticker, interval)
        if not bars:
            return []
        clock = get_clock()
        return clock.filter_rows(bars, date_key="date")

    def latest_cached_date(self, ticker: str, interval: str) -> str | None:
        """Return the ISO-8601 date string of the most recent cached bar.

        Returns ``None`` if no cache exists.  Used by ``update()`` to decide
        the fetch start date for incremental updates.
        """
        bars = self.read_all(ticker, interval)
        if not bars:
            return None
        return str(sorted(b["date"] for b in bars)[-1])

    def exists(self, ticker: str, interval: str) -> bool:
        """Return ``True`` if a cache file exists for (ticker, interval)."""
        return self._path(ticker, interval).exists()


class NewsCache:
    """Parquet-backed cache for timestamped news items.

    The ``datetime`` column stores ISO-8601 strings (``"YYYY-MM-DDTHH:MM:SS"``).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        from mcp_quant_agent.config import settings

        root = cache_dir if cache_dir is not None else settings.cache_dir
        self._news_dir: Path = Path(root) / "news"
        self._news_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        safe_ticker = ticker.replace(".", "_").replace("/", "_")
        return self._news_dir / f"{safe_ticker}_news.parquet"

    def write(self, ticker: str, items: list[dict[str, Any]]) -> None:
        """Persist news *items* (overwrite)."""
        if not items:
            return
        df = pd.DataFrame(items)
        df = df.sort_values("datetime").reset_index(drop=True)
        df.to_parquet(self._path(ticker), index=False, compression="snappy")

    def read_all(self, ticker: str) -> list[dict[str, Any]]:
        """Return all cached news items (no t_now filter)."""
        path = self._path(ticker)
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")  # type: ignore[return-value]

    def read_filtered(self, ticker: str) -> list[dict[str, Any]]:
        """Return news items filtered to datetime ≤ t_now."""
        items = self.read_all(ticker)
        if not items:
            return []
        clock = get_clock()
        return clock.filter_rows(items, date_key="datetime")
