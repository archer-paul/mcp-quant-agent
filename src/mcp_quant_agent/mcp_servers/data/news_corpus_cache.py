"""Causal timestamped news corpus cache.

The corpus parquet stores raw articles without a t_now filter.  Reads are
filtered at access time by ``published_at <= get_clock().t_now`` and sorted
most-recent first.  The feature is off by default through
``settings.news_corpus_enabled`` and is not on the PM critical path.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

from mcp_quant_agent.clock import get_clock

logger = logging.getLogger(__name__)

REQUIRED_NEWS_CORPUS_COLUMNS = [
    "ticker",
    "published_at",
    "title",
    "body",
    "source",
    "url",
]

_CORPUS_DIR = Path("data/cache/news_corpus")
_PARQUET_SUFFIX = ".parquet"
_DEFAULT_MAX_ITEMS = 10


class NewsCorpusError(RuntimeError):
    """Base class for timestamped news corpus failures."""


class NewsCorpusUnavailable(NewsCorpusError):
    """Raised when a requested corpus file is absent in strict mode."""


def _parse_published_at(raw: Any) -> dt.datetime:
    """Parse publication timestamps to UTC-naive datetimes.

    Accepted formats:
    - ISO datetime, with or without timezone offset
    - ISO datetime ending in ``Z``
    - ``YYYY-MM-DD HH:MM:SS``
    - Unix seconds as int, float, or numeric string
    """
    if raw is None:
        raise ValueError("published_at is required")

    if isinstance(raw, dt.datetime):
        parsed = raw
    elif isinstance(raw, dt.date):
        parsed = dt.datetime.combine(raw, dt.time.min)
    elif isinstance(raw, int | float):
        parsed = dt.datetime.fromtimestamp(float(raw), tz=dt.UTC)
    else:
        text = str(raw).strip()
        if not text:
            raise ValueError("published_at is empty")
        try:
            parsed = dt.datetime.fromtimestamp(float(text), tz=dt.UTC)
        except (ValueError, OverflowError, OSError):
            iso_text = text.replace("Z", "+00:00") if text.endswith("Z") else text
            try:
                parsed = dt.datetime.fromisoformat(iso_text)
            except ValueError as exc:
                raise ValueError(f"Cannot parse published_at={raw!r}") from exc

    if parsed.tzinfo is not None:
        return parsed.astimezone(dt.UTC).replace(tzinfo=None)
    return parsed


def _normalise_t_now(raw: dt.datetime | dt.date | str | None) -> dt.datetime:
    if raw is None:
        return get_clock().t_now
    return _parse_published_at(raw)


def _safe_ticker(ticker: str) -> str:
    return ticker.upper().replace(".", "_").replace("/", "_")


def _article_excerpt(article: dict[str, Any], max_chars: int = 240) -> str:
    title = str(article.get("title") or "").strip()
    body = str(article.get("body") or "").strip()
    text = title if title else body
    if title and body:
        text = f"{title} - {body}"
    return " ".join(text.split())[:max_chars]


class NewsCorpusCache:
    """Parquet-backed, causally filtered news corpus."""

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        max_items: int = _DEFAULT_MAX_ITEMS,
    ) -> None:
        if corpus_dir is None:
            from mcp_quant_agent.config import settings

            corpus_dir = settings.news_corpus_dir
        self._dir = Path(corpus_dir)
        self.max_items = max_items

    def parquet_path(self, ticker: str) -> Path:
        return self._dir / f"{_safe_ticker(ticker)}{_PARQUET_SUFFIX}"

    def exists(self, ticker: str) -> bool:
        return self.parquet_path(ticker).exists()

    def read_all(self, ticker: str, *, strict: bool = True) -> list[dict[str, Any]]:
        """Read raw corpus rows for *ticker* without a t_now filter."""
        path = self.parquet_path(ticker)
        if not path.exists():
            if strict:
                raise NewsCorpusUnavailable(f"News corpus missing for {ticker}: {path}")
            return []

        try:
            import pandas as pd

            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            raise NewsCorpusError(f"Failed to read news corpus for {ticker}: {exc}") from exc

        missing = [col for col in REQUIRED_NEWS_CORPUS_COLUMNS if col not in df.columns]
        if missing:
            raise NewsCorpusError(
                f"News corpus for {ticker} missing required columns: {missing}"
            )
        return df[REQUIRED_NEWS_CORPUS_COLUMNS].to_dict(orient="records")  # type: ignore[return-value]

    def read_filtered(
        self,
        ticker: str,
        t_now: dt.datetime | dt.date | str | None = None,
        *,
        limit: int | None = None,
        strict: bool = True,
    ) -> list[dict[str, Any]]:
        """Return articles with ``published_at <= t_now``, newest first."""
        rows = self.read_all(ticker, strict=strict)
        t_now_dt = _normalise_t_now(t_now)
        filtered: list[tuple[dt.datetime, dict[str, Any]]] = []
        for row in rows:
            pub_dt = _parse_published_at(row.get("published_at"))
            if pub_dt <= t_now_dt:
                clean = {col: row.get(col, "") for col in REQUIRED_NEWS_CORPUS_COLUMNS}
                clean["ticker"] = str(clean["ticker"] or ticker).upper()
                clean["published_at"] = pub_dt.isoformat()
                filtered.append((pub_dt, clean))

        filtered.sort(key=lambda pair: pair[0], reverse=True)
        item_limit = self.max_items if limit is None else limit
        return [row for _, row in filtered[:item_limit]]

    def write_corpus(
        self,
        ticker: str,
        items: list[dict[str, Any]],
    ) -> None:
        """Write raw corpus rows for *ticker* without applying a date filter."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise NewsCorpusError("pandas is required to write the news corpus") from exc

        normalised_rows: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            missing = [col for col in REQUIRED_NEWS_CORPUS_COLUMNS if col not in item]
            if missing:
                raise NewsCorpusError(
                    f"Article {idx} for {ticker} missing required columns: {missing}"
                )
            pub_dt = _parse_published_at(item["published_at"])
            normalised_rows.append(
                {
                    "ticker": str(item["ticker"] or ticker).upper(),
                    "published_at": pub_dt.isoformat(),
                    "title": str(item["title"] or ""),
                    "body": str(item["body"] or ""),
                    "source": str(item["source"] or ""),
                    "url": str(item["url"] or ""),
                }
            )

        self._dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(normalised_rows, columns=REQUIRED_NEWS_CORPUS_COLUMNS)
        df.to_parquet(self.parquet_path(ticker), index=False, compression="snappy")
        logger.info("Wrote %d raw news corpus rows for %s", len(df), ticker.upper())


def get_news_corpus(
    ticker: str,
    *,
    limit: int | None = None,
    strict: bool = False,
    cache: NewsCorpusCache | None = None,
) -> dict[str, Any]:
    """Public flag-gated access to the timestamped news corpus.

    When disabled, this returns an explicit unavailable marker and an empty
    item list.  When enabled and ``strict=True``, a missing or invalid corpus
    raises; otherwise the caller receives an unavailable marker.
    """
    from mcp_quant_agent.config import settings

    ticker = ticker.upper()
    if not settings.news_corpus_enabled:
        return {
            "status": "sentiment_unavailable",
            "reason": "news_corpus_enabled=False",
            "ticker": ticker,
            "items": [],
            "items_count": 0,
        }

    corpus = cache or NewsCorpusCache()
    try:
        items = corpus.read_filtered(ticker, limit=limit, strict=True)
    except NewsCorpusError as exc:
        if strict:
            raise
        return {
            "status": "sentiment_unavailable",
            "reason": str(exc),
            "ticker": ticker,
            "items": [],
            "items_count": 0,
        }

    return {
        "status": "ok" if items else "sentiment_unavailable",
        "reason": "" if items else "no causal news articles",
        "ticker": ticker,
        "items": items,
        "items_count": len(items),
    }


def evidence_from_article(article: dict[str, Any], max_chars: int = 200) -> str:
    """Format one audit-friendly evidence item from a corpus article."""
    source = str(article.get("source") or "unknown")
    published_at = str(article.get("published_at") or "")
    excerpt = _article_excerpt(article, max_chars=max_chars)
    return f"{source} {published_at}: {excerpt}"[:max_chars]
