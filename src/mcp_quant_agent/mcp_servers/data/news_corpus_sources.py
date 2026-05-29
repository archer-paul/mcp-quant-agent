"""Live population sources for the timestamped news corpus."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any
from urllib.parse import urlparse

import requests

from mcp_quant_agent.config import settings
from mcp_quant_agent.mcp_servers.data.news_corpus_cache import (
    REQUIRED_NEWS_CORPUS_COLUMNS,
    _parse_published_at,
)

logger = logging.getLogger(__name__)

FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"
FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"

_PUBLICATION_METADATA_KEYS = (
    "article:published_time",
    "og:published_time",
    "datePublished",
    "date_published",
    "publishedDate",
    "publishDate",
    "published_time",
    "pubdate",
)
_MODIFIED_METADATA_KEYS = (
    "article:modified_time",
    "og:updated_time",
    "dateModified",
    "modifiedDate",
    "lastModified",
    "lastmod",
)


def _utc_naive_iso_from_unix(raw: Any) -> str:
    parsed = dt.datetime.fromtimestamp(float(raw), tz=dt.UTC).replace(tzinfo=None)
    return parsed.isoformat(timespec="seconds")


def fetch_finnhub_company_news(
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch ticker-specific company news and map it to corpus schema."""
    key = api_key if api_key is not None else settings.finnhub_api_key
    if not key:
        raise RuntimeError("FINNHUB_API_KEY is not set.")

    response = requests.get(
        FINNHUB_COMPANY_NEWS_URL,
        params={
            "symbol": ticker.upper(),
            "from": start_date,
            "to": end_date,
            "token": key,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Finnhub response for {ticker}: {payload!r}")

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("datetime") in (None, ""):
            continue
        try:
            published_at = _utc_naive_iso_from_unix(item["datetime"])
        except (TypeError, ValueError, OverflowError, OSError):
            continue
        rows.append(
            {
                "ticker": ticker.upper(),
                "published_at": published_at,
                "title": str(item.get("headline") or ""),
                "body": str(item.get("summary") or ""),
                "source": str(item.get("source") or "Finnhub"),
                "url": str(item.get("url") or ""),
            }
        )
    return dedupe_articles(rows)


def dedupe_articles(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate corpus rows by URL when present, otherwise title/timestamp."""
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip().lower()
        published_at = _parse_published_at(item.get("published_at")).isoformat(
            timespec="seconds"
        )
        key = (url, "", "") if url else ("", published_at, title)
        if key in seen:
            continue
        seen.add(key)
        clean = {col: str(item.get(col) or "") for col in REQUIRED_NEWS_CORPUS_COLUMNS}
        clean["ticker"] = clean["ticker"].upper()
        clean["published_at"] = published_at
        deduped.append(clean)
    return sorted(deduped, key=lambda row: str(row["published_at"]))


def _first_metadata_timestamp(
    metadata: dict[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        raw = metadata.get(key)
        if raw:
            return str(raw)
    return None


def _source_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.") if host else "unknown"


def _firecrawl_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _firecrawl_tbs(start_date: str, end_date: str) -> str:
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    return (
        "cdr:1,"
        f"cd_min:{start.month}/{start.day}/{start.year},"
        f"cd_max:{end.month}/{end.day}/{end.year}"
    )


def _normalise_firecrawl_search_payload(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("news", "web"):
        raw_rows = data.get(key, [])
        if isinstance(raw_rows, list):
            rows.extend(item for item in raw_rows if isinstance(item, dict))
    return rows


def fetch_firecrawl_news_search(
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    query: str | None = None,
    limit: int = 10,
    api_key: str | None = None,
    include_domains: list[str] | None = None,
    allow_modified_timestamp: bool = False,
    allow_search_result_date: bool = True,
    timeout: float = 60.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Search Firecrawl news and accept only rows with a usable timestamp.

    Publication metadata is preferred.  A modified timestamp can be enabled as
    a conservative fallback: it may delay article availability but does not
    make an article visible before that modification time.
    """
    key = api_key if api_key is not None else settings.firecrawl_api_key
    if not key:
        raise RuntimeError("FIRECRAWL_API_KEY is not set.")
    body: dict[str, Any] = {
        "query": query or f"{ticker} stock news",
        "limit": limit,
        "sources": ["news"],
        "country": "US",
        "tbs": _firecrawl_tbs(start_date, end_date),
        "timeout": int(timeout * 1000),
        "ignoreInvalidURLs": True,
        "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
    }
    if include_domains:
        body["includeDomains"] = include_domains
    response = requests.post(
        FIRECRAWL_SEARCH_URL,
        headers=_firecrawl_headers(key),
        json=body,
        timeout=timeout + 5,
    )
    response.raise_for_status()
    rows = _normalise_firecrawl_search_payload(response.json())

    articles: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        timestamp_kind = "published_metadata"
        raw_ts = _first_metadata_timestamp(metadata, _PUBLICATION_METADATA_KEYS)
        if raw_ts is None and allow_modified_timestamp:
            raw_ts = _first_metadata_timestamp(metadata, _MODIFIED_METADATA_KEYS)
            timestamp_kind = "modified_metadata"
        if raw_ts is None and allow_search_result_date:
            raw_ts = str(row.get("date") or "")
            timestamp_kind = "search_result_date"
        if not raw_ts:
            audit.append({"url": row.get("url"), "status": "rejected_no_timestamp"})
            continue
        try:
            published_at = _parse_published_at(raw_ts)
        except ValueError:
            audit.append(
                {
                    "url": row.get("url"),
                    "status": "rejected_bad_timestamp",
                    "timestamp": raw_ts,
                }
            )
            continue
        if not (start <= published_at.date() <= end):
            audit.append(
                {
                    "url": row.get("url"),
                    "status": "rejected_out_of_range",
                    "timestamp": published_at.isoformat(),
                }
            )
            continue
        url = str(row.get("url") or metadata.get("sourceURL") or metadata.get("url") or "")
        title = str(row.get("title") or metadata.get("title") or "")
        body_text = str(row.get("markdown") or row.get("snippet") or row.get("description") or "")
        articles.append(
            {
                "ticker": ticker.upper(),
                "published_at": published_at.isoformat(timespec="seconds"),
                "title": title,
                "body": " ".join(body_text.split())[:4000],
                "source": f"{_source_from_url(url)}:{timestamp_kind}",
                "url": url,
            }
        )
        audit.append(
            {
                "url": url,
                "status": "accepted",
                "timestamp_kind": timestamp_kind,
                "published_at": published_at.isoformat(timespec="seconds"),
            }
        )
    return dedupe_articles(articles), audit


def scrape_article_with_firecrawl(
    url: str,
    *,
    api_key: str | None = None,
    timeout: float = 45.0,
) -> dict[str, Any]:
    """Scrape one article URL with Firecrawl for body enrichment.

    The returned metadata is audit-only.  Corpus ``published_at`` must still
    come from the primary news source, not Firecrawl crawl metadata.
    """
    key = api_key if api_key is not None else settings.firecrawl_api_key
    if not key:
        raise RuntimeError("FIRECRAWL_API_KEY is not set.")
    response = requests.post(
        FIRECRAWL_SCRAPE_URL,
        headers=_firecrawl_headers(key),
        json={
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "timeout": int(timeout * 1000),
        },
        timeout=timeout + 5,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        return {"body": "", "metadata": {}}
    return {
        "body": str(data.get("markdown") or data.get("content") or ""),
        "metadata": data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
    }


def enrich_articles_with_firecrawl(
    articles: list[dict[str, Any]],
    *,
    max_articles: int,
    api_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fill article bodies from Firecrawl for up to *max_articles* URLs."""
    enriched: list[dict[str, Any]] = [dict(article) for article in articles]
    audit: list[dict[str, Any]] = []
    n_attempted = 0
    for article in enriched:
        if n_attempted >= max_articles:
            break
        url = str(article.get("url") or "").strip()
        if not url:
            continue
        n_attempted += 1
        try:
            scraped = scrape_article_with_firecrawl(url, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firecrawl enrichment failed for %s: %s", url, exc)
            audit.append({"url": url, "status": "error", "error": str(exc)})
            continue
        body = " ".join(str(scraped.get("body") or "").split())
        if body:
            article["body"] = body[:4000]
        audit.append(
            {
                "url": url,
                "status": "ok" if body else "empty",
                "body_chars": len(body),
                "metadata": scraped.get("metadata", {}),
            }
        )
    return enriched, audit
