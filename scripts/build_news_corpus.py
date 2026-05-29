#!/usr/bin/env python
"""Populate the timestamped news corpus from Finnhub, with optional Firecrawl enrichment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_quant_agent.mcp_servers.data.news_corpus_cache import NewsCorpusCache
from mcp_quant_agent.mcp_servers.data.news_corpus_sources import (
    enrich_articles_with_firecrawl,
    fetch_alpha_vantage_news_sentiment,
    fetch_finnhub_company_news,
    fetch_firecrawl_news_search,
)

app = typer.Typer(add_completion=False)


def _parse_tickers(raw: str) -> list[str]:
    return [ticker.strip().upper() for ticker in raw.split(",") if ticker.strip()]


@app.command()
def main(
    tickers: str = typer.Option(..., help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA."),
    start_date: str = typer.Option(..., help="ISO start date, inclusive."),
    end_date: str = typer.Option(..., help="ISO end date, inclusive."),
    corpus_dir: Path = typer.Option(
        Path("data/cache/news_corpus"),
        help="Output directory for <TICKER>.parquet files.",
    ),
    source: str = typer.Option(
        "finnhub",
        help="Primary source: finnhub, alpha-vantage, or firecrawl-search.",
    ),
    alpha_vantage_limit: int = typer.Option(
        1000,
        min=1,
        max=1000,
        help="Max Alpha Vantage feed rows per ticker.",
    ),
    firecrawl_limit_per_ticker: int = typer.Option(
        10,
        min=1,
        max=100,
        help="Search result limit per ticker when source=firecrawl-search.",
    ),
    allow_modified_timestamp: bool = typer.Option(
        False,
        help="Allow Firecrawl modified metadata as a conservative timestamp fallback.",
    ),
    allow_search_result_date: bool = typer.Option(
        True,
        help="Allow Firecrawl news result date when publication metadata is absent.",
    ),
    enrich_firecrawl: bool = typer.Option(
        False,
        help="Use Firecrawl to enrich article body text. Publication time still comes from Finnhub.",
    ),
    max_enrich_per_ticker: int = typer.Option(
        0,
        min=0,
        help="Max article URLs to enrich with Firecrawl per ticker.",
    ),
    dry_run: bool = typer.Option(False, help="Fetch and report counts without writing parquet."),
) -> None:
    """Fetch real timestamped news and write causal corpus parquet files."""
    cache = NewsCorpusCache(corpus_dir=corpus_dir)
    manifest: dict[str, Any] = {
        "source": source,
        "start_date": start_date,
        "end_date": end_date,
        "corpus_dir": str(corpus_dir),
        "alpha_vantage_limit": alpha_vantage_limit,
        "firecrawl_limit_per_ticker": firecrawl_limit_per_ticker,
        "allow_modified_timestamp": allow_modified_timestamp,
        "allow_search_result_date": allow_search_result_date,
        "enrich_firecrawl": enrich_firecrawl,
        "max_enrich_per_ticker": max_enrich_per_ticker,
        "tickers": {},
    }

    for ticker in _parse_tickers(tickers):
        audit: list[dict[str, Any]] = []
        if source == "finnhub":
            typer.echo(f"Fetching {ticker} {start_date}->{end_date} from Finnhub...")
            articles = fetch_finnhub_company_news(
                ticker,
                start_date=start_date,
                end_date=end_date,
            )
        elif source == "alpha-vantage":
            typer.echo(
                f"Fetching {ticker} {start_date}->{end_date} from Alpha Vantage..."
            )
            articles = fetch_alpha_vantage_news_sentiment(
                ticker,
                start_date=start_date,
                end_date=end_date,
                limit=alpha_vantage_limit,
            )
        elif source == "firecrawl-search":
            typer.echo(
                f"Searching {ticker} {start_date}->{end_date} with Firecrawl news..."
            )
            articles, audit = fetch_firecrawl_news_search(
                ticker,
                start_date=start_date,
                end_date=end_date,
                limit=firecrawl_limit_per_ticker,
                allow_modified_timestamp=allow_modified_timestamp,
                allow_search_result_date=allow_search_result_date,
            )
        else:
            raise typer.BadParameter(
                "source must be 'finnhub', 'alpha-vantage', or 'firecrawl-search'"
            )

        if enrich_firecrawl and max_enrich_per_ticker > 0 and articles:
            typer.echo(f"  Firecrawl enrichment: first {max_enrich_per_ticker} URL(s)")
            articles, enrich_audit = enrich_articles_with_firecrawl(
                articles,
                max_articles=max_enrich_per_ticker,
            )
            audit.extend({"phase": "enrich", **row} for row in enrich_audit)
        if not dry_run:
            cache.write_corpus(ticker, articles)
        manifest["tickers"][ticker] = {
            "n_articles": len(articles),
            "parquet": str(cache.parquet_path(ticker)),
            "firecrawl_audit": audit,
        }
        typer.echo(f"  {ticker}: {len(articles)} article(s)")

    if not dry_run:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = corpus_dir / f"_manifest_{start_date}_{end_date}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        typer.echo(f"Manifest: {manifest_path}")
    else:
        typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
