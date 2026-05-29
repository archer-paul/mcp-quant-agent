"""Project-wide configuration, loaded from environment / .env file.

All settings live here so they are typed, documented, and validated in one place.
No other module reads ``os.environ`` directly — they import ``settings`` from here.

Usage
-----
>>> from mcp_quant_agent.config import settings
>>> settings.agent_model_dev
'gpt-4.1-mini'
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated project settings.

    Values are read from environment variables (case-insensitive) and
    optionally from a ``.env`` file in the project root.  See ``.env.example``
    for the full list of variables.
    """

    # ── LLM backbone ──────────────────────────────────────────────────────────
    # Agents run on OpenAI. Claude Code (this tool) is for writing code only.
    openai_api_key: str = Field(
        default="", description="OpenAI API key for agent runs."
    )
    agent_model_dev: str = Field(
        default="gpt-4.1-mini",
        description="Cheap model for dev/debug. Never burn budget looping on a bug.",
    )
    agent_model_final: str = Field(
        default="gpt-4.1",
        description="Capable model for final evaluation runs only.",
    )

    # ── Market data ───────────────────────────────────────────────────────────
    finnhub_api_key: str = Field(default="", description="Finnhub API key.")
    financial_datasets_api_key: str = Field(
        default="", description="financial-datasets.ai API key."
    )

    # ── Observability — Langfuse Cloud ────────────────────────────────────────
    # Do NOT self-host (six containers of overhead for no benefit). YC credits cover $100/mo.
    langfuse_public_key: str = Field(default="", description="Langfuse public key.")
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key.")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host. Always Langfuse Cloud — do NOT self-host.",
    )

    # ── News scraping ─────────────────────────────────────────────────────────
    firecrawl_api_key: str = Field(
        default="",
        description=(
            "Firecrawl API key for optional scraping experiments. Not recommended "
            "for the thesis news corpus because crawl time is not publication time."
        ),
    )

    # ── Caching ───────────────────────────────────────────────────────────────
    cache_dir: Path = Field(
        default=Path("./data/cache"),
        description="Root directory for parquet caches and LLM response caches.",
    )
    llm_cache: bool = Field(
        default=True,
        description=(
            "Cache LLM responses by prompt hash in dev so re-runs cost $0. "
            "Disable for final runs to ensure fresh responses."
        ),
    )

    # ── Transaction costs ─────────────────────────────────────────────────────
    transaction_cost_bps: float = Field(
        default=10.0,
        description=(
            "One-way transaction cost in basis points applied to every buy and sell. "
            "10 bps = 5 bps commission + 5 bps slippage, matching baselines.py. "
            "Must be identical for agent AND baselines for a fair comparison."
        ),
    )

    # ── News corpus (experimental, off the critical path) ─────────────────────
    news_corpus_enabled: bool = Field(
        default=False,
        description=(
            "Enable the timestamped news corpus for the sentiment analyst. "
            "OFF by default — the news analyst falls back to sentiment_unavailable. "
            "Only activate after the corpus passes the anti-lookahead test on real "
            "data (see docs/DECISIONS.md 'News integration scaffold'). The corpus "
            "is filtered by published_at <= t_now at read time."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Ignore unknown env vars (e.g. LANGFUSE_BASE_URL which is a common
        # alias — the SDK reads LANGFUSE_HOST; unknown vars are silently dropped).
        extra="ignore",
    )


# Module-level singleton — all modules import this rather than constructing their own.
settings: Settings = Settings()
