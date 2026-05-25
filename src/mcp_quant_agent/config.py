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
        default="", description="Firecrawl API key for timestamped news corpus."
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Module-level singleton — all modules import this rather than constructing their own.
settings: Settings = Settings()
