# mcp-quant-agent

**Autonomous Trading Agents via the Model Context Protocol (MCP)**
MSc Mathematics & Finance thesis · Imperial College London · Supervisor: Cristopher Salvi

An LLM-based trading agent that accesses market data, technical indicators, news
sentiment and order execution **exclusively through MCP tool servers**, inside an
agentic `perceive → reason → act → observe` loop. The project's primary contribution
is **rigorous, regime-aware evaluation** of such an agent — financial performance,
latency/cost, and reasoning (chain-of-thought) quality — benchmarked against classical
systematic baselines (Buy & Hold, momentum, mean-reversion).

> **Research question.** Can an MCP-driven LLM agent translate its reasoning into
> profitable, risk-aware decisions, and how does the *quality and faithfulness* of that
> reasoning vary across market regimes?

---

## Why this is not "just another trading bot"

Two design choices carry the scientific weight:

1. **MCP-as-time-machine (no look-ahead, by construction).** A single simulation clock
   `t_now` governs the whole system. Every MCP server — including thin wrappers around
   external data servers — refuses to return any datapoint dated after `t_now`. The agent
   calls the *exact same tools* in backtest and in live; only the clock differs. Look-ahead
   bias is therefore *structurally impossible*, not merely avoided by discipline. Tests
   assert this and fail if any future bar leaks.

2. **Regime-conditioned CoT evaluation.** Beyond Sharpe/Sortino/Calmar/maxDD, we score the
   agent's reasoning for **faithfulness** (does the executed order match the stated
   reasoning?), **grounding** (does it cite tool outputs that were actually returned, or
   hallucinate numbers?) and **sophistication** (a compact expert rubric). All metrics are
   reported *per market regime* (bull / bear / range / high-vol).

This directly targets the *knowledge–action gap* documented in KellyBench (Thomas Grady et al.):
models articulate good strategies but leak data, fail to adapt to regime shifts, and don't
execute what they reason. Our harness is built to measure exactly that gap.

---

## Architecture (high level)

```
            ┌───────────────────────────────────────────────┐
            │     Orchestrator (custom Python loop)         │
            │     perceive → reason → act → observe         │
            │     single-agent now; extensible to roles     │
            └───────────────┬───────────────────────────────┘
                            │  tool calls over MCP
        ┌───────────────────┼────────────────────────┐
        ▼                   ▼                        ▼
  MCP data wrappers     MCP analytics            MCP execution
  (financial-datasets,  (indicators, regime,     (place_order,
   finnhub) + t_now     backtest)                portfolio, pnl)
   filter                                        paper, in-memory
        │
        ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Simulation clock t_now drives data exposure. The SAME   │
  │  MCP tools serve backtest and live; only the clock moves.│
  └──────────────────────────────────────────────────────────┘

  Observability: Langfuse traces every decision (CoT, tool calls, latency, cost).
```

Full rationale, planning and evaluation design live in `docs/PLAN.md`.

---

## Repository layout

```
mcp-quant-agent/
├── README.md
├── CLAUDE.md                  # operating manual for Claude Code (read first)
├── pyproject.toml
├── .env.example
├── .gitignore
├── docs/
│   ├── PLAN.md                # 2-week plan, architecture, evaluation design
│   └── DECISIONS.md           # running log of design decisions (ADR-lite)
├── papers/                    # state-of-the-art PDFs + annotated README (read first)
│   └── README.md
├── reference/                 # gitignored clones of third-party repos (read-only)
├── src/mcp_quant_agent/
│   ├── clock.py               # simulation clock t_now  (CRITICAL, anti-look-ahead)
│   ├── mcp_servers/
│   │   ├── data/              # wrappers around external MCP data servers + t_now filter
│   │   ├── analytics/         # indicators, regime detection, backtest tools
│   │   └── execution/         # paper-trading order/portfolio/pnl tools
│   ├── agents/
│   │   ├── orchestrator.py    # perceive→reason→act→observe loop
│   │   └── roles/             # (later) technical / sentiment / risk / trader
│   ├── backtest/
│   │   ├── baselines.py       # buy&hold, momentum, mean-reversion (vectorbt)
│   │   └── engine.py          # event-driven drip-feed engine for the agent loop
│   ├── eval/
│   │   ├── financial.py       # sharpe, sortino, calmar, maxdd, turnover, bootstrap
│   │   ├── regime.py          # regime labelling (vol + trend buckets)
│   │   └── reasoning.py       # faithfulness, grounding, sophistication rubric
│   ├── observability/
│   │   └── langfuse_setup.py  # tracing wiring (Cloud free tier by default)
│   └── config.py
├── tests/
│   ├── test_clock_no_lookahead.py   # MUST exist before implementation (TDD)
│   ├── test_data_temporal_filter.py
│   └── ...
├── scripts/
│   ├── run_baselines.py
│   ├── run_agent_backtest.py
│   └── reproduce_all.py       # one command to regenerate every result
└── data/                      # gitignored; timestamped news corpus, caches
```

---

## Quickstart

```bash
# 1. Python env (uv recommended for speed; venv works too)
uv venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# 2. Secrets
cp .env.example .env                          # then fill in keys

# 3. Sanity: anti-look-ahead tests must pass before anything else
pytest tests/test_clock_no_lookahead.py -v

# 4. Baselines (your yardstick — run these before judging the agent)
python scripts/run_baselines.py --universe AAPL MSFT NVDA JPM XOM --start 2022-01-01 --end 2024-12-31

# 5. Agent backtest
python scripts/run_agent_backtest.py --config configs/single_agent.yaml
```

---

## Data & scope

- **Primary:** US equities, daily bars (yfinance / Finnhub via MCP), 2022→2024 to span
  ≥2 regimes (2022 bear → 2023–24 bull, plus high-vol episodes).
- **Extension (robustness, off the critical path):** crypto, to test whether the agent
  and its CoT generalise across asset classes.
- News sentiment uses a **timestamped corpus** (scraped once with Firecrawl) served
  filtered by `t_now` — no temporal leakage.

---

## Status

Early development. See `docs/PLAN.md` for the day-by-day plan and `docs/DECISIONS.md`
for the design-decision log.

## License

Academic use. To be finalised.
