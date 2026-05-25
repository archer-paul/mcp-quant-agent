# CLAUDE.md — Operating manual for Claude Code

> Read this file fully before writing any code. It defines the architecture, the
> non-negotiable constraints, the conventions, and what "done" means. When a request
> conflicts with this file, follow this file and flag the conflict.

## What this project is

`mcp-quant-agent` is an MSc thesis (Imperial, supervisor C. Salvi):
**Autonomous Trading Agents via the Model Context Protocol.**

An LLM agent accesses market data, technical indicators, news sentiment and order
execution **only through MCP tool servers**, in a `perceive → reason → act → observe`
loop. The thesis is graded primarily on **rigour of evaluation**, not on orchestration
sophistication. Optimise every decision for that.

## The single most important rule: no look-ahead, by construction

We use an **"MCP-as-time-machine"** design:

- A simulation clock exposes `t_now`.
- Every MCP server (and every wrapper around an external data server) MUST refuse to
  return any datapoint dated strictly after `t_now`.
- The agent calls the **same MCP tools** in backtest and in live; only the clock advances.
- Look-ahead must be **structurally impossible**, never "avoided by being careful".

**Test-Driven Development is mandatory here.** Before implementing the data layer, write
tests that FAIL if any tool returns a bar/news item dated `> t_now`. These tests
(`tests/test_clock_no_lookahead.py`, `tests/test_data_temporal_filter.py`) are the
backbone of the project's credibility. Never weaken them to make code pass.

Also guard against:
- **Adjusted-price leakage** (split/dividend-adjusted history encodes future corporate
  actions). Document how prices are sourced and adjusted.
- **Weight-memory contamination** (the LLM may "know" 2022–24 outcomes). Instruct the
  agent to follow a rule-based process over tools; prefer a partially post-cutoff test
  window where possible; document this limitation honestly (see KellyBench).

## Architecture (target)

- `clock.py` — the simulation clock. Small, central, heavily tested.
- `mcp_servers/data/` — thin MCP wrappers around external data servers
  (financial-datasets, finnhub). Each wrapper enforces the `t_now` filter.
- `mcp_servers/analytics/` — MCP tools for custom indicators, regime detection, backtest.
- `mcp_servers/execution/` — paper-trading MCP tools (place_order, get_portfolio, get_pnl),
  in-memory state.
- `agents/orchestrator.py` — custom Python loop. Single-agent first. Design it so adding
  roles (technical / sentiment / risk + a synthesising trader) is a clean extension, but
  DO NOT build multi-agent until the single-agent eval pipeline is complete.
- `backtest/baselines.py` — Buy & Hold, momentum (time-series), mean-reversion, in vectorbt.
- `backtest/engine.py` — event-driven, drip-feed engine for the agent loop.
- `eval/` — financial metrics, regime labelling, reasoning (CoT) metrics.
- `observability/langfuse_setup.py` — tracing wiring.

## Orchestration: LangGraph, introduced in the right order

We **do** use LangGraph — the author wants to learn it and it strengthens the thesis. The
risk is not LangGraph itself; it is learning it *on the critical path of the evaluation*.
So sequence it deliberately:

1. **Single-agent first, as a minimal LangGraph graph** (2–3 nodes: perceive → reason/act →
   observe). This is the canonical tutorial shape — enough to learn `StateGraph`, nodes,
   conditional edges and checkpointing without risk. Get the **evaluation pipeline running
   on top of this** before adding complexity.
2. **Then, in week 2, use LangGraph for what it is genuinely good at**: a *cyclic*
   multi-agent debate (e.g. Bull vs Bear researchers exchanging arguments, a risk node, a
   synthesising trader). This is awkward in a hand-rolled loop and clean in LangGraph — so
   the framework earns its place here rather than being overhead.

Keep nodes small and the state object explicit and typed. Do not over-abstract. If the
author is blocked learning a LangGraph concept, prefer the simplest graph that works and
add a short note in `docs/DECISIONS.md`.

## Observability: Langfuse (Cloud)

- Use **Langfuse Cloud**. Cost is covered by YC credits ($100/month) — that is far more
  than a thesis needs. **Do NOT self-host** (six containers of overhead for no benefit here).
- Use the **current** Langfuse Python SDK (v4+). Correct imports:
  - Drop-in OpenAI tracing: `from langfuse.openai import openai`
  - Decorator: `from langfuse import observe, get_client`
  - LangGraph: use Langfuse's native callback handler (pass it in the graph's config),
    do not hand-instrument every node.
  - **Do NOT** use `from langfuse.decorators import ...` — that module was removed in v4.
- Trace every decision as a structured object: inputs seen, tool calls + their real
  outputs, the model's CoT, the resulting order, latency, token count, cost.
- Use Langfuse **scores** to attach the reasoning-eval metrics (faithfulness, grounding,
  sophistication) back onto each trace — this gives you regime-segmented dashboards for free.
- The reasoning-eval metrics (`eval/reasoning.py`) consume these traces. Faithfulness and
  grounding are computed by comparing the *stated* reasoning/decision against the *actual*
  tool outputs and executed order — so the trace schema must capture both cleanly.

## LLM backbones & budget

- Agents run on the **OpenAI API** (key via env). This is the experiment fuel.
- Claude Code (this tool) is for **writing/refactoring code**, not for running experiments.
- Keep the LLM client behind a thin interface so swapping/adding a backbone
  (e.g. a second OpenAI model for an architecture comparison) is trivial.
- In debug/dev, default to the cheapest capable model and tiny date ranges. Reserve the
  expensive model for final runs. Never burn budget looping on a bug.

## External code living next to us

The supervisor's reference MCP code (the gen-ai week-3 tutorial) is on disk at:

```
..\gen-ai-imperial\code\week3_agent\
```

(i.e. a sibling of this repo's parent — `gen-ai-imperial` sits next to the folder
containing `mcp-quant-agent`). It contains a working FastMCP server (`trading_server.py`)
plus `market_data.py`, `indicators.py`, `sentiment.py`, `paper_trading.py`, `backtest.py`.

- **Use it as a reference and a starting point**, especially `indicators.py` and
  `sentiment.py` which we will reuse. Read it before reimplementing anything.
- **Do not edit files in that directory.** It is read-only reference. If you need to reuse
  code, copy it into `src/mcp_quant_agent/` and adapt (add the `t_now` filter, types, tests).
- Treat its `backtest.py` as vectorised/illustrative; our agent loop needs the event-driven
  engine instead. Reuse its metric formulas but harden them (Sortino/Calmar/bootstrap).

## Cloned reference repos (read-only) and papers

- `reference/` (gitignored) holds **clones of third-party repos** we study but do not ship:
  `TradingAgents/` (role architecture + Bull/Bear debate pattern),
  `mcp-financial-datasets/` and `mcp-finnhub/` (external MCP data servers we may run),
  `hedge_fund_agents/` (a simplified TradingAgents fork — shows what to cut).
  **Read these for reference; do not import from them directly.** If you reuse code, copy it
  into `src/` and adapt it (types, tests, `t_now` filter). Cite the source in the thesis.
- `papers/` holds the state-of-the-art PDFs plus `papers/README.md`, which summarises each
  paper and why it matters. **Read `papers/README.md` first** — it is the fast path to the
  methodology and metrics the thesis must align with (esp. StockBench, LiveTradeBench,
  KellyBench). Do not paraphrase large chunks of any paper into our docs; cite and link.

## Performance & best practices (the code must run fast locally)

- **Cache all market data to disk** (parquet) keyed by ticker/date/interval. Never re-hit
  an API for data already fetched. The backtest must be re-runnable offline.
- **Precompute indicators once** per dataset; the agent reads cached features through MCP,
  it does not recompute them every bar.
- Vectorise baseline computation (vectorbt / numpy / pandas); reserve the slow event-driven
  loop for the agent path only.
- Cache LLM responses in dev (keyed by prompt hash) so re-runs during debugging cost $0.
- Use `uv` for env/deps (fast). Pin versions in `pyproject.toml`.
- Profile before optimising; don't micro-optimise cold paths. Keep hot paths (per-bar loop,
  metric computation) allocation-light.
- Prefer pure functions for metrics and regime labelling (easy to test, easy to cache).

### General engineering conventions
- Python 3.11+. **Type hints everywhere.** Docstrings on every public function/class.
- **pytest** on every module. Integration tests of the loop and the anti-look-ahead tests
  are the priority suite.
- `ruff` for lint+format, `mypy` for types. Code must pass both before "done".
- Small, atomic commits with clear messages. One logical change per commit.
- No hidden global state except the explicitly-designed in-memory portfolio and clock.
- When a choice has trade-offs, write 2 lines explaining it (in code comment or
  `docs/DECISIONS.md`) before committing to it.
- Fail loudly: raise on unexpected states rather than silently returning defaults. A
  silent fallback that hides a leak or a missing tool call is the worst possible bug here.

## Definition of "done" for a task

1. Code is typed, documented, and passes `ruff` + `mypy`.
2. Tests exist and pass — including, where relevant, an anti-look-ahead assertion.
3. For data/metric code: a tiny reproducible example in a docstring or `scripts/`.
4. `docs/DECISIONS.md` updated if a non-obvious choice was made.
5. Nothing in `..\gen-ai-imperial\` was modified.

## Workflow with the author

- Work in **atomic tasks** aligned to `docs/PLAN.md` (roughly one per day).
- For any large task, **propose a plan first and wait for approval** before coding.
- Ask at most 3 clarifying questions when genuinely blocked; otherwise proceed and state
  your assumptions.
- The author manually reviews two things especially: the clock/anti-look-ahead layer and
  the metric computations. Make those maximally readable.
