# Design decisions (ADR-lite)

> One short entry per non-obvious choice: context → decision → consequence. Newest on top.
> Keep it terse; this is the audit trail for the thesis methodology chapter.

---

### 2026-05 — Langfuse SDK version: v3+ (CLAUDE.md says "v4+")
- **Context:** CLAUDE.md specifies "SDK v4+" but the PyPI package ``langfuse`` is
  at v3.x (as of May 2026 the package version numbering may have advanced).
  The import paths in CLAUDE.md (``from langfuse.openai import openai``,
  ``from langfuse import observe, get_client``) match the v3 API.
- **Decision:** Use ``langfuse>=3.0,<5`` and the v3+ import paths.  If v4 is
  released and breaks the imports, update then (it will be a one-file change
  in ``observability/langfuse_setup.py``).
- **Consequence:** Minor uncertainty on exact SDK version; imports are correct.

### 2026-05 — Unadjusted prices (auto_adjust=False) to prevent split leakage
- **Context:** yfinance's default ``auto_adjust=True`` applies the *current*
  split/dividend adjustment factors to historical prices.  In a backtest set in
  2022, this encodes splits that occurred in 2023–2025 — a look-ahead bias.
  CLAUDE.md explicitly calls this out as "adjusted-price leakage".
- **Decision:** Always call ``yf.download(..., auto_adjust=False)``.  Work with
  raw (unadjusted) prices throughout.
- **Consequence:** Daily returns are not corrected for stock splits, which
  slightly distorts long-period comparisons.  This is clearly preferable to
  silent future-corporate-action leakage.  Document in the thesis methodology.

### 2026-05 — Module-level clock singleton (explicitly permitted)
- **Context:** MCP tool calls arrive from external processes over stdio/HTTP.
  There is no Python call stack through which a clock could be
  dependency-injected at call time.
- **Decision:** Use a module-level ``_CLOCK`` in ``clock.py``, set via
  ``set_clock()`` at server/backtest startup.  CLAUDE.md explicitly permits
  this singleton (along with the portfolio) as "designed global state".
- **Consequence:** Startup order matters: ``set_clock()`` must be called before
  any data tool is invoked.  Tests use the ``autouse`` fixture in conftest.py
  to reset ``_CLOCK`` to ``None`` between tests.

### 2026-05 — filter_rows vs assert_not_future: two modes of the same guarantee
- **Context:** We need both a "raise loudly" guard (for single-item checks
  where a future timestamp is unambiguously a bug) and a "silently drop"
  batch filter (for slicing a cached price series to the backtest window).
- **Decision:** ``assert_not_future`` for single items (raises FutureDataError),
  ``filter_rows`` for batch data (drops future rows, returns filtered list).
  Both call the same internal ``_to_datetime`` comparison.
- **Consequence:** Clear semantic distinction between "this is a bug" and
  "this is the expected trim".  Data wrappers always use ``filter_rows``.

### 2026-05 — Cache stores raw data; t_now filter applied at read time
- **Context:** We want to cache market data to avoid re-hitting APIs on each
  backtest run.  The same cache file should be usable across backtest runs at
  different ``t_now`` values (e.g. rolling backtests).
- **Decision:** Parquet cache stores raw API data without any t_now filter.
  The filter is applied by ``PriceCache.read_filtered()`` / ``clock.filter_rows()``
  at read time.
- **Consequence:** A backtest run at ``t_now = 2022-06-15`` and one at
  ``t_now = 2022-06-16`` can share the same parquet file; no re-fetch needed.

### 2026-05 — Execution server accepts explicit price parameter
- **Context:** The execution MCP server needs a price for each order.  Options:
  (a) look up live price internally, (b) accept price as a parameter.
- **Decision:** Accept ``price`` as an explicit ``place_order`` parameter.
  The orchestrator fetches the t_now close from the data server and passes it.
- **Consequence:** Execution server is independent of the data server (no
  circular deps).  Price assumptions are explicit in the agent's reasoning
  and traceable in Langfuse.

### 2026-05 — Regime detection: rule-based (vol + trend) over HMM
- **Context:** HMM would give probabilistic regime assignments but requires
  fitting parameters on historical data — which, done naively on the full
  training set, leaks future information (look-ahead via fitted transitions).
- **Decision:** Rule-based regime labelling: rolling realised vol (20 bars) +
  rolling momentum (60 bars).  Thresholds are fixed before seeing the data.
  Priority: HIGH_VOL > BULL > BEAR > RANGE.
- **Consequence:** Fully causal at every bar (no parameter fitting).  Less
  theoretically elegant than HMM but more appropriate for a thesis making
  look-ahead claims.  HMM noted as a future extension.

### 2026-05 — Bootstrap CI for Sharpe (iid, not stationary)
- **Context:** KellyBench (arXiv:2604.27865) shows that single-run Sharpe
  estimates can vary wildly.  Politis & Romano (1994) stationary bootstrap
  is the gold standard for weakly autocorrelated returns.
- **Decision:** Use iid bootstrap for now (simpler, sufficient for daily returns
  which have weak autocorrelation).  Note the approximation in the thesis.
- **Consequence:** CI widths may be slightly underestimated for strategies with
  momentum.  ``bootstrap_sharpe_ci`` is in ``eval/financial.py`` and always
  reported alongside the point estimate.

### 2026-05 — Orchestration: LangGraph, sequenced to de-risk learning
- **Context:** Author is new to LangGraph but wants to learn it and maximise thesis quality.
  Risk is learning it on the evaluation critical path.
- **Decision:** Use LangGraph. Start with a minimal 2–3 node single-agent graph; get the full
  evaluation pipeline working on it; then use LangGraph's cyclic graphs for the Bull/Bear
  multi-agent debate in week 2.
- **Consequence:** Framework learning happens on the simplest possible surface first; the
  multi-agent complexity lands only after evaluation is solid.

### 2026-05 — Observability: Langfuse Cloud (not self-hosted)
- **Context:** YC credits cover ~$100/month of Langfuse. Self-hosting is 6 containers.
- **Decision:** Langfuse Cloud. SDK v3+ (``from langfuse.openai import openai``; native
  LangGraph callback). Attach reasoning-eval scores to traces for regime dashboards.
- **Consequence:** No infra overhead; generous trace budget; regime-segmented eval for free.

### 2026-05 — Anti-look-ahead: "MCP-as-time-machine" + TDD
- **Context:** Look-ahead bias is the cardinal sin and KellyBench shows leakage is common.
- **Decision:** A single ``t_now`` clock; all MCP tools refuse data dated > t_now; same tools in
  backtest and live. Anti-leak tests written before implementation (TDD).
- **Consequence:** Look-ahead is structurally impossible, not discipline-dependent. Strong
  methodology claim for the thesis.

### 2026-05 — Asset scope: US equities primary, crypto as robustness extension
- **Context:** Regime labelling and baselines are cleaner in equities; SOTA (StockBench) is
  equity-centric. Crypto is 24/7 and noisier.
- **Decision:** US equities daily (2022→2024, ≥2 regimes) on the critical path; crypto only as
  an off-critical-path generalisation test.
- **Consequence:** Cleaner regime analysis now; an impressive "cross-asset generalisation"
  chapter if time allows, an honest "future work" note if not.

### 2026-05 — Evaluation priority over architecture richness
- **Context:** Thesis graded primarily on rigour of evaluation; TradingAgents already exists.
- **Decision:** Differentiator = MCP tool layer + regime-aware CoT evaluation
  (faithfulness, grounding, sophistication), multi-seed + bootstrap. Multi-agent is a
  robustness bonus, not a prerequisite.
- **Consequence:** If time runs short, cut multi-agent (week 2 days), never the eval rigour.
