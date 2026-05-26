# Design decisions (ADR-lite)

### 2026-05 — Reasoning eval: faithfulness, grounding, sophistication (ÉTAPE 3)
- **Context:** KellyBench (arXiv:2604.27865) shows "knowledge-action gap" — models state
  correct reasoning but execute inconsistent actions.  StockBench evaluates output actions
  but not the chain-of-thought.  This thesis claims CoT eval as its key differentiator.
  Three complementary metrics are needed: consistency check (faithfulness), factual accuracy
  (grounding), and reasoning quality (sophistication).
- **Decision — Faithfulness:** Keyword-based intent extraction from the rationale text.
  Patterns match ``\bbuy\b``, ``\bsell\b``, ``\bhold\b``, ``\bnot buy\b``, etc. (word
  boundaries, case-insensitive).  ``_extract_intent`` returns the dominant signal or None.
  Faithfulness = fraction of decisions where ``intent == action``.  Decisions with no
  detectable signal are excluded (not penalised): an ambiguous rationale is not a bug.
  Rationale: a fully deterministic, zero-cost check that runs instantly on any
  ``decisions.jsonl`` without an API call.  Pattern list is conservative and auditable.
- **Decision — Grounding:** Regex extracts ``(label, float)`` pairs from the rationale
  (three patterns: parenthesised ``SMA (136.22)``, ``at/of/= X.xx``, ``value (label)``).
  Integers without decimal point are excluded (noise reduction: share counts, year numbers).
  Each claim is matched against the ``indicators`` dict within 1% relative tolerance, or
  against recent bar prices (open/high/low/close).  Grounding = ``n_grounded / n_claims``.
  Decisions with zero numeric claims score as "no signal" (not penalised).
  Rationale: directly tests whether numbers cited in the CoT are hallucinated or real.
- **Decision — Sophistication:** LLM judge (gpt-4.1-mini) with 4 criteria — risk_management,
  uncertainty_acknowledgement, regime_adaptation, coherence — each scored 0.0–1.0.
  ``response_format=json_object`` enforces structured output.  A random sample of
  ``max_sophistication_per_regime`` decisions is drawn per regime (default 50) to bound cost.
  Rationale: the 4-criterion rubric maps directly to the thesis claims (regime-adaptive
  reasoning, calibrated uncertainty).  Cost ≈ $0.00032/decision.
- **Decision — Aggregation:** All three metrics are aggregated BY REGIME (bull/bear/range/hv).
  This is the core thesis claim: does agent quality degrade in adversarial regimes?
- **Consequence:** The faithfulness/grounding metrics are cheap (regex + dict lookup) and
  deterministic — always re-runnable on ``decisions.jsonl`` for free.  Sophistication costs
  ~$0.50 for the full thesis run (5 tickers × 501 bars, 50 decisions sampled per regime).
  Regime-segmented tables are the primary thesis output.

### 2026-05 — decisions.jsonl schema: incremental write, full schema, flush-per-entry
- **Context:** The LLM backbone costs ~$2.37 for the thesis run.  If the process crashes
  mid-run, all paid-for decisions must be recoverable without re-calling the API.
- **Decision:** Write one JSON line per decision immediately after ``graph.invoke`` returns,
  call ``f.flush()`` after each write.  Schema includes: ``date``, ``ticker``, ``action``,
  ``quantity``, ``rationale``, ``fill``, ``regime``, ``indicators``, ``tool_outputs``,
  ``latency_ms``, ``errors``.  ``tool_outputs`` (actual data returned by MCP tools) is
  required for the grounding metric; ``rationale`` is the full CoT for faithfulness and
  sophistication.  Error decisions (``action="error"``) are written with the exception message
  and excluded by the reasoning metrics.
- **Consequence:** The file is always in a valid append-only state.  Reasoning metrics can be
  (re-)computed at any time without re-running the agent.  File path:
  ``runs/<run_id>/decisions.jsonl``; run ID format ``{backbone}_{timestamp}`` auto-generated.

### 2026-05 — Regime label stability: causal min-hold smoothing (default min_hold=3)
- **Context:** Diagnostic on AAPL/MSFT/NVDA 2022-2024 with 20d-trend v2 detector: 29-36%
  of regime runs last only 1 bar (single-day "spike"), 41-56% last ≤ 2 bars.  Thesis
  metrics are segmented by regime; with 30% of runs being 1-day episodes the segmentation
  becomes meaningless (every decision is in its own micro-regime).
- **Decision:** Add causal ``min_hold`` smoothing to ``label_regimes_v2``: a regime change
  is only confirmed after the new raw label has appeared for ``min_hold`` consecutive bars.
  Default ``min_hold=3`` (one trading week).  ``min_hold=1`` disables smoothing (identity).
  Implementation: ``_apply_min_hold_smoothing(raw_labels, min_hold)`` is O(n), fully causal
  (``smoothed[t]`` depends only on ``raw[0..t]``).  The raw label is preserved as
  ``regime_raw`` in the output dict for transparency.
  Test ``TestMinHoldSmoothing.test_smoothing_causal_with_future_extreme`` verifies the
  anti-lookahead guarantee with smoothing enabled.
- **Consequence:** Regime transitions are ~3× less frequent; longer regime episodes that
  are meaningful for segmentation.  K-day lag in detecting genuine transitions is the
  accepted trade-off.  Documented in the thesis as a design choice (not a flaw).

### 2026-05 — Regime detector v2: reactive 20d trend + causal vol threshold (default)
- **Context:** v1 used a 60-day trend window.  The 2022–2023 recovery was systematically
  delayed in v1's labels: the 60d lookback at the start of 2023 still reached into the
  Nov/Dec 2022 crash, keeping the label at BEAR or RANGE weeks after prices had recovered.
  Additionally, v1's HIGH_VOL threshold used the full-series 80th-percentile -- a
  look-ahead bias: future high-vol bars raise the threshold and can silently re-label
  past HIGH_VOL bars as RANGE/BULL when the series is extended.
- **Decision:** v2 (default) uses:
  1. **20-day trend** (bull/bear threshold: +/-2%) -- captures the current price direction.
     Verified empirically: v1/v2 disagree on 29 of 36 months across AAPL/MSFT/NVDA
     2022-2024; v2 pivots to BULL ~1 month earlier at 2022/2023 recoveries (e.g.
     AAPL/MSFT Feb 2023: v1=RNG/HV, v2=BULL).
  2. **Trailing 252-bar percentile** for the HIGH_VOL threshold -- fully causal.
     At bar t only vol data from bars 1..t is used.  The test
     ``TestRegimeLookAheadGuard.test_v2_causal_vol_threshold`` fails if a
     full-series percentile is used.
  3. 60d trend retained as diagnostic output field ``trend_long_60d`` (not used for labelling).
  4. v1 accessible via ``version="v1"`` for comparison.
  5. Vol threshold uses strict ``>`` (not ``>=``) to avoid all-HIGH_VOL in flat markets.
- **Window choices:** 20d = ~1 trading month (reactive); 252d vol lookback = ~1 year
  (stable baseline; first 252 bars use a shorter expanding window -- documented limitation).
- **Consequence:** v2 labels are meaningfully more reactive.  29/36 month-ticker pairs
  disagree between v1 and v2 (see ``results/regime_diagnostic_*.csv``).  The systematic
  lag in v1 makes regime-segmented evaluation misleading; v2 eliminates it.  The look-ahead
  bias in v1's vol threshold is structurally fixed by the trailing percentile.  Note: Jan 2023
  itself is BEAR in both detectors for AAPL (20d lookback still reaches Dec 2022 weakness);
  the pivot to BULL appears in Feb 2023, ~4 weeks faster than v1.

### 2026-05 — Langfuse SDK v4 API: no CallbackHandler, use openai drop-in + start_as_current_observation
- **Context:** Langfuse v4.6.1 removed ``langfuse.callback.CallbackHandler``.
  LangGraph tracing now uses OpenTelemetry.
- **Decision:** (a) ``from langfuse.openai import openai`` drop-in for all OpenAI
  calls -- traces model, tokens, cost, latency automatically.  (b) In ``observe_node``,
  use ``client.start_as_current_observation(...)`` to create a structured "agent"
  span with the full decision context (inputs, tool outputs, reasoning, fill).
  (c) ``get_langfuse_handler()`` returns None (kept for API compatibility).
- **Consequence:** All LLM calls are traced.  Decision context is captured in a
  separate "agent" observation.  The Langfuse dashboard shows both.

### 2026-05 — pydantic-settings: extra="ignore" for .env forward-compatibility
- **Context:** The .env file may contain keys not in ``Settings`` (e.g.
  ``LANGFUSE_BASE_URL``).  pydantic-settings raises ValidationError by default.
- **Decision:** Add ``extra="ignore"`` to ``model_config`` in ``Settings``.
  Unknown env vars are silently dropped; the application uses only declared fields.
- **Consequence:** The .env file can have extra keys for other tools without
  breaking the app.  Typos in env var names are silently ignored (acceptable
  for a research project with a small team).

> One short entry per non-obvious choice: context → decision → consequence. Newest on top.
> Keep it terse; this is the audit trail for the thesis methodology chapter.

---

### 2026-05 — UTC-naive internal timestamps (timezone-aware inputs converted first)
- **Context:** ``_to_datetime`` stripped timezone with ``.replace(tzinfo=None)`` — a
  bare strip that discards the UTC offset without adjusting the wall-clock time.  A
  Tokyo +09:00 news item at ``2022-06-15T01:00+09:00`` (= ``2022-06-14T16:00 UTC``)
  would be misidentified as *future* relative to ``t_now = 2022-06-15T00:00 UTC``,
  and a US -08:00 item at ``2022-06-14T20:00-08:00`` (= ``2022-06-15T04:00 UTC``)
  would be misidentified as *past*.
- **Decision:** If the input datetime is timezone-aware, call
  ``.astimezone(dt.timezone.utc).replace(tzinfo=None)`` to convert to UTC first.
  Naive inputs are treated as UTC already (yfinance/Finnhub return UTC-naive).
  ISO-8601 strings have no offset — always treated as UTC-naive.
  Internal convention: **all timestamps are UTC-naive throughout the codebase**.
- **Consequence:** Timezone-aware inputs (e.g. from external APIs that include offsets)
  are correctly compared.  Tests ``TestTimezoneHandling`` verify both directions of
  boundary crossing.  No breaking change for existing tests (they use UTC-naive inputs).

### 2026-05 — Baselines use pandas, not vectorbt
- **Context:** PLAN.md listed vectorbt; CLAUDE.md says "baselines in vectorbt".
  vectorbt is an optional extra (heavy scipy/numba dependency).  The baseline logic
  is simple daily arithmetic — no vectorbt-specific features are needed.
- **Decision:** Implement baselines in plain pandas + our existing ``eval/financial.py``
  metrics.  vectorbt retained as optional extra for potential future use (e.g.
  portfolio-level optimisation).  The change is noted here; the thesis methodology
  chapter describes the signal rules, not the library.
- **Consequence:** No extra install step; baselines are readable without vectorbt docs.
  If a reviewer asks "why not vectorbt?", the answer is: simplicity and readability
  outweigh vectorbt's performance benefits on daily data of this scale.

### 2026-05 — Stub backbone for smoke-testing agent loop (NEVER for thesis results)
- **Context:** The agent loop needs an OpenAI API key.  During development/CI we want
  to verify the plumbing (clock → perceive → reason → execute → NAV) without burning
  budget.
- **Decision:** A ``StubBackbone`` class that uses a simple deterministic rule
  (trend-following: buy if price > SMA20, sell if < SMA20·0.98) is enabled via
  ``use_stub=True``.  The engine raises loudly if ``OPENAI_API_KEY`` is missing and
  ``use_stub=False``.  Stub output is clearly labelled in all reports.
- **Consequence:** Any run with ``backbone=stub`` in the results JSON must NEVER be
  cited in the thesis.  Only runs with ``backbone=gpt-4.1`` (or similar) are valid.

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
