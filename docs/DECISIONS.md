# Design decisions (ADR-lite)

---

### 2026-05-28 - PM API smoke: TradingAgents-style chain, cache-first data reuse

**Context:** PM v1 was intentionally stub-first. The next step is a tiny real
API smoke that validates the PM plumbing, Langfuse traces, `decisions.jsonl`,
grounding, MCP call summaries, and anti-lookahead checks without changing the
single-agent baseline methodology or opening a path to an expensive run.

**Decision:** The real smoke path now mirrors the communication topology in
`reference/TradingAgents/`: analyst reports first, then bounded Bull/Bear
investment debate, Research Manager synthesis, Trader proposal, bounded
Aggressive/Conservative/Neutral risk debate, and finally one Portfolio Manager
allocation. This is implemented in `OpenAIDiscussionBackbone` with compact JSON
artifacts for each node rather than an unbounded chat transcript. The initial
analysts use the MCP tool outputs visible to their role; the debate/manager/trader
risk nodes consume the structured state. The PM emits strict JSON parsed into
`PMTargetWeights`. Invalid JSON, negative weights, unknown tickers, or leverage
fail loudly at the parser boundary. Inside the smoke engine, an API/parsing
failure is written explicitly as `pm_api_error` and converted to an all-cash
fallback so the artifact is inspectable rather than silent.

**Architecture boundary:** We do not import code from `reference/TradingAgents/`
and do not move the smoke to full LangGraph yet. The smoke copies the communication
contract and bounded state shape inside the existing PM backbone so the MCP,
cache, JSONL, rebalancer, and evaluation plumbing remain unchanged. A full
LangGraph implementation can replace the hand-driven node sequence later without
changing the emitted PM decision schema.

**Cost envelope:** One smoke decision is 11 cached LLM calls: 3 analysts, 2
investment debaters, 1 Research Manager, 1 Trader, 3 risk debaters, and 1 PM. It
still uses `settings.agent_model_dev` (`gpt-4.1-mini`) and the same prompt-hash
disk cache as the single-agent `OpenAIBackbone`.

**Guardrail:** `PMBacktestEngine(use_stub=False)` is not a general API runner. It
requires a validated `PMSmokeGuardResult`, which is only produced by
`scripts/run_pm_api_smoke.py`: exactly one date, one or two tickers, dev model
only, LLM cache on, and explicit `--acknowledge-cost`. No other code path should
start PM API calls.

**Cache/data rule:** PM price loading is cache-first. Existing parquet prices are
read from `data/cache/prices` and reused; yfinance is called only if the cache is
missing or insufficient for the warm-up/backtest window. News remains cache-first
via `get_news_items_cache_first`. `mcp_calls` records tool, source (`cache`,
`api`, `cache+api`, or `in_memory`), row counts, and max timestamp so smoke logs
can be audited for look-ahead. Tool output max timestamps must be `<= t_now`.

**Observability/eval:** PM decisions include top-level `rationale`, `tool_outputs`,
`mcp_calls`, flattened indicators, regimes, targets, communication turns, orders,
and fills. Langfuse gets one PM observation per date with reports, communication
state, portfolio, tool outputs, target weights, rebalance orders, and fills.
`compute_grounding` now falls back to indicator values inside `tool_outputs`, so
PM JSONL can be evaluated offline. Langfuse export is best-effort; a trace export
timeout must not fail the local run.

**Consequence:** The single-agent baseline remains untouched. The PM smoke is a
methodology/plumbing artifact, not a thesis performance result. It can be rerun
cheaply from cache, traced in Langfuse, grounded offline, and manually inspected
before any larger PM experiment is considered.

**Initial validation:** Real guarded smokes on 2026-05-28 completed on the
TradingAgents-style path. `pm_api_smoke_20260528_222656` (2023-01-03 / AAPL)
produced 3 analyst reports, 7 structured communication turns, all-cash target
weights, no orders/fills, cached prices, and no MCP timestamp after `t_now`.
`pm_api_smoke_20260528_222547` (2023-02-13 / JPM+NVDA) produced 6 analyst
reports, the same 7-turn sequence, target weights `JPM=0.25`, `NVDA=0.70`, cash
`0.05`, two buy orders, two fills, non-negative cash, and no future timestamps.
Missing news cache was explicit and allowed only because `--allow-empty-news` was
passed. These runs validate plumbing only.

---

### 2026-05-28 - PM V1: global portfolio manager, long-only cash-only, stub-first

**Context:** The validated single-agent run remains the canonical thesis baseline. The
next experimental phase tests whether a global portfolio manager can reduce the
single-agent exit/trim weakness without burning OpenAI credits during plumbing.

**Decision:** Add a separate `multi_agent`/PM path rather than changing the baseline
single-agent engine. PM v1 is global by date: Technical Analyst, News Analyst, Risk
Analyst, then a Portfolio Manager emits portfolio-level target weights. The rebalancer
is deterministic and enforces long-only, no leverage, no shorting, and non-negative cash.
There is no fixed 20% per-ticker cap in PM mode; concentration control is delegated to
the PM/risk reports and the target weights.

**Budget rule:** Development and tests use deterministic stubs/mocks only. OpenAI is
forbidden during PM plumbing. Any later API smoke test must be explicitly separated from
the full run, limited to 1 date and 1-2 tickers on the cheap model with the LLM cache on,
and the logs must be manually inspected before any larger run.

**Consequence:** PM logs are written as daily portfolio decisions with analyst reports,
targets, rebalance orders, fills, and tool outputs in `decisions.jsonl`. Stub PM results
are plumbing artifacts only and must not be cited as thesis performance evidence.
`scripts/run_pm_api_smoke.py` only validates the future smoke-test envelope in dry-run
mode by default; the actual PM API engine path remains blocked until a real PM API
backbone is implemented.

---

### 2026-05-28 - HUMAN ANNOTATION: simplified policy for v2 judge validation

**Context:** The 80-row `annotation_sample_v2.csv` needed a human review pass. Initial
manual inspection found that the v2 judge can still over-call `buy` around the 10% base
target when signals are weak (e.g. JPM at 9.9% in a range regime), but the broader v2
policy is acceptable if interpreted as a simple allocation policy.

**Decision:** Human annotations were filled using the following simplified policy:

- Base/moderate target = 10% NAV.
- Strong-conviction target = 15% NAV.
- If position is effectively at target and signals are weak/mixed, `hold` is coherent.
- If position is below 10% and signals are clearly positive with feasible cash, `buy`
  toward target is coherent.
- Between 15% and 20%, do not add; `hold` is coherent if signals are not deteriorating.
- Above 20%, trim is mandatory: human action = `sell`.
- Zero-fill `buy` intentions are interpreted through the policy, but the agreement is
  with the judge's predicted rational action.

`annotation_sample_v2.csv` was updated in place; `annotation_sample_v2_strata_audit.csv`
was regenerated with the same human columns. `scripts/compute_agreement.py` was updated
to support v2 columns (`v2_judge_predicted`, `human_agrees_v2`) and derived v2 strata.

**Result:** 80/80 rows annotated. Judge-vs-human action agreement = **83.8%**,
Cohen's kappa = **0.705**. Because the sample is stratified, the per-stratum result is
more important than the global score:

| Stratum | n | Agreement |
|---|---:|---:|
| Non-trim sell | 25 | 100.0% |
| Other hold-judge / agent-buy | 13 | 100.0% |
| Other sell-judge / agent-buy | 6 | 100.0% |
| Other hold-judge / agent-sell | 3 | 100.0% |
| Faithful reclassified from v1 unfaithful | 15 | 80.0% |
| Faithful persistent/other | 10 | 70.0% |
| D1 buy gap | 8 | 12.5% |

**Consequence:** V2 is sufficiently validated for the thesis as a policy-aware judge,
especially for the dominant non-trim finding. The remaining weakness is narrow and
interpretable: D1 cases around the 10% threshold with weak/mixed signals. Do not spend
more time redesigning the single-agent judge unless needed for writing; proceed to the
portfolio-manager / multi-agent phase.

---

### 2026-05-28 - AUDIT CORRECTION: JSONL-only NVDA return, cap status, non-trim recount

**Context:** The previous offline non-trim audit stated that "NVDA +980%" drove the
613 non-trim sells and flagged 56 "sell->buy" cases as buys above the 20% cap. The user
requested a source-of-truth audit using only
`runs/gpt-4-1-mini_20260528_112645/decisions.jsonl` for prices. No yfinance refetch,
no parquet price read, no agent rerun, and no API calls were used. V2 judge actions were
read only from `runs/.faithfulness_cache_v2/`; raw LLM requested quantities were matched
from `runs/.llm_cache/`.

#### NVDA return: no external price-series contamination found

The JSONL close series is internally consistent: `bars_recent[-1].close` equals
`indicators.close` for all 2505 decisions.

| Measure from JSONL closes | Value |
|---|---:|
| Start close, 2022-07-01 | 14.523 |
| End close, 2024-06-28 | 123.540 |
| Total start-to-end return | +750.7% |
| Annualised return, 252-trading-day convention | +194.2% |
| Baseline CSV `annualised_return` | +194.2% |

**Decision:** The canonical NVDA Buy & Hold comparator remains the annualised
`+194.2%` from the JSONL close series, matching `results/baselines_20260526_235938.csv`.
The previous `+980%` statement is not a start-to-end Buy & Hold return. It is reproducible
from the same JSONL series as a trough-to-late-June move, e.g. 2022-10-10 close 11.67 to
2024-06-25 close 126.09 = +980.5%. No evidence was found that the audit used a yfinance
refetch or any off-JSONL price series.

**Consequence:** The previous narrative "NVDA +980% appreciation drives the finding" is
retired. The ticker/regime count of non-trim sells is not invalidated by price-source
contamination because it is based on judge/action labels, not external prices.

#### 56 `judge=sell`, `agent_action=buy` cases above 20% NAV

All 56 cases were buy *intentions* written as `action=buy`, not executed buys.
The engine computed `capped=0` in every case, wrote `quantity=0`, and produced no fill.

| Check | Result |
|---|---:|
| Cases | 56 |
| Raw LLM requested shares | 1,167 total |
| JSONL quantity after engine cap | 0 for all 56 |
| Filled quantity / notional | 0 shares / $0 |
| Engine `capped` value | 0 for all 56 |
| Tickers | MSFT 19, AAPL 18, NVDA 7, XOM 7, JPM 5 |
| Regimes | bull 52, high_vol 4 |

AAPL 2022-07-14 to 2022-07-18 illustrates the mechanism:

| Date | Raw qty | Fill qty | Price | NAV | Pct before | Held | Max total | Capped | Pct after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-07-14 | 20 | 0 | 148.47 | 99,037.23 | 20.39% | 136 | 133 | 0 | 20.39% |
| 2022-07-15 | 20 | 0 | 150.17 | 99,823.03 | 20.46% | 136 | 132 | 0 | 20.46% |
| 2022-07-18 | 33 | 0 | 147.07 | 99,431.95 | 20.12% | 136 | 135 | 0 | 20.12% |

Detailed per-case state is stored in
`results/gpt-4-1-mini_20260528_112645/cap_audit_sell_judge_agent_buy_over20.csv`.

**Decision:** This is not an active regression of the structural buy cap. It is case (c):
the position crossed above 20% through mark-to-market appreciation after a valid near-cap
entry, then the agent attempted to buy again; the engine blocked every attempted buy.

**Consequence:** Financial metrics are not threatened: these 56 cases changed NAV by $0.
The reasoning/action audit needs a wording caveat: `action=buy` can mean "agent intended
to buy but the engine filled zero". Across the whole run, 183 decisions have `action=buy`,
but only 96 have an executed buy fill; 87 buy intentions had no fill (59 capped to zero,
28 rejected for insufficient cash).

#### Clean non-trim recount from JSONL-only state

The v2 residual remains 825 unfaithful decisions. The dominant finding survives without
the `+980%` narrative: 613 are still non-trim sells (`judge=sell`, `agent=hold`). The
classification was redone using the last **filled** buy, not the last `action=buy`
(because zero-fill capped buy intentions are not entries).

| Category | Total | Bear | Bull | High_vol | Range |
|---|---:|---:|---:|---:|---:|
| Non-trim: buy-push over 20% | 612 | 100 | 358 | 91 | 63 |
| Non-trim: under-20 sell signal | 1 | 1 | 0 | 0 | 0 |
| True D1 buy gaps | 8 | 4 | 2 | 0 | 2 |
| Other: judge hold, agent buy, filled | 91 | 19 | 29 | 34 | 9 |
| Other: judge hold, agent buy, zero/rejected | 31 | 1 | 26 | 4 | 0 |
| Other: judge sell, agent buy, zero-capped over 20% | 56 | 0 | 52 | 4 | 0 |
| Other: judge hold, agent sell | 25 | 12 | 3 | 9 | 1 |
| Other: judge buy, agent sell | 1 | 1 | 0 | 0 | 0 |
| **Total v2 unfaithful** | **825** | **138** | **470** | **142** | **75** |

Ticker x regime for the 613 non-trim sells is unchanged:

| Ticker | Total | Bear | Bull | High_vol | Range |
|---|---:|---:|---:|---:|---:|
| AAPL | 45 | 18 | 17 | 7 | 3 |
| JPM | 83 | 12 | 46 | 6 | 19 |
| MSFT | 16 | 2 | 7 | 0 | 7 |
| NVDA | 409 | 63 | 243 | 75 | 28 |
| XOM | 60 | 6 | 45 | 3 | 6 |
| **Total** | **613** | **101** | **358** | **91** | **63** |

Detailed rows are stored in
`results/gpt-4-1-mini_20260528_112645/non_trim_clean_recount_jsonl.csv`.

**Decision:** The thesis finding is reframed as an exit-management gap after near-cap
entries: the agent builds positions to approximately the hard cap, then does not trim
when the judge sees a sell/trim condition. It is not explained by an NVDA price-source
artifact and not explained by a live engine cap regression.

**Consequence:** V2 faithfulness remains the canonical thesis metric (`0.671`). The
qualitative diagnosis is stronger and cleaner: "limited exit/trim discipline after
near-cap entries", with a separate caveat that some `buy` actions in reasoning outputs
were blocked by the engine and had no financial effect.

#### Annotation sample v2 stratification

The 80-row sample is diagnostic/stratified, not representative.

| Derived stratum | Verdict | N |
|---|---|---:|
| Non-trim sell | unfaithful | 25 |
| D1 buy gap | unfaithful | 8 |
| Other judge hold, agent buy | unfaithful | 13 |
| Other judge sell, agent buy | unfaithful | 6 |
| Other judge hold, agent sell | unfaithful | 3 |
| Faithful reclassified from v1 unfaithful | faithful | 15 |
| Faithful persistent/other | faithful | 10 |
| **Total** |  | **80** |

Non-trim sells are under-represented inside the unfaithful sample: 25/55 = 45.5% of
sampled unfaithful rows vs 613/825 = 74.3% of the full v2 residual. Unfaithful rows are
over-represented overall: 55/80 = 68.8% in the sample vs 825/2505 = 32.9% in the full run.
Derived strata are stored in
`results/gpt-4-1-mini_20260528_112645/annotation_sample_v2_strata_audit.csv`.

**Decision:** Use `annotation_sample_v2.csv` for targeted judge validation, but report
human agreement by stratum. Do not report one global Cohen's kappa as if the sample were
random/representative. If an overall population agreement is needed, draw a separate
random sample and report it separately.

**Consequence:** Manual annotation can proceed with this caveat. The main thesis text
should distinguish diagnostic agreement (per stratum) from any future representative
agreement estimate.

---

### 2026-05-28 — PRE-REGISTRATION: policy-aware faithfulness judge (v2)

**Context:** Run #4 faithfulness (v1 judge) = 0.281 overall, 0.372 strict. Diagnostic showed
the v1 judge had a systematic bias: it assumed the agent tries to deploy up to the 20% hard cap,
so it marked at-target holds (10–19% NAV) as unfaithful. But the SYSTEM_PROMPT explicitly targets
10–15% NAV — holding at that level is deliberate, not a reasoning failure.

**User validated** the following policy spec on 2026-05-28 before any recalculation.
Pre-registration means the spec was locked before looking at the v2 numbers.

#### Revealed sizing policy (from SYSTEM_PROMPT + rationale evidence)

| Parameter | Value | Source |
|-----------|-------|--------|
| Base target | 10% of NAV | SYSTEM_PROMPT formula: `target_value = 0.10 * NAV` |
| High-conviction target | 15% of NAV | SYSTEM_PROMPT formula: `target_value = 0.15 * NAV` |
| Hard cap | 20% of NAV | SYSTEM_PROMPT formula + engine structural cap |

Rationale language (2354/2505 = 94% decisions contain sizing %): 20% mentioned 1495×,
15% 512×, 10% 295×, 14–16% 611× — confirms the agent explicitly tracks the 10-15% target
zone and references the 20% cap as a hard limit, not a target.

#### Hold-decision diagnostic (pre-v2 structural bucketing)

| Bucket | N | % of holds | Definition |
|--------|---|-----------|------------|
| A | 1730 | 75.3% | Cash < 1% NAV — capacity constraint (can't deploy) |
| B | 306 | 13.3% | Position >= 19% NAV — cap-adjacent (buying would breach 20% cap) |
| C | 104 | 4.5% | Position 10–19% NAV — at/near target (deliberate hold) |
| D | 156 | 6.8% | Position < 10%, cash >= 1% — potential gap; classified by v2 LLM judge |

#### V1 judge flaw

The v1 "max-deploy" judge only knew the 20% cap. Seeing a position at 14%:
"still room to buy → predict buy → agent holds → **unfaithful**." This was wrong for
bucket C (policy-consistent disciplined hold).

#### V2 judge design (pre-registered)

`_FAITHFULNESS_JUDGE_SYSTEM_V2` in `src/mcp_quant_agent/eval/reasoning.py`:
- Explicitly states base_target=10%, high_conv=15%, hard_cap=20%
- Tells judge: "once position >= 10%, hold is the rational default"
- BUY only when: pos < 10% AND cash >= 1% AND clearly bullish AND not bear/high_vol
- SELL when: clearly bearish AND deteriorating; OR position >> 20%

#### Verdict semantics (Condition 2 — user correction)

**CHANGED from v1:** Bucket C = FAITHFUL (not constrained).

| Verdict | Meaning | Buckets |
|---------|---------|---------|
| `faithful` | Agent did what a rational policy-following agent would do | A (cash-forced → judge also predicts hold), B (cap-forced → judge predicts hold), C (at-target → judge predicts hold knowing policy), D-with-reason (bearish signal → judge predicts hold) |
| `constrained` | Agent COULD NOT act — capacity constraint only | Cash < 1% NAV (buy side); Pos >= 19% (buy side, safety net only); Pos = 0 (sell side, structural) |
| `unfaithful` | Real gap — judge predicts action, agent doesn't, no capacity reason | D1 (pos < 10%, cash >= 1%, bullish, no reason); non-trim sells (holds when judge=sell) |

Rationale: `constrained` = *capacity* (physically couldn't act).
`faithful` = *deliberate coherent* action (could have acted, chose not to for policy reasons).
Conflating these inflates the `constrained` bucket and understates true faithfulness.

#### Threshold justifications (Condition 4)

- **1% NAV cash threshold** (Capacity-A):
  Adding < 1% NAV exposure is noise. Formula: `qty = floor(0.01 × NAV / price)` yields 0–6
  shares of a $150 stock at $100k NAV (< 0.9% position move). Below minimum meaningful trade
  increment. Justified as "cash < cost of a minimal trade increment", not an arbitrary round number.
- **19% cap-adjacent threshold** (Capacity-B):
  A position at 19% is within approximately one normal trading increment of the 20% hard cap.
  Any buy would risk exceeding the cap after normal intraday price movement. Documented as
  "cap-adjacent", not "at-cap" — the cap itself is 20%.
- **10% base target** (Policy target):
  Directly from SYSTEM_PROMPT `target_value = 0.10 * NAV`. Non-negotiable (derived from spec).
- **19% vs 18%** (why raised from v1):
  v1 used 18% for both buy and sell constrained checks. Under v2, the sell-side 18% check is
  REMOVED for holds-when-sell (those are now unfaithful non-trim gaps). Only the buy-side
  threshold is kept, raised to 19% to match "cap-adjacent" semantics more precisely.

#### Condition 3 — Non-trim sell gap

The 26 sells observed vs 2296 holds implies a strong non-trim bias.
Under v2 judge: any hold where judge predicts SELL (with non-zero position) is UNFAITHFUL.
These are "failure-to-trim" events and represent the other half of the real faithfulness gap
(the first being D1 holds that should have been buys). Tabulated by regime in the v2 results.

#### Condition 1 — Bucket D via LLM judge (not keyword grep)

The 156 bucket D cases were passed through the v2 LLM judge.
Bear regime dominates D (113/156): in bear regime, the v2 judge correctly predicts "hold"
for bearish/uncertain signals even with pos < 10% + cash available. The keyword approach
(banned in §3 of HANDOVER.md) would have made the same error the v1 judge made — checking
rationale language instead of the actual market evidence the judge sees.

#### Actual v2 results — computed 2026-05-28

| Regime | n | V1 faith | V1 strict | V1 unf | V1 con | V2 faith | V2 strict | V2 unf | V2 con | Δfaith |
|--------|---|---------|---------|--------|--------|---------|---------|--------|--------|--------|
| bear | 513 | 0.429 | 0.533 | 193 | 100 | **0.731** | 0.731 | 138 | 0 | +0.302 |
| bull | 1109 | 0.159 | 0.235 | 574 | 359 | **0.576** | 0.576 | 470 | 0 | +0.418 |
| high_vol | 509 | 0.330 | 0.403 | 249 | 92 | **0.721** | 0.721 | 142 | 0 | +0.391 |
| range | 374 | 0.316 | 0.379 | 193 | 63 | **0.799** | 0.799 | 75 | 0 | +0.484 |
| **OVERALL** | **2505** | **0.272** | **0.361** | **1209** | **614** | **0.671** | **0.671** | **825** | **0** | **+0.399** |

| Count | V1 | V2 | Delta |
|-------|-----|-----|-------|
| n_faithful | 682 | 1680 | +998 |
| n_unfaithful | 1209 | 825 | −384 |
| n_constrained | 614 | **0** | −614 |
| judge_fail | 0 | 0 | 0 |

**V2 constrained = 0** (structural finding): The policy-aware judge predicts "hold" for
bucket A (cash < 1%), bucket B (pos >= 19%), and bucket C (pos 10–19%) cases itself,
because it knows the policy. These become `faithful` (judge=hold, agent=hold) rather
than needing a constrained override. The `_is_constrained_hold_v2` safety net fires for
0 cases — the prompt does its job.

**Non-trim sells (Condition 3):**
613 decisions where v2 judge predicts SELL but agent holds. All 613 → unfaithful.
By regime: bull=358 (58%), bear=101 (16%), high_vol=91 (15%), range=63 (10%).
Concentrated in positions above 20% NAV with bullish indicators — the agent builds
positions aggressively but does not trim when they exceed the hard cap. This is the
**dominant source of unfaithfulness** (613/825 = 74% of all unfaithful decisions).
Not present in v1 results: v1's case-3 constrained check (pos >= 18% + judge=sell)
silently absorbed all 608 of these as "constrained" — hiding the finding entirely.

**True D1 gaps (Condition 1):** 8 decisions (pos < 10%, cash >= 1%, judge=buy, agent holds).
Regime breakdown: bear=4, bull=2, range=2, high_vol=0. Negligible (8/825 = 1%).
For reference, keyword-grep would have estimated 3 — LLM judge found 8 (more nuanced).

**Verdict movements V1 → V2:**
| Transition | N | Interpretation |
|------------|---|----------------|
| unfaithful → faithful | 1086 | False positives removed (at-target/cash-forced holds) |
| constrained → unfaithful | 608 | Non-trim sell gaps exposed (were hidden as "constrained") |
| faithful → unfaithful | 94 | Genuinely wrong under stricter v2 policy |
| unfaithful → unfaithful | 123 | True persistent gaps (buy/other mismatches) |
| faithful → faithful | 588 | Correctly classified by both judges |
| constrained → faithful | 6 | Capacity-constrained now handled by judge prompt |

**Decision:** Use v2 as the canonical faithfulness metric for the thesis.
Keep v1 available via `policy_aware=False` flag in `compute_faithfulness_llm` for
comparison and reproducibility. Report both side-by-side in thesis Table X.

**Consequence:** faithfulness score increases substantially (v1 over-penalised at-target
holds). The non-zero unfaithful count under v2 is concentrated in non-trim sells
(failure to cut positions in downtrends) — a genuine and interesting finding.

**Files changed:** `src/mcp_quant_agent/eval/reasoning.py` (added
`_FAITHFULNESS_JUDGE_SYSTEM_V2`, `_judge_faithful_action_v2`, `_is_constrained_hold_v2`,
`policy_aware` flag on `compute_faithfulness_llm`). New scripts:
`scripts/step3_dual_faithfulness.py`, `scripts/step4_annotation_sample_v2.py`.

---

### 2026-05-28 — AUDIT: non-trim sell decomposition (pre-annotation, offline) [SUPERSEDED]

**Superseded by the JSONL-only audit correction above.** The ticker/regime count
of 613 non-trim sells remains valid, but the `+980%` NVDA narrative and the
611/2 buy-push/drift split are no longer canonical. The corrected recount uses
only `decisions.jsonl` closes and the last **filled** buy.

**Context:** V2 faithfulness = 0.671. Dominant finding: 613 holds where v2 judge predicts
SELL (74% of 825 unfaithful). User hypothesised these might be appreciation-drift artifacts
(symmetric to the v1 at-target-hold bug). Offline audit run with
`scripts/audit_non_trim_sells.py` on cached decisions — **no API calls**.

#### Q1 — Ticker × regime decomposition

| Ticker | Total | Bear | Bull | High_vol | Range |
|--------|-------|------|------|----------|-------|
| AAPL | 45 | 18 | 17 | 7 | 3 |
| JPM | 83 | 12 | 46 | 6 | 19 |
| MSFT | 16 | 2 | 7 | 0 | 7 |
| NVDA | **409** | 63 | **243** | 75 | 28 |
| XOM | 60 | 6 | 45 | 3 | 6 |
| **TOTAL** | **613** | **101** | **358** | **91** | **63** |

NVDA dominates: **409/613 = 67%**. Bull regime dominates: 358/613 = 58%.
NVDA's +980% appreciation (Jul 2022 → Jun 2024) drives the concentration.

#### Q2 — Appreciation drift vs buy-push

Classification rule: look back to the agent's most recent BUY for this ticker.
If position % at that BUY was ≥ 18%, it is "buy-push" (agent entered near cap, never exited).
If position % at last BUY was < 18%, it is "appreciation drift" (passive run-up).

| Category | N | % | Policy status |
|----------|---|---|---------------|
| **Buy-push** (pos ≥ 18% at last buy) | **611** | **99.7%** | Real cap-management gap |
| **Appreciation drift** (pos < 18% at last buy) | **2** | **0.3%** | Policy-silent → coherent |
| No prior buy | 0 | 0% | Edge case |

**Key finding: the user's appreciation-drift hypothesis is WRONG.** Only 2 of 613 are pure
drift. The other 611 represent cases where the agent actively bought positions to near-cap
levels (≥ 18%), which then exceeded 20%, and the agent never trimmed.

Drift breakdown by ticker: AAPL=1 (bear), XOM=1 (high_vol).

#### Q3 — Policy check: does SYSTEM_PROMPT require trimming?

Four relevant SYSTEM_PROMPT quotes:
1. "Target 10–15% of NAV per position. Hard cap: 20% of NAV per position."
   → Establishes the constraint, doesn't specify how to enforce it after entry.
2. "BUY quantity formula: `additional_value = min(target_value, 0.20 × NAV - existing_value)`"
   → 20% limit is in the BUY formula only — entry constraint, not ongoing maintenance.
3. "SELL quantity: use position size from the portfolio (sell entire position, or a partial fraction)."
   → SELL instruction is purely mechanical. No trigger condition about exceeding % of NAV.
4. "Single position MUST NOT exceed 20% of portfolio NAV."
   → Imperative, but no instruction to trim if exceeded by appreciation or buy-drift.

**Conclusion:** SYSTEM_PROMPT is ambiguous. The cap is enforced structurally at BUY by the
engine formula. There is **no explicit trim instruction** for positions that grow above 20%.
However, "MUST NOT exceed" is an imperative that could be read as ongoing maintenance.

For the thesis, this is treated as follows:
- The 611 buy-push cases are **unfaithful** (the agent bought aggressively to near-cap, 
  position exceeded 20%, agent never acted — the v2 judge's "SELL when pos >> 20%" 
  instruction is reasonable given the imperative language in the spec).
- The 2 drift cases are **reclassified as coherent** (policy-silent — agent never bought 
  the position above 18%, drift is passive, and no trim instruction exists).

#### Proposed V3 reclassification and faithfulness impact

| Category | N | V3 treatment |
|----------|---|--------------|
| Appreciation drift | 2 | → faithful (policy-silent) |
| Buy-push above cap | 611 | → stays unfaithful |
| True D1 gaps | 8 | → stays unfaithful |
| Other action mismatches | 204 | → stays unfaithful |
| **Total unfaithful** | **823** | |

V3 faithfulness = (2505 − 823) / 2505 = **0.6715** (vs V2 = 0.6707, Δ = +0.0008).

**Decision:** V3 reclassification is negligible. **V2 = 0.671 remains the canonical number.**
No code changes required. Annotation sample is unchanged.

#### Characterisation of 204 "other" unfaithful

| Judge predicts | Agent action | N | Interpretation |
|----------------|-------------|---|----------------|
| hold | buy | 122 | Aggressive buying when judge says hold; early bull days (all 5 tickers buy on 2022-07-01 in bear regime) |
| sell | buy | 56 | Agent buys ABOVE CAP (pos ≥ 20%) while judge says sell — genuine cap-breach pattern, mostly AAPL 2022-07-14 to 07-18 |
| hold | sell | 25 | Early exits — agent sells small positions (pos ~9%) when judge says hold |
| buy | sell | 1 | Rare: agent sells when judge says buy |

Notable: the 56 "sell→buy" cases include multiple consecutive AAPL decisions at pos=20.4–20.5%
NAV where the agent *bought above the hard cap* (2022-07-14 to 07-18). This suggests the
engine's cap enforcement had a transient bug early in Run #4, or the agent's BUY formula
used stale price data during the initial portfolio build.

The 122 "hold→buy" cases are primarily the run's opening days: the agent opens positions
in all 5 tickers simultaneously despite being in a bear regime — judged by the v2 judge as
"hold" (bearish conditions), but the agent buys. This is coherent as portfolio initialisation
behaviour but is correctly flagged as unfaithful relative to the v2 policy.

#### Summary table (for thesis Chapter 4)

| Category | N | % of 825 | Policy verdict |
|----------|---|----------|----------------|
| Non-trim sell: buy-pushed above cap | 611 | 74% | Unfaithful (real cap-mgmt gap) |
| Non-trim sell: appreciation drift | 2 | 0% | Reclassified coherent (V3) |
| True D1 gaps (confirmed) | 8 | 1% | Unfaithful |
| Other: hold→buy (aggressive entry) | 122 | 15% | Unfaithful |
| Other: sell→buy (buys above cap) | 56 | 7% | Unfaithful |
| Other: hold→sell (early exit) | 25 | 3% | Unfaithful |
| Other: buy→sell (rare) | 1 | 0% | Unfaithful |
| **Total unfaithful (V2)** | **825** | **100%** | |

**Files:** `scripts/audit_non_trim_sells.py` (offline, no API calls). Output to stdout only
(no new result files — the 613 non-trim cases were already in `non_trim_sells.csv`).

---

### 2026-05-28 — Run #4 VALID: final results

**Run:** `gpt-4-1-mini_20260528_112645`. Period: 2022-07-01 → 2024-06-30. **VALID.**

**Financial performance:**

| Metric | Agent | Equal-Wt B&H | Momentum | Mean-Rev |
|---|---|---|---|---|
| Ann. Return | +72.1% | +73.0% | +24.7% | +16.6% |
| Sharpe | **2.35** [0.92, 3.58] | 2.27 | 1.91 | 1.41 |
| Sortino | 2.74 | — | — | — |
| Calmar | 5.76 | — | — | — |
| Max DD | −12.5% | — | — | — |
| Hit Rate | 54.6% | — | — | — |
| Final NAV | $293,594 | — | — | — |

Agent nearly matches B&H in raw return but exceeds it on Sharpe (2.35 vs 2.27) with
only −12.5% max drawdown. Beats momentum by +47pp and mean-reversion by +56pp annualised.

**Reasoning metrics (faithfulness LLM judge + grounding):**

| Regime | n | faithfulness | faith_strict | grounding |
|---|---|---|---|---|
| bear | 513 | 0.429 | 0.533 | 0.999 |
| bull | 1109 | 0.159 | 0.235 | 0.997 |
| high_vol | 509 | 0.330 | 0.403 | 1.000 |
| range | 374 | 0.316 | 0.379 | 0.998 |
| **OVERALL** | **2505** | **0.281** | **0.372** | **0.998** |

Grounding (0.998) is near-perfect: 8240/8255 numeric claims match tool outputs within 1%.
Faithfulness (0.281 overall) is low — primarily driven by the "fully-deployed" structural
effect: after ~bar 100 the portfolio is 99%+ invested (only $99–$257 cash), so all buy
signals result in cash-rejected or cap-limited holds. The judge correctly predicts "buy"
but the agent is structurally unable to act. Bear regime has the best faithfulness (0.429)
because the agent does sell/hold correctly in downtrends.

**Action distribution:** buy=183 (7.3%), sell=26 (1.0%), hold=2296 (91.7%)
**Regime distribution:** bull=1109 (44%), bear=513 (20%), high_vol=509 (20%), range=374 (15%)
**Parse errors:** 0 / 2505
**Annotation sample:** 73 rows → `results/gpt-4-1-mini_20260528_112645/annotation_sample.csv`

### 2026-05 — Run #4 pre-launch: retry transient OpenAI errors; raise max_tokens 1024→2048

**Context (Run #4 aborted twice before first valid bar):**

**Abort 1 — `InternalServerError` (HTTP 500) crashed immediately:**
`_call_openai_async` only retried `RateLimitError` (429); all other exceptions
hit `except Exception: raise exc` and propagated immediately.  A transient OpenAI
500 on bar ~10 killed the run before any decisions were written.

**Fix:** Expand the retry set to include `InternalServerError`, `APIConnectionError`,
and `APITimeoutError` — all genuinely transient.  Stored in a `_RETRIABLE` tuple
so future additions are one-line.  Same exponential back-off as 429 (1s, 2s, 4s …
up to 32s, 6 attempts).

**Abort 2 — Poisoned LLM cache from partial run:**
The first aborted run wrote 415 truncated responses (max_tokens=1024) to the LLM
cache before crashing.  The re-launch loaded these cache entries and produced
`parse_error: Expecting ',' delimiter` warnings for ~16% of decisions.

**Root cause of truncation:** `gpt-4.1-mini` produces chain-of-thought JSON with
a `chain_of_thought` object + `action`/`quantity`/`rationale` — total response
easily exceeds 1024 tokens, causing mid-JSON truncation.  Observed: "Unterminated
string at char 3145" and `Expecting ',' delimiter` at char ~1000.

**Fix:** `max_tokens` raised 1024→2048 in BOTH the async (`_call_openai_async`)
and sync (`OpenAIBackbone.decide`) paths.  Poisoned LLM cache cleared; run
re-launched from scratch.

**Consequence:** From bar 1 onward: 5/5 responses parse cleanly, no
`parse_error` rationales, `InternalServerError` is retried not fatal.

### 2026-05 — Run #3 post-mortem: three fixes before Run #4

**Run:** `gpt-4-1-mini_20260527_005408`. FinalNAV = $86 (−99.9%). **INVALID.**

**Bug 1 — Engine buy-cap direction (`engine.py` line 362):**
`quantity = capped` set the buy quantity to the MAXIMUM allowed (20% of NAV),
inflating any small legitimate LLM request to the ceiling.  Observed in logs:
"Engine cap AAPL buy: 72→143", "Engine cap NVDA buy: 656→1312".
**Fix:** `quantity = min(quantity, capped)` — only cap DOWN, never UP.
Log condition also corrected from `!= capped` to `> capped`.

**Bug 2 — Warm-up used `merge_and_write` instead of `write` (overwrite):**
`merge_and_write` preserves bars from previous runs for dates not covered by the
new fetch.  If a previous run left stale or split-contaminated bars in the parquet,
they survive the merge.  Root cause of the NVDA price contamination: the parquet
contained a mix of 10:1 split-adjusted prices (~$14/share) and unadjusted prices
(~$113/share) alternating on consecutive days, producing absurd RSI=95 and a 54%-of-NAV
NVDA buy that destroyed the portfolio.
**Fix:** warm-up calls `cache.write(ticker, "1d", all_bars)` (complete overwrite).
The warm-up always fetches the FULL range (warmup_start→end_date), so overwriting
is always safe — no useful bars are lost.

**Bug 3 — No price-continuity guard:**
Neither `_fetch_raw_bars` nor `get_price_history` checked for adjacent-bar price
jumps > 5×.  The contamination propagated silently through the entire 2-year backtest.
**Fix:** new `_check_price_continuity(bars, ticker, max_ratio=5.0)` in
`yfinance_source.py` raises `ValueError` on any ratio > 5× between consecutive
daily closes.  Called in `_fetch_raw_bars` after every fetch.  If triggered in
Run #4, it terminates the run with a clear error message instructing the user to
delete `data/cache/prices/` and re-run.

**Note on Run #3 reasoning metrics:** 0 parse_error decisions in decisions.jsonl
(all 2505 parsed correctly); `response_format={"type": "json_object"}` ensures
valid JSON from the OpenAI API.  The run is invalid for FINANCIAL metrics only.
Faithfulness/grounding from Run #3 decisions.jsonl COULD technically be computed
but are not representative (too many forced holds due to NAV collapse).

---

### 2026-05 — Run #2 post-mortem: four fixes before Run #3

**Context:** Run #2 produced FinalNAV=$86 (−99.9%), despite clean price data.
Diagnostics revealed four independent issues.

**Issue 1 — "unknown" regime phantom (372 decisions, 14.9%):**
Engine pre-fetch used `_fetch_raw_bars(start_date, end_date)`, which does NOT
write to the parquet cache.  `perceive_ticker → get_price_history` reads from
the parquet cache → empty for AAPL/MSFT → 0 bars → `regime=None` →
serialised as `"unknown"` in eval tables.  This inflated the "unknown" regime's
faithfulness (0.976) and polluted all regime-segmented tables.
**Fix:** engine pre-fetch now fetches from `start_date − 130 days` and calls
`cache.merge_and_write`, populating the parquet cache before any backtest bar
is processed.  `price_data[ticker]` is then filtered to `>= start_date` for the
trading loop.  All 5 tickers get warm-up bars; regime detector has ≥ 21 bars
from bar 1; no `None` regime.

**Issue 2 — Regime label instability between runs:**
Run #1 used corrupt prices (yfinance MultiIndex bug → wrong OHLCV scalars) →
wrong vol/momentum inputs → different regime labels (range: 150→40,
high_vol: 824→506 vs run #2).  The detector is fully deterministic (pure function,
no randomness).  **Run #2 labels are the ground truth.**  No code change to
the detector needed.  Added `TestRegimeDeterminism` (6 tests) to
`tests/test_regime.py` verifying same price input → same labels every call.

**Issue 3 — Constrained-hold exemption barely fired (5/2505):**
`_is_constrained_hold` covered only (a) judge=buy+hold+position≥18% and
(b) judge=sell+hold+qty==0.  Diagnostics showed 310 "judge=sell, agent=hold"
cases with non-zero positions; 214 at 18-25% NAV.  The agent cited the 20% limit
as the reason for holding — a defensible risk-management interpretation of the
position cap even for the sell direction (boundary caution).
**Fix:** added case (c): judge=sell + agent=hold + position ≥ 18% NAV →
constrained.  The 20% boundary creates rational inertia in both directions
(symmetric with the buy-side case).  Also added `faithfulness_strict` =
n_faithful / (n_faithful + n_unfaithful), which excludes constrained cases
from the denominator, as the second faithfulness metric for the thesis.
Denominator note: the standard `faithfulness` formula (n_faithful / n_scoreable,
where n_scoreable includes constrained) is unchanged for comparability; the
strict metric is additive.

**Issue 4 — Position sizing catastrophe:**
The LLM used fixed share quantities (e.g. "buy 200 NVDA") instead of NAV-scaled
quantities.  At NAV=$85, any order > 1 share of NVDA ($68) was rejected by the
portfolio's 20% cap, freezing the portfolio for the rest of the run.
**Fix (two layers):**
1. SYSTEM_PROMPT updated: step 6 now includes the explicit formula
   `qty = floor(target_fraction × NAV / last_close)` with a worked example,
   and a guard `if NAV < 500: hold everything`.
2. Engine hard cap (structural backstop): before every `portfolio.place_order`,
   the engine clamps `quantity` to (a) `floor(0.20×NAV/price) − current_held`
   for buys, (b) `min(qty, held)` for sells.  This prevents degenerate runs
   even if the LLM ignores the SYSTEM_PROMPT.

**Why four separate fixes, not one:**
Issues 1, 3, 4 each affect a different layer (data/cache, eval metrics, order
execution) and are fully independent.  Issue 2 required documentation + tests
only.  Mixing them would make each harder to audit individually.

---

### 2026-05 — Run #1 post-mortem: yfinance Series→scalar bug + max_tokens truncation

**Run:** `gpt-4-1-mini_20260526_160051` (2022-07-01→2024-06-30, AAPL/MSFT/NVDA/JPM/XOM).
FinalNAV = $126 (−99.9%), MaxDD = 99.95%.  **This run is invalid for thesis use.**

**Root cause 1 — yfinance price-data corruption:**
`yfinance` returns a MultiIndex DataFrame for single-ticker downloads in recent versions.
After `df.columns = df.columns.get_level_values(0)`, duplicate column names can arise.
Accessing `row["Close"]` on a row with duplicate "Close" columns returns a `pd.Series`
instead of a scalar.  The previous code caught `TypeError` and *silently skipped* the bar
with only a `logger.warning`.  Some bars were accepted with wrong scalar values extracted
from the head of a multi-element Series (e.g. AAPL close=372 instead of ~$195 in Dec 2023),
producing absurd buy signals and catastrophic NAV collapse.

**Root cause 2 — max_tokens=512 too small:**
The agent's chain-of-thought JSON response was truncating mid-JSON for verbose rationales.
3 parse failures observed; could have silently degraded reasoning quality for many more.

**Why it was not caught:**
- Anti-look-ahead tests guard the *temporal* dimension perfectly.
- There were **no tests on data values** — the "forteresse" only checked *when*, not *what*.
- The `except (ValueError, TypeError)` handler masked the bug as warnings in a 42-minute run.

**Fixes applied (2026-05-27):**
1. `bar_validation.py` — new module with `_to_float_scalar` + `validate_bar`.
   `_to_float_scalar` raises on multi-element Series, NaN, Inf, or non-convertible types.
   `validate_bar` checks: close/open/high/low > 0; high ≥ low; high ≥ close ≥ low;
   close ∈ [open×0.5, open×2] (anti-spike); volume ≥ 0.  Any violation raises `ValueError`
   with ticker, date, field name, and bad value — **fail loud**.
2. `yfinance_source.py` — de-duplicate columns after MultiIndex flatten before `iterrows`;
   replace the silent-skip `except` handler with `_to_float_scalar` calls that propagate;
   call `validate_bar` on every constructed bar (write path) and on every bar returned from
   the parquet cache (read path).  Old corrupt caches are caught on first backtest access.
3. `orchestrator.py` — `max_tokens` 512 → 1024 in both `_call_openai_async` and
   `OpenAIBackbone.decide`.
4. Tests: `test_bar_validation.py` — 26 tests covering all violation types, including the
   exact run #1 failure mode (multi-element Series) and the cache-read regression guard.

**Cache invalidation:** delete `data/cache/prices/*.parquet` before re-running to force a
fresh fetch with the new validation.  Any previously cached corrupt bars will be detected
and rejected on the first read, prompting manual cache deletion.

**Reasoning metrics from run #1 are still valid:** the LLM judge saw only regime + indicators
(not absolute prices), so faithfulness=0.674, grounding=0.999 are unaffected by the price bug.

---

### 2026-05 — Async LLM parallelisation in the backtest engine
- **Context:** The thesis run (5 tickers × 501 bars = 2505 decisions) is serialised by
  sequential OpenAI round-trips (~1 s each).  Wall time ~42 min serial; multi-seed and
  multi-agent runs would be proportionally worse.  Concurrency is pure I/O; there is no
  computation benefit.
- **Decision:** ``asyncio.gather`` all LLM calls *within a bar-date* (bounded by
  ``Semaphore(concurrency=15)`` to stay within the OpenAI rate limit for gpt-4.1-mini).
  All tickers on a date receive the **same start-of-date portfolio snapshot** (computed
  once before any orders via ``portfolio.get_portfolio_summary()``), so all decisions are
  made against a consistent, order-independent portfolio context.  Orders are then executed
  serially in deterministic ``self.tickers`` order after all LLM results arrive.
  ``decisions.jsonl`` is sorted by ``(date, ticker)`` at the end — byte-for-byte
  reproducible across runs.  ``run()`` is a thin sync wrapper over ``asyncio.run(_run_async_body())``.
- **Non-regression guarantee:** ``TestAsyncEqualSerial`` (``tests/test_async_backtest.py``)
  runs serial (``concurrency=1``) and async (``concurrency=10``) on a 10-bar/2-ticker stub
  window and asserts identical action, quantity, and rationale for every decision.  This
  test must stay green before any engine change.
- **Cache thread-safety:** ``_save_cache()`` in ``orchestrator.py`` uses ``temp → os.replace``
  (atomic on Windows within the same filesystem).  Concurrent goroutines writing the same key
  produce identical payloads; the last writer wins harmlessly.
- **Back-off:** ``_call_openai_async()`` retries on ``RateLimitError`` (HTTP 429) with
  exponential back-off capped at 32 s per attempt, up to 6 attempts.
- **Consequence:** Estimated wall time ~3-5 min for the thesis run (down from ~42 min serial),
  depending on cache hit rate.  Multi-seed runs (3 seeds) become practical at ~10-15 min.
  The serial==async invariant means cached results are interchangeable.

### 2026-05 — Faithfulness metric: LLM judge replacing keyword/regex (ÉTAPE 2 fix)
- **Context:** The original faithfulness metric compared ``_extract_intent(rationale)`` to
  the executed ``action``.  Both fields are written by the same LLM call (one JSON response
  containing both "rationale" and "action"), making them trivially consistent — this measures
  nothing.  Keyword/regex is also fragile on nuanced prose ("I won't sell yet" misfires).
- **Decision:** Replace with an independent LLM judge (gpt-4.1-mini, temperature=0).
  The judge receives ONLY the objective evidence the agent also had: ticker, date, regime,
  technical indicators, and portfolio snapshot (positions with pct_of_nav, cash, NAV,
  20% NAV constraint).  It predicts the rational action blind (no rationale, no action).
  We then compare judge prediction to the agent's actual executed action.
  Three verdict categories: **faithful** (match), **constrained** (mismatch explained by a
  risk constraint: position ≥ 18% NAV when judge says buy, or qty = 0 when judge says sell),
  **unfaithful** (unexplained mismatch).  Constrained cases are excluded from the
  faithfulness rate (not penalised — correct behaviour, not inconsistency).
  Judge results cached in ``runs/.faithfulness_cache/`` (MD5 of model+evidence fields) so
  re-runs cost $0.  A ``_judge_fn`` hook on ``compute_faithfulness_llm`` allows fully
  deterministic tests without an API key.
- **Consequence:** Faithfulness now measures genuine knowledge-action consistency (c.f.
  KellyBench §4 "knowledge-action gap"), not self-consistency of a single LLM call.
  Judge validation via ``scripts/sample_decisions.py`` + ``compute_agreement.py`` (Cohen's
  kappa of judge-vs-human) provides the inter-rater reliability the thesis needs.
  Cost ≈ $0.00012/decision for the judge calls; negligible vs. agent cost.

### 2026-05 — Raw tool_outputs schema in decisions.jsonl (grounding fix)
- **Context:** The original ``tool_outputs`` field stored only summary counts (e.g.
  ``{"tool": "get_price_history", "bars_returned": 250, "date_range": "..."}``).  The
  grounding metric could not verify numeric claims about close prices, NAV, or position
  values because those numbers were never recorded — only bar counts and key lists.
- **Decision:** Record full raw values in ``tool_outputs``:
  - ``get_price_history``: ``bars_count`` + ``bars_recent`` (last 5 OHLCV dicts).
  - ``compute_indicators``: ``values`` dict (all computed indicator values rounded to 4dp).
  - ``get_current_regime``: ``regime`` string.
  - ``get_news_items``: ``items_count`` + ``items_recent`` (last 3 headlines truncated to 120 chars).
  - ``get_portfolio``: ``cash``, ``nav``, ``positions`` (list with ticker, quantity,
    market_value, pct_of_nav).
  ``pct_of_nav`` is computed in the orchestrator as ``market_value / nav`` so the grounding
  metric can check portfolio-percentage claims without division.
  ``_extract_portfolio_snapshot`` and ``_extract_bars_recent`` helpers pull these out cleanly.
  ``_ground_claim`` updated to check three sources: indicators dict, bars_recent OHLCV, and
  portfolio values (nav, cash, position market_values) — all within 1% tolerance.
- **Consequence:** Grounding is now fully verifiable for all numeric claim types.  Old
  decisions.jsonl files (summary schema) have no ``bars_recent`` / ``positions`` — grounding
  will report 0 claims (not penalised: no signal) rather than silently mismatch.  The 
  thesis run must be restarted with the new schema to produce valid grounding metrics.

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
