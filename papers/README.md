# Papers — annotated state of the art

> Drop the PDFs in this folder. This README is the fast path: read it before reopening any
> PDF. For each paper: what it is, **why it matters for this thesis**, and a BibTeX stub.
>
> ⚠️ The figures/claims below are working notes (orders of magnitude and angle), not
> citable quotes. **Verify exact numbers in the PDF before citing them in the thesis.**

---

## Methodology backbone (evaluation — the thesis's graded core)

### StockBench (Chen et al., 2025)
- **What:** A contamination-free benchmark for LLM agents making sequential buy/sell/hold
  decisions in realistic multi-month stock environments; daily price/fundamentals/news.
- **Why it matters:** Closest setup to ours and the template for our methodology. Borrow its
  "contamination-free" framing and its return+risk metric set. Equity-centric → directly
  comparable to our US-equities primary track.
- `arXiv:2510.02209`

### LiveTradeBench (Yu et al., 2025)
- **What:** Real-time trading benchmark, 21 LLMs over a live window, fixed lookback to avoid
  leakage; stocks + Polymarket.
- **Why it matters:** Two quotable findings. (1) General LLM leaderboard scores barely
  correlate with trading returns (ρ≈0) — strong justification that trading needs its own
  evaluation. (2) Big cross-regime generalisation gaps — supports our regime-segmented axis.
- `arXiv:2511.xxxxx` (verify)

### KellyBench our methodological guardrail
- **What:** Long-horizon sequential decision-making in EPL betting markets; every frontier
  model loses money on average; introduces a 52-point "sophistication" rubric.
- **Why it matters:** Catalogues the exact failure modes we must avoid/measure — the
  *knowledge–action gap* (Kelly coded but never invoked), non-stationarity (promoted teams =
  regime shift), label leakage (`CalibratedClassifierCV` on full train set), premature
  termination, single-seed unreliability. Our faithfulness/grounding metrics and our
  multi-seed + bootstrap protocol are direct responses to this paper. **Cite heavily in the
  methodology and "pitfalls avoided" sections.**
- `arXiv:2604.27865`

---

## Architecture references (multi-agent "fund")

### TradingAgents (Xiao et al., 2024/2025)
- **What:** Multi-agent firm: fundamental/sentiment/technical analysts, Bull/Bear researcher
  debate, trader, risk team. Reports improved cumulative return / Sharpe / maxDD vs baselines.
- **Why it matters:** The reference architecture. We reuse the *role decomposition* and the
  *Bull/Bear debate* pattern (the latter is where LangGraph's cyclic graphs pay off in week 2).
  Our differentiator vs them = MCP tool layer + regime-aware CoT evaluation.
- `arXiv:2412.20138` · repo: github.com/TauricResearch/TradingAgents

### Agent Market Arena / AMA (Qian et al., 2025)
- **What:** Live benchmark comparing InvestorAgent / TradeAgent / HedgeFundAgent /
  DeepFundAgent across several backbones.
- **Why it matters:** Key result — **agent architecture drives behaviour more than the model
  backbone**. Justifies comparing {single-agent vs multi-agent} and not over-investing in
  backbone sweeps. Also a clean source of standard financial metrics.
- `arXiv:2510.11695`

### ATLAS (2025)
- **What:** Adaptive trading via dynamic prompt optimisation + multi-agent coordination;
  gains concentrated in **volatile regimes**.
- **Why it matters:** Direct support for the regime axis and for the hypothesis that
  reasoning quality/strategy must adapt to volatility. Useful related-work contrast.
- `arXiv:2510.15949`

---

## (Optional) deeper background
- **HedgeAgents** (WWW '25, `arXiv:2502.13165`) — central fund manager + asset-class experts,
  coordination via "conferences". Inspiration if we build the full fund hierarchy.
- **FinCon** — hierarchical manager–analyst comms with risk control.

---

## How to cite in the thesis
Keep a single `.bib` (e.g. `docs/references.bib`). Pull canonical BibTeX from the arXiv
abstract pages rather than hand-typing. Cross-check every numeric claim against the PDF.
