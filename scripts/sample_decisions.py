#!/usr/bin/env python
"""Export a human-annotation sample from a decisions.jsonl file.

Runs the faithfulness LLM judge (cached) and exports ~N decisions per category
to CSV and markdown for manual annotation.  Inter-rater reliability between the
judge and the human annotator is computed by ``compute_agreement.py`` once the
human fills in the annotation columns.

Usage::

    # Export 20 per category, write to results/<run_id>/annotation_sample.*
    python scripts/sample_decisions.py runs/<run_id>/decisions.jsonl

    # More per category, custom output dir
    python scripts/sample_decisions.py runs/<run_id>/decisions.jsonl \\
        --n-per-category 30 --out-dir annotation/

Categories exported
-------------------
- ``unfaithful``  : judge prediction != actual action (unexplained by constraints)
- ``constrained`` : agent held due to a risk constraint (judge said buy/sell)
- ``ungrounded``  : decision with at least one numeric claim not in tool_outputs
- ``random``      : random sample across all decisions (baseline comparison)

Output columns
--------------
date, ticker, regime, action, judge_predicted_action, faithfulness_verdict,
key_indicators (RSI/SMA/momentum), portfolio_pct_of_nav, rationale,
human_predicted_action (blank), human_agrees_with_verdict (blank), notes (blank)

The human fills in the last three columns then runs ``compute_agreement.py``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key_indicators(indicators: dict[str, Any]) -> str:
    """Return a compact string of the key indicator values."""
    keys = ["rsi_14", "sma_20", "momentum_20d", "atr_14", "macd_hist"]
    parts = []
    for k in keys:
        v = indicators.get(k)
        if v is not None:
            parts.append(f"{k}={round(float(v), 2)}")
    return " | ".join(parts) if parts else "(none)"


def _portfolio_pct(decision: dict[str, Any]) -> str:
    """Return the ticker's portfolio exposure as '14.3% of NAV (100 shares)'."""
    from mcp_quant_agent.eval.reasoning import _extract_portfolio_snapshot

    ticker = str(decision.get("ticker", ""))
    snap = _extract_portfolio_snapshot(decision)
    for pos in snap.get("positions", []):
        if pos.get("ticker") == ticker:
            pct = float(pos.get("pct_of_nav", 0.0))
            qty = float(pos.get("quantity", 0.0))
            return f"{pct * 100:.1f}% ({qty:.0f} sh)"
    return "0.0% (no position)"


def _row(
    d: dict[str, Any],
    judge_action: str | None,
    verdict: str,
) -> dict[str, Any]:
    """Build one annotation row from a decision."""
    indicators = d.get("indicators") or {}
    return {
        "date": d.get("date", ""),
        "ticker": d.get("ticker", ""),
        "regime": d.get("regime", ""),
        "action": d.get("action", ""),
        "judge_predicted_action": judge_action or "FAILED",
        "faithfulness_verdict": verdict,
        "key_indicators": _key_indicators(indicators),
        "portfolio_pct_of_nav": _portfolio_pct(d),
        "rationale": str(d.get("rationale", ""))[:500],
        # Human fills these in:
        "human_predicted_action": "",
        "human_agrees_with_verdict": "",
        "notes": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@app.command()
def main(
    jsonl_path: Path = typer.Argument(..., help="Path to decisions.jsonl"),
    judge_model: str = typer.Option("gpt-4.1-mini", help="LLM judge model."),
    n_per_category: int = typer.Option(20, help="Max decisions exported per category."),
    seed: int = typer.Option(42, help="Random seed for sampling."),
    out_dir: Path = typer.Option(
        Path(""),
        help="Output directory (default: results/<run_id>/).",
    ),
) -> None:
    """Export a human-annotation sample from decisions.jsonl."""

    if not os.environ.get("OPENAI_API_KEY"):
        from mcp_quant_agent.config import settings

        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    from mcp_quant_agent.observability.langfuse_setup import _is_langfuse_configured

    _is_langfuse_configured()
    logging.getLogger("langfuse").setLevel(logging.ERROR)

    if not jsonl_path.exists():
        typer.echo(f"[ERROR] Not found: {jsonl_path}", err=True)
        raise typer.Exit(1)

    # Load decisions
    decisions: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))
    typer.echo(f"Loaded {len(decisions)} decisions.")

    # Run faithfulness judge (cached)
    from mcp_quant_agent.eval.reasoning import (
        _extract_bars_recent,
        _extract_numeric_claims,
        _extract_portfolio_snapshot,
        _ground_claim,
        compute_faithfulness_llm,
    )

    typer.echo("Running faithfulness LLM judge (cached)…")
    faith_result = compute_faithfulness_llm(
        decisions, judge_model=judge_model, seed=seed
    )
    judge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for ja in faith_result["judge_actions"]:
        key = (str(ja.get("date", "")), str(ja.get("ticker", "")))
        judge_map[key] = ja

    # Identify ungrounded decisions
    def _has_ungrounded(d: dict[str, Any]) -> bool:
        rationale = str(d.get("rationale", ""))
        indicators = d.get("indicators") or {}
        bars_recent = _extract_bars_recent(d)
        portfolio = _extract_portfolio_snapshot(d)
        claims = _extract_numeric_claims(rationale)
        return any(
            not _ground_claim(label, value, indicators, bars_recent, portfolio)
            for label, value in claims
        )

    # Build per-category pools
    unfaithful_pool: list[dict[str, Any]] = []
    constrained_pool: list[dict[str, Any]] = []
    valid_pool: list[dict[str, Any]] = []

    for d in decisions:
        if d.get("action") == "error":
            continue
        key = (str(d.get("date", "")), str(d.get("ticker", "")))
        ja = judge_map.get(key, {})
        verdict = ja.get("verdict", "judge_failed")
        if verdict == "unfaithful":
            unfaithful_pool.append(d)
        elif verdict == "constrained":
            constrained_pool.append(d)
        valid_pool.append(d)

    ungrounded_pool = [d for d in decisions if _has_ungrounded(d)]
    typer.echo(
        f"  unfaithful={len(unfaithful_pool)}, constrained={len(constrained_pool)}, "
        f"ungrounded={len(ungrounded_pool)}, total_valid={len(valid_pool)}"
    )

    # Sample
    rng = random.Random(seed)
    categories: dict[str, list[dict[str, Any]]] = {
        "unfaithful": unfaithful_pool,
        "constrained": constrained_pool,
        "ungrounded": ungrounded_pool,
        "random": valid_pool,
    }
    sample_rows: list[dict[str, Any]] = []
    for cat_name, pool in categories.items():
        chosen = rng.sample(pool, min(n_per_category, len(pool)))
        for d in chosen:
            key = (str(d.get("date", "")), str(d.get("ticker", "")))
            ja = judge_map.get(key, {})
            row = _row(d, ja.get("judge"), ja.get("verdict", f"{cat_name}_sampled"))
            row["category"] = cat_name
            sample_rows.append(row)

    # De-duplicate (a decision may appear in multiple pools)
    seen_keys: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in sample_rows:
        k = (row["date"], row["ticker"])
        if k not in seen_keys:
            seen_keys.add(k)
            deduped.append(row)
    sample_rows = deduped
    typer.echo(f"  Sample size (deduplicated): {len(sample_rows)}")

    # Output directory
    if str(out_dir) == "":
        run_id = jsonl_path.parent.name
        out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    fieldnames = [
        "category", "date", "ticker", "regime",
        "action", "judge_predicted_action", "faithfulness_verdict",
        "key_indicators", "portfolio_pct_of_nav",
        "rationale",
        "human_predicted_action", "human_agrees_with_verdict", "notes",
    ]
    csv_path = out_dir / "annotation_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)
    typer.echo(f"  CSV  -> {csv_path}")

    # Markdown (human-readable)
    md_path = out_dir / "annotation_sample.md"
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write("# Annotation Sample — Faithfulness Judge Validation\n\n")
        mf.write(
            "Fill in `human_predicted_action` (buy/sell/hold based on indicators only),\n"
            "`human_agrees_with_verdict` (Y/N), and `notes`.\n"
            "Then run `python scripts/compute_agreement.py` on the CSV.\n\n"
        )
        mf.write(f"**Total decisions annotated:** {len(sample_rows)}\n\n")

        for cat_name in ["unfaithful", "constrained", "ungrounded", "random"]:
            cat_rows = [r for r in sample_rows if r.get("category") == cat_name]
            if not cat_rows:
                continue
            mf.write(f"---\n## {cat_name.upper()} ({len(cat_rows)} decisions)\n\n")
            for i, row in enumerate(cat_rows, 1):
                mf.write(
                    f"### {i}. {row['date']} {row['ticker']} "
                    f"[{row['regime']}]\n\n"
                )
                mf.write(f"- **Agent action**: `{row['action']}`\n")
                mf.write(
                    f"- **Judge predicted**: `{row['judge_predicted_action']}`\n"
                )
                mf.write(
                    f"- **Verdict**: `{row['faithfulness_verdict']}`\n"
                )
                mf.write(f"- **Indicators**: {row['key_indicators']}\n")
                mf.write(f"- **Portfolio**: {row['portfolio_pct_of_nav']}\n")
                mf.write(f"- **Rationale**: _{row['rationale']}_\n\n")
                mf.write(
                    "| Field | Value |\n|---|---|\n"
                    "| human_predicted_action | _(fill in: buy / sell / hold)_ |\n"
                    "| human_agrees_with_verdict | _(fill in: Y / N)_ |\n"
                    "| notes | |\n\n"
                )

    typer.echo(f"  MD   -> {md_path}")
    typer.echo()
    typer.echo(
        "Next step: annotate the markdown/CSV, then run:\n"
        f"  python scripts/compute_agreement.py {csv_path}"
    )


if __name__ == "__main__":
    app()
