#!/usr/bin/env python
"""Apply reasoning metrics (faithfulness, grounding, sophistication) to a run.

Usage::

    python scripts/run_reasoning_eval.py runs/<run_id>/decisions.jsonl
    python scripts/run_reasoning_eval.py runs/<run_id>/decisions.jsonl --no-sophistication

Output
------
Prints the regime-segmented table to stdout.
Writes results/<run_id>/reasoning_metrics.csv.
Pushes scores to Langfuse (best-effort).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _print_table(header: list[str], rows: list[list[Any]]) -> None:
    """Print a text table to stdout."""
    # Compute column widths
    col_widths = [len(str(h)) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    typer.echo(fmt.format(*header))
    typer.echo(sep)
    for row in rows:
        typer.echo(fmt.format(*[str(c) for c in row]))


@app.command()
def main(
    jsonl_path: Path = typer.Argument(..., help="Path to decisions.jsonl"),
    judge_model: str = typer.Option("gpt-4.1-mini", help="LLM judge model."),
    no_sophistication: bool = typer.Option(
        False, "--no-sophistication", help="Skip LLM judge (faster, cheaper)."
    ),
    max_soph_per_regime: int = typer.Option(
        50, help="Max decisions sent to LLM judge per regime."
    ),
    seed: int = typer.Option(42, help="Random seed for judge sampling."),
) -> None:
    """Compute faithfulness, grounding, and sophistication from decisions.jsonl."""
    import os

    from mcp_quant_agent.eval.reasoning import compute_all_reasoning_metrics

    # Inject API keys from pydantic-settings into os.environ (needed by OpenAI SDK)
    if not os.environ.get("OPENAI_API_KEY"):
        from mcp_quant_agent.config import settings

        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    from mcp_quant_agent.observability.langfuse_setup import _is_langfuse_configured

    _is_langfuse_configured()
    # Silence Langfuse "no active span" warnings — this script runs standalone,
    # outside a live trace context, so span-level operations will warn noisily.
    logging.getLogger("langfuse").setLevel(logging.ERROR)

    if not jsonl_path.exists():
        typer.echo(f"[ERROR] File not found: {jsonl_path}", err=True)
        raise typer.Exit(1)

    # Load decisions
    decisions: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))

    typer.echo(f"Loaded {len(decisions)} decisions from {jsonl_path}")
    typer.echo()

    # Compute metrics
    if no_sophistication:
        # Faithfulness (LLM judge, cached) + grounding; skip sophistication judge
        from mcp_quant_agent.eval.reasoning import (
            _segment_by_regime,
            compute_faithfulness_llm,
            compute_grounding,
        )

        regimes = _segment_by_regime(decisions)
        faith_all = compute_faithfulness_llm(decisions, judge_model=judge_model, seed=seed)
        ground_all = compute_grounding(decisions)

        header = ["regime", "n_decisions", "faithfulness", "faith_strict",
                  "n_unfaithful", "n_constrained", "grounding"]
        rows: list[list[Any]] = []
        for regime, rd in sorted(regimes.items()):
            faith = compute_faithfulness_llm(rd, judge_model=judge_model, seed=seed)
            grnd = compute_grounding(rd)
            rows.append([
                regime, len(rd),
                f"{faith['faithfulness']:.3f}",
                f"{faith['faithfulness_strict']:.3f}",
                faith["n_unfaithful"], faith["n_constrained"],
                f"{grnd['grounding']:.3f}",
            ])
        rows.append([
            "OVERALL", len(decisions),
            f"{faith_all['faithfulness']:.3f}",
            f"{faith_all['faithfulness_strict']:.3f}",
            faith_all["n_unfaithful"], faith_all["n_constrained"],
            f"{ground_all['grounding']:.3f}",
        ])

        typer.echo("=" * 70)
        typer.echo(" REASONING METRICS (faithfulness LLM judge + grounding)")
        typer.echo("=" * 70)
        _print_table(header, rows)
        typer.echo()
        typer.echo("  Faithfulness detail (LLM judge):")
        typer.echo(f"    n_scoreable       : {faith_all['n_scoreable']}")
        typer.echo(f"    n_faithful        : {faith_all['n_faithful']}")
        typer.echo(f"    n_unfaithful      : {faith_all['n_unfaithful']}")
        typer.echo(f"    n_constrained     : {faith_all['n_constrained']}")
        typer.echo(f"    n_judge_fail      : {faith_all['n_judge_failed']}")
        typer.echo(f"    faithfulness      : {faith_all['faithfulness']:.4f}  (faithful / scoreable)")
        typer.echo(f"    faithfulness_strict: {faith_all['faithfulness_strict']:.4f}  (faithful / (faithful+unfaithful))")
        typer.echo()
        typer.echo("  Grounding detail:")
        typer.echo(f"    n_claims     : {ground_all['n_claims_total']}")
        typer.echo(f"    n_grounded   : {ground_all['n_grounded']}")
        typer.echo()

        # Save CSV
        run_dir = jsonl_path.parent
        run_id = run_dir.name
        results_dir = Path("results") / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        import csv
        csv_path = results_dir / "reasoning_metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(header)
            writer.writerows(rows)
        typer.echo(f"  Saved to {csv_path}")
        return

    # Full evaluation including sophistication
    report = compute_all_reasoning_metrics(
        decisions,
        judge_model=judge_model,
        max_sophistication_per_regime=max_soph_per_regime,
        seed=seed,
    )

    table = report["summary_table"]
    typer.echo("=" * 90)
    typer.echo(" REASONING METRICS — FAITHFULNESS / GROUNDING / SOPHISTICATION by REGIME")
    typer.echo("=" * 90)
    _print_table(table["header"], table["rows"])
    typer.echo()

    # Detail
    fd = report["faithfulness_detail"]
    gd = report["grounding_detail"]
    sd = report["sophistication_detail"]
    typer.echo("  Faithfulness detail (LLM judge):")
    typer.echo(f"    n_scoreable       : {fd['n_scoreable']}")
    typer.echo(f"    n_faithful        : {fd['n_faithful']}")
    typer.echo(f"    n_unfaithful      : {fd['n_unfaithful']}")
    typer.echo(f"    n_constrained     : {fd['n_constrained']}")
    typer.echo(f"    n_judge_fail      : {fd['n_judge_failed']}")
    typer.echo(f"    faithfulness      : {fd['faithfulness']:.4f}  (faithful / scoreable)")
    typer.echo(f"    faithfulness_strict: {fd.get('faithfulness_strict', 0.0):.4f}  (faithful / (faithful+unfaithful))")
    if fd.get("examples_unfaithful"):
        typer.echo("    Unfaithful examples (first 3):")
        for ex in fd["examples_unfaithful"][:3]:
            typer.echo(f"      {ex['date']} {ex['ticker']} [{ex['regime']}]: "
                       f"action={ex['action']}, judge={ex['judge_action']}")
            typer.echo(f"        '{ex['rationale_snippet'][:120]}'")
    typer.echo()
    typer.echo("  Grounding detail:")
    typer.echo(f"    n_claims={gd['n_claims_total']}, n_grounded={gd['n_grounded']}")
    if gd.get("examples_ungrounded"):
        typer.echo("    Ungrounded examples (first 3):")
        for ex in gd["examples_ungrounded"][:3]:
            typer.echo(f"      {ex['date']} {ex['ticker']}: "
                       f"'{ex['label']}' claimed {ex['value']}")
    typer.echo()
    typer.echo("  Sophistication detail:")
    typer.echo(f"    n_scored={sd['n_scored']}, n_failed={sd['n_failed']}")
    crit = sd.get("criteria_means", {})
    for c, v in crit.items():
        typer.echo(f"    {c:<25}: {v:.3f}")
    typer.echo()

    # Save CSV
    run_dir = jsonl_path.parent
    run_id = run_dir.name
    results_dir = Path("results") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    import csv
    csv_path = results_dir / "reasoning_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(table["header"])
        writer.writerows(table["rows"])
    typer.echo(f"  Saved to {csv_path}")

    # Push to Langfuse (best-effort)
    # Use judge_actions from the faithfulness result (already computed above).
    try:
        from mcp_quant_agent.observability.langfuse_setup import (
            _is_langfuse_configured,
            get_langfuse_client,
        )

        _is_langfuse_configured()
        client = get_langfuse_client()
        if client is not None:
            pushed = 0
            for ja in fd.get("judge_actions", []):
                verdict = ja.get("verdict", "")
                if verdict not in ("faithful", "unfaithful", "constrained"):
                    continue
                faith_score: float = 1.0 if verdict == "faithful" else 0.0
                try:
                    client.score_current_trace(
                        name="faithfulness",
                        value=faith_score,
                        comment=(
                            f"judge={ja.get('judge')}, actual={ja.get('actual')}, "
                            f"verdict={verdict}"
                        ),
                    )
                    pushed += 1
                except Exception:
                    pass
            typer.echo(f"  Pushed {pushed} Langfuse faithfulness scores.")
    except Exception as exc:
        typer.echo(f"  Langfuse push skipped: {exc}")


if __name__ == "__main__":
    app()
