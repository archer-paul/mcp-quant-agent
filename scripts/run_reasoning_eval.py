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
    from mcp_quant_agent.eval.reasoning import compute_all_reasoning_metrics

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
        # Compute faithfulness + grounding only, skip LLM judge
        from mcp_quant_agent.eval.reasoning import (
            _segment_by_regime,
            compute_faithfulness,
            compute_grounding,
        )

        regimes = _segment_by_regime(decisions)
        faith_all = compute_faithfulness(decisions)
        ground_all = compute_grounding(decisions)

        header = ["regime", "n_decisions", "faithfulness", "grounding"]
        rows: list[list[Any]] = []
        for regime, rd in sorted(regimes.items()):
            faith = compute_faithfulness(rd)
            grnd = compute_grounding(rd)
            rows.append([regime, len(rd), f"{faith['faithfulness']:.3f}", f"{grnd['grounding']:.3f}"])
        rows.append([
            "OVERALL", len(decisions),
            f"{faith_all['faithfulness']:.3f}",
            f"{ground_all['grounding']:.3f}",
        ])

        typer.echo("=" * 60)
        typer.echo(" REASONING METRICS (faithfulness + grounding)")
        typer.echo("=" * 60)
        _print_table(header, rows)
        typer.echo()
        typer.echo("  Faithfulness detail:")
        typer.echo(f"    n_scoreable  : {faith_all['n_scoreable']}")
        typer.echo(f"    n_faithful   : {faith_all['n_faithful']}")
        typer.echo(f"    n_unfaithful : {faith_all['n_unfaithful']}")
        typer.echo(f"    n_no_signal  : {faith_all['n_no_signal']}")
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
    typer.echo("  Faithfulness detail:")
    typer.echo(f"    n_scoreable={fd['n_scoreable']}, n_faithful={fd['n_faithful']}, "
               f"n_unfaithful={fd['n_unfaithful']}, n_no_signal={fd['n_no_signal']}")
    if fd.get("examples_unfaithful"):
        typer.echo("    Unfaithful examples (first 3):")
        for ex in fd["examples_unfaithful"][:3]:
            typer.echo(f"      {ex['date']} {ex['ticker']} [{ex['regime']}]: "
                       f"action={ex['action']}, intent={ex['intent']}")
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
    try:
        from mcp_quant_agent.observability.langfuse_setup import (
            _is_langfuse_configured,
            get_langfuse_client,
        )

        _is_langfuse_configured()
        client = get_langfuse_client()
        if client is not None:
            # Push as dataset-level scores using the trace for each decision
            pushed = 0
            for d in decisions:
                if d.get("action") == "error":
                    continue
                # Push faithfulness intent signal (1.0/0.0 or None)
                faith_score: float | None = None
                from mcp_quant_agent.eval.reasoning import _extract_intent

                intent = _extract_intent(str(d.get("rationale", "")))
                if intent is not None:
                    faith_score = 1.0 if intent == d.get("action") else 0.0

                if faith_score is not None:
                    try:
                        client.score_current_trace(
                            name="faithfulness",
                            value=faith_score,
                            comment=f"intent={intent}, action={d.get('action')}",
                        )
                        pushed += 1
                    except Exception:
                        pass
            typer.echo(f"  Pushed {pushed} Langfuse faithfulness scores.")
    except Exception as exc:
        typer.echo(f"  Langfuse push skipped: {exc}")


if __name__ == "__main__":
    app()
