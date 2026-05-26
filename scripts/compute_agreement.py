#!/usr/bin/env python
"""Compute judge-vs-human inter-rater agreement from annotated CSV.

Usage::

    python scripts/compute_agreement.py results/<run_id>/annotation_sample.csv

The human must have filled in ``human_predicted_action`` and
``human_agrees_with_verdict`` in the CSV exported by ``sample_decisions.py``.

Outputs
-------
- % agreement on predicted action (judge vs human)
- Cohen's kappa on predicted action
- % of cases where human agrees with the faithfulness verdict
- Breakdown by category (unfaithful, constrained, ungrounded, random)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False)


def _cohen_kappa(a: list[str], b: list[str], labels: list[str]) -> float:
    """Simple Cohen's kappa for two annotation lists with fixed label set."""
    n = len(a)
    if n == 0:
        return 0.0
    # Observed agreement
    p_o = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    # Expected agreement
    p_e = sum(
        (a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels
    )
    if p_e >= 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 4)


@app.command()
def main(csv_path: Path = typer.Argument(..., help="Annotated annotation_sample.csv")) -> None:
    """Compute inter-rater reliability: LLM judge vs human annotator."""
    if not csv_path.exists():
        typer.echo(f"[ERROR] Not found: {csv_path}", err=True)
        raise typer.Exit(1)

    rows: list[dict[str, Any]] = []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Filter to rows with human annotations
    annotated = [
        r for r in rows
        if r.get("human_predicted_action", "").strip().lower() in ("buy", "sell", "hold")
    ]

    if not annotated:
        typer.echo(
            "[WARNING] No annotated rows found.\n"
            "Fill in 'human_predicted_action' (buy/sell/hold) in the CSV first."
        )
        raise typer.Exit(0)

    judge_actions = [r["judge_predicted_action"].lower() for r in annotated]
    human_actions = [r["human_predicted_action"].strip().lower() for r in annotated]
    labels = ["buy", "sell", "hold"]

    pct_agree_action = sum(j == h for j, h in zip(judge_actions, human_actions, strict=True)) / len(annotated)
    kappa = _cohen_kappa(judge_actions, human_actions, labels)

    verdict_agree = [
        r for r in annotated
        if r.get("human_agrees_with_verdict", "").strip().upper() == "Y"
    ]
    pct_verdict_agree = len(verdict_agree) / len(annotated)

    typer.echo()
    typer.echo("=" * 60)
    typer.echo(" JUDGE-vs-HUMAN INTER-RATER AGREEMENT")
    typer.echo("=" * 60)
    typer.echo(f"  Annotated rows        : {len(annotated)} / {len(rows)}")
    typer.echo()
    typer.echo("  Predicted action:")
    typer.echo(f"    % agreement         : {pct_agree_action * 100:.1f}%")
    typer.echo(f"    Cohen's kappa       : {kappa:.3f}")
    typer.echo()
    typer.echo("  Faithfulness verdict:")
    typer.echo(f"    Human agrees        : {pct_verdict_agree * 100:.1f}%")
    typer.echo()

    # Per-category breakdown
    categories = sorted({r.get("category", "?") for r in annotated})
    typer.echo("  Per-category action agreement:")
    for cat in categories:
        cat_rows = [r for r in annotated if r.get("category") == cat]
        if not cat_rows:
            continue
        j = [r["judge_predicted_action"].lower() for r in cat_rows]
        h = [r["human_predicted_action"].strip().lower() for r in cat_rows]
        pct = sum(a == b for a, b in zip(j, h, strict=True)) / len(cat_rows)
        typer.echo(f"    {cat:<15}: {pct * 100:.1f}%  (n={len(cat_rows)})")
    typer.echo()


if __name__ == "__main__":
    app()
