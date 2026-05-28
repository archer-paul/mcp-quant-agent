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
- Breakdown by category / derived v2 stratum
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


def _pick_column(rows: list[dict[str, Any]], candidates: list[str]) -> str | None:
    """Return the first present column from ``candidates``."""
    if not rows:
        return None
    available = set(rows[0].keys())
    for col in candidates:
        if col in available:
            return col
    return None


def _derived_stratum(row: dict[str, Any]) -> str:
    """Return the v2 diagnostic stratum when explicit strata are absent."""
    if row.get("derived_stratum"):
        return str(row["derived_stratum"])

    actual = str(row.get("action", "")).lower()
    judge = str(row.get("v2_judge_predicted", row.get("judge_predicted_action", ""))).lower()
    verdict = str(row.get("verdict_new_judge", row.get("verdict", ""))).lower()
    old_verdict = str(row.get("verdict_old_judge", "")).lower()

    if verdict == "faithful":
        if old_verdict == "unfaithful":
            return "faithful_reclassified_from_unfaithful"
        return "faithful_persistent_or_other"
    if judge == "sell" and actual == "hold":
        return "non_trim_sell"
    if judge == "buy" and actual == "hold" and row.get("bucket") == "D":
        return "D1_buy_gap"
    if judge == "hold" and actual == "buy":
        return "other_hold_judge_agent_buy"
    if judge == "sell" and actual == "buy":
        return "other_sell_judge_agent_buy"
    if judge == "hold" and actual == "sell":
        return "other_hold_judge_agent_sell"
    if judge == "buy" and actual == "sell":
        return "other_buy_judge_agent_sell"
    return str(row.get("category", "unknown"))


@app.command()
def main(
    csv_path: Path = typer.Argument(..., help="Annotated annotation_sample.csv"),
    judge_col: str | None = typer.Option(
        None,
        help="Judge action column. Auto-detects v2_judge_predicted or judge_predicted_action.",
    ),
    human_col: str = typer.Option(
        "human_predicted_action",
        help="Human action annotation column.",
    ),
    agree_col: str | None = typer.Option(
        None,
        help="Human agreement column. Auto-detects human_agrees_v2 or human_agrees_with_verdict.",
    ),
) -> None:
    """Compute inter-rater reliability: LLM judge vs human annotator."""
    if not csv_path.exists():
        typer.echo(f"[ERROR] Not found: {csv_path}", err=True)
        raise typer.Exit(1)

    rows: list[dict[str, Any]] = []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    judge_col = judge_col or _pick_column(
        rows, ["v2_judge_predicted", "judge_predicted_action", "v1_judge_predicted"]
    )
    agree_col = agree_col or _pick_column(
        rows, ["human_agrees_v2", "human_agrees_with_verdict"]
    )
    if judge_col is None:
        typer.echo(
            "[ERROR] Could not find a judge action column. Expected one of: "
            "v2_judge_predicted, judge_predicted_action, v1_judge_predicted.",
            err=True,
        )
        raise typer.Exit(1)
    if human_col not in rows[0]:
        typer.echo(f"[ERROR] Missing human action column: {human_col}", err=True)
        raise typer.Exit(1)

    # Filter to rows with human annotations
    annotated = [
        r for r in rows
        if r.get(human_col, "").strip().lower() in ("buy", "sell", "hold")
    ]

    if not annotated:
        typer.echo(
            "[WARNING] No annotated rows found.\n"
            "Fill in 'human_predicted_action' (buy/sell/hold) in the CSV first."
        )
        raise typer.Exit(0)

    judge_actions = [r[judge_col].strip().lower() for r in annotated]
    human_actions = [r[human_col].strip().lower() for r in annotated]
    labels = ["buy", "sell", "hold"]

    pct_agree_action = sum(j == h for j, h in zip(judge_actions, human_actions, strict=True)) / len(annotated)
    kappa = _cohen_kappa(judge_actions, human_actions, labels)

    pct_verdict_agree: float | None = None
    if agree_col is not None:
        verdict_agree = [
            r for r in annotated
            if r.get(agree_col, "").strip().upper() == "Y"
        ]
        pct_verdict_agree = len(verdict_agree) / len(annotated)

    typer.echo()
    typer.echo("=" * 60)
    typer.echo(" JUDGE-vs-HUMAN INTER-RATER AGREEMENT")
    typer.echo("=" * 60)
    typer.echo(f"  Annotated rows        : {len(annotated)} / {len(rows)}")
    typer.echo()
    typer.echo("  Predicted action:")
    typer.echo(f"    judge column        : {judge_col}")
    typer.echo(f"    % agreement         : {pct_agree_action * 100:.1f}%")
    typer.echo(f"    Cohen's kappa       : {kappa:.3f}")
    typer.echo()
    if pct_verdict_agree is not None:
        typer.echo("  Faithfulness verdict:")
        typer.echo(f"    agreement column    : {agree_col}")
        typer.echo(f"    Human agrees        : {pct_verdict_agree * 100:.1f}%")
        typer.echo()

    # Per-stratum breakdown. For v2 samples, ``category`` is just faithful/unfaithful;
    # derived strata preserve the diagnostic sampling buckets.
    strata = sorted({_derived_stratum(r) for r in annotated})
    typer.echo("  Per-stratum action agreement:")
    for stratum in strata:
        cat_rows = [r for r in annotated if _derived_stratum(r) == stratum]
        if not cat_rows:
            continue
        j = [r[judge_col].strip().lower() for r in cat_rows]
        h = [r[human_col].strip().lower() for r in cat_rows]
        pct = sum(a == b for a, b in zip(j, h, strict=True)) / len(cat_rows)
        typer.echo(f"    {stratum:<42}: {pct * 100:.1f}%  (n={len(cat_rows)})")
    typer.echo()


if __name__ == "__main__":
    app()
