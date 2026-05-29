#!/usr/bin/env python
"""Reconcile blind human annotations with the sealed judge key."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False)

_ACTIONS = {"buy", "sell", "hold"}

RECONCILIATION_COLS = [
    "decision_id",
    "source_run",
    "date",
    "ticker",
    "regime",
    "sample_stratum",
    "human_intention",
    "judge_predicted_action",
    "actual_agent_action",
    "faithfulness_verdict",
    "judge_human_agree",
    "human_agent_agree",
    "human_note",
]


def _normalise_action(value: Any) -> str:
    action = str(value or "").strip().lower()
    return action if action in _ACTIONS else ""


def _cohen_kappa(pred_a: list[str], pred_b: list[str]) -> float:
    labels = sorted({*pred_a, *pred_b})
    n = len(pred_a)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in zip(pred_a, pred_b, strict=True) if a == b) / n
    expected = sum((pred_a.count(lbl) / n) * (pred_b.count(lbl) / n) for lbl in labels)
    if expected >= 1.0:
        return 1.0
    return round((observed - expected) / (1.0 - expected), 4)


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _agreement(rows: list[dict[str, Any]]) -> tuple[int, float, float]:
    valid = [
        row
        for row in rows
        if row["human_intention"] in _ACTIONS and row["judge_predicted_action"] in _ACTIONS
    ]
    if not valid:
        return 0, 0.0, 0.0
    human = [row["human_intention"] for row in valid]
    judge = [row["judge_predicted_action"] for row in valid]
    agree = sum(1 for h, j in zip(human, judge, strict=True) if h == j) / len(valid)
    return len(valid), round(agree, 4), _cohen_kappa(human, judge)


def _breakdown(
    rows: list[dict[str, Any]],
    key: str,
) -> list[tuple[str, int, float, float]]:
    result: list[tuple[str, int, float, float]] = []
    for value in sorted({str(row.get(key, "")) for row in rows}):
        subset = [row for row in rows if str(row.get(key, "")) == value]
        n_valid, agree, kappa = _agreement(subset)
        if n_valid:
            result.append((value or "(blank)", n_valid, agree, kappa))
    return result


def reconcile_rows(
    annotated_rows: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    key_map = {str(row.get("decision_id", "")): row for row in key_rows}
    merged: list[dict[str, Any]] = []
    missing_or_invalid = 0
    missing_key = 0

    for row in annotated_rows:
        decision_id = str(row.get("decision_id", ""))
        human = _normalise_action(row.get("human_intention"))
        if not human:
            missing_or_invalid += 1
            continue
        key = key_map.get(decision_id)
        if key is None:
            missing_key += 1
            continue
        judge = _normalise_action(key.get("judge_predicted_action"))
        agent = _normalise_action(key.get("actual_agent_action"))
        merged.append(
            {
                "decision_id": decision_id,
                "source_run": key.get("source_run") or row.get("source_run", ""),
                "date": key.get("date") or row.get("t_now", ""),
                "ticker": key.get("ticker") or row.get("ticker", ""),
                "regime": key.get("regime") or row.get("regime", ""),
                "sample_stratum": key.get("sample_stratum", ""),
                "human_intention": human,
                "judge_predicted_action": judge,
                "actual_agent_action": agent,
                "faithfulness_verdict": key.get("faithfulness_verdict", ""),
                "judge_human_agree": str(human == judge).lower(),
                "human_agent_agree": str(human == agent).lower(),
                "human_note": row.get("human_note", ""),
            }
        )

    n_valid, agree, kappa = _agreement(merged)
    disagreements = [
        row
        for row in merged
        if row["judge_predicted_action"] in _ACTIONS
        and row["human_intention"] != row["judge_predicted_action"]
    ]
    bull_rows = [
        row
        for row in merged
        if row["regime"] == "bull" and row["judge_predicted_action"] in _ACTIONS
    ]
    bull_disagreements = [
        row for row in bull_rows if row["human_intention"] != row["judge_predicted_action"]
    ]
    return {
        "merged_rows": merged,
        "n_input_rows": len(annotated_rows),
        "n_valid": n_valid,
        "n_missing_or_invalid": missing_or_invalid,
        "n_missing_key": missing_key,
        "agreement": agree,
        "kappa": kappa,
        "by_regime": _breakdown(merged, "regime"),
        "by_stratum": _breakdown(merged, "sample_stratum"),
        "disagreements": disagreements,
        "bull_rows": bull_rows,
        "bull_disagreements": bull_disagreements,
        "human_distribution": dict(Counter(row["human_intention"] for row in merged)),
        "judge_distribution": dict(
            Counter(row["judge_predicted_action"] for row in merged)
        ),
    }


def _echo_breakdown(title: str, rows: list[tuple[str, int, float, float]]) -> None:
    typer.echo(f"\n--- {title} ---")
    if not rows:
        typer.echo("  (no valid rows)")
        return
    for label, n, agree, kappa in rows:
        typer.echo(f"  {label:32s} n={n:3d}  agree={agree * 100:5.1f}%  kappa={kappa:.3f}")


@app.command()
def main(
    annotated_path: Path = typer.Argument(..., help="Filled blind_annotation CSV."),
    key_path: Path = typer.Argument(..., help="Sealed KEY CSV."),
    out: Path = typer.Option(
        Path(""),
        help="Output reconciliation CSV. Defaults to *_reconciliation.csv.",
    ),
) -> None:
    """Print agreement metrics and write a row-level reconciliation CSV."""
    if not annotated_path.exists():
        typer.echo(f"ERROR: {annotated_path} not found", err=True)
        raise typer.Exit(1)
    if not key_path.exists():
        typer.echo(f"ERROR: {key_path} not found", err=True)
        raise typer.Exit(1)

    report = reconcile_rows(_load_csv(annotated_path), _load_csv(key_path))
    if report["n_valid"] == 0:
        typer.echo("No valid annotated rows to reconcile.")
        raise typer.Exit(1)

    out_path = (
        out
        if str(out) != ""
        else annotated_path.with_name(f"{annotated_path.stem}_reconciliation.csv")
    )
    _write_csv(out_path, report["merged_rows"], RECONCILIATION_COLS)

    typer.echo("\n=== RECONCILIATION REPORT ===")
    typer.echo(f"Annotated rows: {report['n_valid']} / {report['n_input_rows']}")
    if report["n_missing_or_invalid"]:
        typer.echo(
            f"Skipped empty/invalid human_intention: {report['n_missing_or_invalid']}"
        )
    if report["n_missing_key"]:
        typer.echo(f"Skipped rows missing from key: {report['n_missing_key']}")
    typer.echo("\n--- Judge vs Human ---")
    typer.echo(f"  Action agreement = {report['agreement'] * 100:.1f}%")
    typer.echo(f"  Cohen kappa      = {report['kappa']:.3f}")
    typer.echo(f"  Human distribution = {report['human_distribution']}")
    typer.echo(f"  Judge distribution = {report['judge_distribution']}")

    _echo_breakdown("By Regime", report["by_regime"])
    _echo_breakdown("By Sample Stratum", report["by_stratum"])

    disagreements = report["disagreements"]
    typer.echo("\n--- Disagreements ---")
    typer.echo(f"  Total disagreements: {len(disagreements)}")
    for row in disagreements[:30]:
        typer.echo(
            f"  {row['date']} {row['ticker']:5s} [{row['regime']:8s}] "
            f"human={row['human_intention']:5s} "
            f"judge={row['judge_predicted_action']:5s} "
            f"agent={row['actual_agent_action']:5s} "
            f"stratum={row['sample_stratum']}"
        )
    if len(disagreements) > 30:
        typer.echo(f"  ... +{len(disagreements) - 30} more")

    bull_rows = report["bull_rows"]
    bull_disagreements = report["bull_disagreements"]
    if bull_rows:
        bull_agree = 1.0 - (len(bull_disagreements) / len(bull_rows))
        human_hold_agent_buy = sum(
            1
            for row in bull_rows
            if row["human_intention"] == "hold" and row["actual_agent_action"] == "buy"
        )
        typer.echo("\n--- Bull Focus ---")
        typer.echo(f"  n_bull = {len(bull_rows)}")
        typer.echo(f"  Judge-human agree (bull) = {bull_agree * 100:.1f}%")
        typer.echo(f"  Bull disagreements = {len(bull_disagreements)}")
        typer.echo(f"  Cases human=hold and agent=buy = {human_hold_agent_buy}")

    typer.echo(f"\nRow-level reconciliation CSV: {out_path}")


if __name__ == "__main__":
    app()
