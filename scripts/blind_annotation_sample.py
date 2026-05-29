#!/usr/bin/env python
"""Build the combined blind annotation sample for human/judge agreement.

Default output:

  results/blind_annotation_medium_run4_20260529/blind_annotation_medium_run4.csv
  results/blind_annotation_medium_run4_20260529/blind_annotation_medium_run4_KEY.csv
  results/blind_annotation_medium_run4_20260529/blind_annotation_medium_run4_MANIFEST.json

The blind CSV intentionally hides the agent action, judge action, verdict,
rationale, and sample stratum.  The key is kept separate until all human
annotations are complete.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(add_completion=False)

DEFAULT_MEDIUM_RUN = Path("runs/gpt-4-1-mini_20260529_102630/decisions.jsonl")
DEFAULT_RUN4 = Path("runs/gpt-4-1-mini_20260528_112645/decisions.jsonl")
DEFAULT_OUT_DIR = Path("results/blind_annotation_medium_run4_20260529")
DEFAULT_OUTPUT_STEM = "blind_annotation_medium_run4"

BLIND_COLS = [
    "decision_id",
    "source_run",
    "ticker",
    "t_now",
    "regime",
    "market_summary",
    "portfolio_state",
    "constraint_note",
    "human_intention",
    "human_note",
]

KEY_COLS = [
    "decision_id",
    "source_run",
    "actual_agent_action",
    "judge_predicted_action",
    "faithfulness_verdict",
    "sample_stratum",
    "regime",
    "ticker",
    "date",
]

_ACTIONS = {"buy", "sell", "hold"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decision_id(source_run: str, decision: dict[str, Any]) -> str:
    raw = f"{source_run}|{decision.get('date', '')}|{decision.get('ticker', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _market_summary(decision: dict[str, Any]) -> str:
    indicators = decision.get("indicators") or {}
    fields = [
        ("close", "close"),
        ("sma_20", "SMA20"),
        ("sma_50", "SMA50"),
        ("rsi_14", "RSI14"),
        ("macd_histogram", "MACD_hist"),
        ("macd_hist", "MACD_hist"),
        ("momentum_20d", "mom20d"),
        ("atr_14", "ATR14"),
        ("bollinger_upper", "BB_upper"),
        ("bb_upper", "BB_upper"),
        ("bollinger_lower", "BB_lower"),
        ("bb_lower", "BB_lower"),
    ]
    parts: list[str] = []
    seen: set[str] = set()
    for key, label in fields:
        if label in seen:
            continue
        value = indicators.get(key)
        if value is None:
            continue
        try:
            rendered = round(float(value), 4)
        except (TypeError, ValueError):
            rendered = str(value)
        parts.append(f"{label}={rendered}")
        seen.add(label)
    return " | ".join(parts) if parts else "(no indicators)"


def _portfolio_snapshot(decision: dict[str, Any]) -> dict[str, Any]:
    from mcp_quant_agent.eval.reasoning import _extract_portfolio_snapshot

    return _extract_portfolio_snapshot(decision)


def _position_for_ticker(
    decision: dict[str, Any],
) -> tuple[float, float]:
    ticker = str(decision.get("ticker", "")).upper()
    snapshot = _portfolio_snapshot(decision)
    for pos in snapshot.get("positions", []):
        if str(pos.get("ticker", "")).upper() == ticker:
            return float(pos.get("pct_of_nav", 0.0)), float(pos.get("quantity", 0.0))
    return 0.0, 0.0


def _portfolio_state(decision: dict[str, Any]) -> str:
    snapshot = _portfolio_snapshot(decision)
    nav = float(snapshot.get("nav", 1.0)) or 1.0
    cash_pct = float(snapshot.get("cash", 0.0)) / nav
    position_pct, _ = _position_for_ticker(decision)
    return f"Position {position_pct * 100:.1f}% of NAV | Cash {cash_pct * 100:.1f}% of NAV"


def _constraint_note(decision: dict[str, Any]) -> str:
    snapshot = _portfolio_snapshot(decision)
    nav = float(snapshot.get("nav", 1.0)) or 1.0
    cash_pct = float(snapshot.get("cash", 0.0)) / nav
    position_pct, quantity = _position_for_ticker(decision)
    notes: list[str] = []
    if position_pct >= 0.19:
        notes.append(
            f"Position at {position_pct * 100:.1f}% NAV; buying more would likely breach the 20% cap."
        )
    if quantity <= 0:
        notes.append("No position held; selling is structurally impossible.")
    if cash_pct < 0.01:
        notes.append(f"Cash is {cash_pct * 100:.2f}% of NAV; buying is not feasible.")
    return "; ".join(notes) if notes else "No structural constraint."


def _is_constraint_stress(decision: dict[str, Any]) -> bool:
    snapshot = _portfolio_snapshot(decision)
    nav = float(snapshot.get("nav", 1.0)) or 1.0
    cash_pct = float(snapshot.get("cash", 0.0)) / nav
    position_pct, _ = _position_for_ticker(decision)
    return cash_pct < 0.01 or position_pct >= 0.19


def _has_ungrounded_claim(decision: dict[str, Any]) -> bool:
    from mcp_quant_agent.eval.reasoning import (
        _extract_bars_recent,
        _extract_indicators_from_tool_outputs,
        _extract_numeric_claims,
        _ground_claim,
    )

    rationale = str(decision.get("rationale", ""))
    claims = _extract_numeric_claims(rationale)
    if not claims:
        return False
    indicators = decision.get("indicators") or _extract_indicators_from_tool_outputs(
        decision
    )
    bars_recent = _extract_bars_recent(decision)
    snapshot = _portfolio_snapshot(decision)
    return any(
        not _ground_claim(label, value, indicators, bars_recent, snapshot)
        for label, value in claims
    )


def _cached_judge_action(
    decision: dict[str, Any],
    *,
    judge_model: str,
    policy_aware: bool,
) -> str | None:
    from mcp_quant_agent.eval.reasoning import (
        _FAITHFULNESS_CACHE_DIR,
        _FAITHFULNESS_CACHE_DIR_V2,
        _faithfulness_cache_key,
        _faithfulness_cache_key_v2,
    )

    if policy_aware:
        cache_dir = _FAITHFULNESS_CACHE_DIR_V2
        cache_key = _faithfulness_cache_key_v2(judge_model, decision)
    else:
        cache_dir = _FAITHFULNESS_CACHE_DIR
        cache_key = _faithfulness_cache_key(judge_model, decision)
    cache_file = cache_dir / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    action = str(payload.get("action", "")).lower()
    return action if action in _ACTIONS else None


def _verdict_for(
    decision: dict[str, Any],
    judge_action: str | None,
    *,
    policy_aware: bool,
) -> str:
    if judge_action is None:
        return "judge_failed"
    actual = str(decision.get("action", "hold")).lower()
    if judge_action == actual:
        return "faithful"
    from mcp_quant_agent.eval.reasoning import (
        _is_constrained_hold,
        _is_constrained_hold_v2,
    )

    constrained = (
        _is_constrained_hold_v2(decision, judge_action, actual)
        if policy_aware
        else _is_constrained_hold(decision, judge_action, actual)
    )
    return "constrained" if constrained else "unfaithful"


def _candidate_from_decision(
    decision: dict[str, Any],
    *,
    source_run: str,
    judge_action: str | None,
    policy_aware: bool = True,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "decision_id": _decision_id(source_run, decision),
        "source_run": source_run,
        "judge_predicted_action": judge_action or "",
        "faithfulness_verdict": _verdict_for(
            decision,
            judge_action,
            policy_aware=policy_aware,
        ),
        "regime": str(decision.get("regime") or "unknown"),
        "sample_stratum": "",
        "is_constraint_stress": _is_constraint_stress(decision),
        "has_ungrounded_claim": _has_ungrounded_claim(decision),
    }


def _build_candidates_cache_only(
    run_paths: list[Path],
    *,
    judge_model: str,
    policy_aware: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    cache_misses: list[dict[str, Any]] = []
    for run_path in run_paths:
        source_run = run_path.parent.name
        for decision in _load_jsonl(run_path):
            if str(decision.get("action", "")).lower() == "error" or decision.get(
                "is_error", False
            ):
                continue
            judge_action = _cached_judge_action(
                decision,
                judge_model=judge_model,
                policy_aware=policy_aware,
            )
            if judge_action is None:
                cache_misses.append(
                    {
                        "source_run": source_run,
                        "date": decision.get("date"),
                        "ticker": decision.get("ticker"),
                    }
                )
            candidates.append(
                _candidate_from_decision(
                    decision,
                    source_run=source_run,
                    judge_action=judge_action,
                    policy_aware=policy_aware,
                )
            )
    return candidates, cache_misses


def _build_candidates_with_api(
    run_paths: list[Path],
    *,
    judge_model: str,
    policy_aware: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from mcp_quant_agent.eval.reasoning import compute_faithfulness_llm

    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run_path in run_paths:
        source_run = run_path.parent.name
        decisions = [
            d
            for d in _load_jsonl(run_path)
            if str(d.get("action", "")).lower() != "error"
            and not d.get("is_error", False)
        ]
        faith = compute_faithfulness_llm(
            decisions,
            judge_model=judge_model,
            seed=seed,
            policy_aware=policy_aware,
        )
        judge_by_key = {
            (str(row.get("date")), str(row.get("ticker"))): row
            for row in faith["judge_actions"]
        }
        for decision in decisions:
            key = (str(decision.get("date")), str(decision.get("ticker")))
            judge_row = judge_by_key.get(key, {})
            judge_action = str(judge_row.get("judge") or "").lower() or None
            if judge_action not in _ACTIONS:
                judge_action = None
                failures.append(
                    {
                        "source_run": source_run,
                        "date": decision.get("date"),
                        "ticker": decision.get("ticker"),
                    }
                )
            candidates.append(
                _candidate_from_decision(
                    decision,
                    source_run=source_run,
                    judge_action=judge_action,
                    policy_aware=policy_aware,
                )
            )
    return candidates, failures


def _candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate["decision_id"])


def _sample_pool(
    pool: list[dict[str, Any]],
    *,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    pool = list(pool)
    rng.shuffle(pool)
    return pool[: max(0, min(n, len(pool)))]


def _add_candidates(
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    pool: list[dict[str, Any]],
    *,
    n: int | None,
    sample_stratum: str,
    rng: random.Random,
) -> int:
    available = [c for c in pool if _candidate_key(c) not in selected_ids]
    chosen = available if n is None else _sample_pool(available, n=n, rng=rng)
    for candidate in chosen:
        candidate = dict(candidate)
        candidate["sample_stratum"] = sample_stratum
        selected.append(candidate)
        selected_ids.add(_candidate_key(candidate))
    return len(chosen)


def _stratified_run4_fill(
    pool: list[dict[str, Any]],
    *,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    by_regime: dict[str, list[dict[str, Any]]] = {}
    for candidate in pool:
        by_regime.setdefault(str(candidate["regime"]), []).append(candidate)
    regimes = sorted(by_regime)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < n and any(by_regime.values()):
        for regime in regimes:
            if len(chosen) >= n:
                break
            bucket = by_regime[regime]
            if not bucket:
                continue
            rng.shuffle(bucket)
            chosen.append(bucket.pop())
    return chosen


def select_sample(
    candidates: list[dict[str, Any]],
    *,
    medium_run: str,
    run4: str,
    target_n: int = 49,
    seed: int = 42,
    constrained_target: int = 8,
    ungrounded_target: int = 5,
    unfaithful_non_bull_target: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    manifest_notes: list[str] = []

    medium_bull_unfaithful = [
        c
        for c in candidates
        if c["source_run"] == medium_run
        and c["regime"] == "bull"
        and c["faithfulness_verdict"] == "unfaithful"
    ]
    _add_candidates(
        selected,
        selected_ids,
        medium_bull_unfaithful,
        n=None,
        sample_stratum="medium_bull_unfaithful_all",
        rng=rng,
    )

    constrained = [
        c for c in candidates if c["faithfulness_verdict"] == "constrained"
    ]
    if constrained:
        _add_candidates(
            selected,
            selected_ids,
            constrained,
            n=constrained_target,
            sample_stratum="constrained_v2",
            rng=rng,
        )
    else:
        manifest_notes.append(
            "No v2 constrained cases were available; sampled constraint_stress cases instead."
        )
        _add_candidates(
            selected,
            selected_ids,
            [c for c in candidates if c["is_constraint_stress"]],
            n=constrained_target,
            sample_stratum="constraint_stress",
            rng=rng,
        )

    _add_candidates(
        selected,
        selected_ids,
        [c for c in candidates if c["has_ungrounded_claim"]],
        n=ungrounded_target,
        sample_stratum="ungrounded_real",
        rng=rng,
    )

    unfaithful_non_bull = [
        c
        for c in candidates
        if c["faithfulness_verdict"] == "unfaithful" and c["regime"] != "bull"
    ]
    _add_candidates(
        selected,
        selected_ids,
        unfaithful_non_bull,
        n=unfaithful_non_bull_target,
        sample_stratum="unfaithful_non_bull_sample",
        rng=rng,
    )

    remaining_n = target_n - len(selected)
    run4_fill_pool = [
        c
        for c in candidates
        if c["source_run"] == run4 and _candidate_key(c) not in selected_ids
    ]
    for candidate in _stratified_run4_fill(run4_fill_pool, n=remaining_n, rng=rng):
        candidate = dict(candidate)
        candidate["sample_stratum"] = "run4_random_regime_fill"
        selected.append(candidate)
        selected_ids.add(_candidate_key(candidate))

    if len(selected) < target_n:
        fallback_pool = [
            c for c in candidates if _candidate_key(c) not in selected_ids
        ]
        _add_candidates(
            selected,
            selected_ids,
            fallback_pool,
            n=target_n - len(selected),
            sample_stratum="random_fallback_fill",
            rng=rng,
        )

    rng.shuffle(selected)
    manifest = {
        "target_n": target_n,
        "actual_n": len(selected),
        "seed": seed,
        "medium_run": medium_run,
        "run4": run4,
        "notes": manifest_notes,
        "available_counts": _counts_manifest(candidates),
        "selected_counts": _counts_manifest(selected),
    }
    return selected, manifest


def _counts_manifest(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_source_run": dict(Counter(str(c["source_run"]) for c in candidates)),
        "by_regime": dict(Counter(str(c["regime"]) for c in candidates)),
        "by_faithfulness_verdict": dict(
            Counter(str(c["faithfulness_verdict"]) for c in candidates)
        ),
        "by_sample_stratum": dict(
            Counter(str(c.get("sample_stratum") or "unassigned") for c in candidates)
        ),
        "constraint_stress": sum(1 for c in candidates if c["is_constraint_stress"]),
        "ungrounded_real": sum(1 for c in candidates if c["has_ungrounded_claim"]),
    }


def _render_rows(
    sampled: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for candidate in sampled:
        decision = candidate["decision"]
        decision_id = str(candidate["decision_id"])
        source_run = str(candidate["source_run"])
        blind_rows.append(
            {
                "decision_id": decision_id,
                "source_run": source_run,
                "ticker": decision.get("ticker", ""),
                "t_now": decision.get("date", ""),
                "regime": candidate.get("regime", ""),
                "market_summary": _market_summary(decision),
                "portfolio_state": _portfolio_state(decision),
                "constraint_note": _constraint_note(decision),
                "human_intention": "",
                "human_note": "",
            }
        )
        key_rows.append(
            {
                "decision_id": decision_id,
                "source_run": source_run,
                "actual_agent_action": str(decision.get("action", "")).lower(),
                "judge_predicted_action": candidate.get(
                    "judge_predicted_action", ""
                ),
                "faithfulness_verdict": candidate.get("faithfulness_verdict", ""),
                "sample_stratum": candidate.get("sample_stratum", ""),
                "regime": candidate.get("regime", ""),
                "ticker": decision.get("ticker", ""),
                "date": decision.get("date", ""),
            }
        )
    return blind_rows, key_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    sampled: list[dict[str, Any]],
    *,
    out_dir: Path,
    output_stem: str,
    manifest: dict[str, Any],
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = out_dir / f"{output_stem}.csv"
    key_path = out_dir / f"{output_stem}_KEY.csv"
    manifest_path = out_dir / f"{output_stem}_MANIFEST.json"
    blind_rows, key_rows = _render_rows(sampled)
    _write_csv(blind_path, blind_rows, BLIND_COLS)
    _write_csv(key_path, key_rows, KEY_COLS)
    manifest = dict(manifest)
    manifest["generated_at_utc"] = (
        dt.datetime.now(dt.UTC).replace(tzinfo=None, microsecond=0).isoformat()
    )
    manifest["blind_columns"] = BLIND_COLS
    manifest["key_columns"] = KEY_COLS
    manifest["hidden_from_blind_csv"] = [
        "actual_agent_action",
        "judge_predicted_action",
        "faithfulness_verdict",
        "sample_stratum",
        "rationale",
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return blind_path, key_path, manifest_path


@app.command()
def main(
    medium_run_path: Path = typer.Option(
        DEFAULT_MEDIUM_RUN,
        help="Medium-run decisions.jsonl.",
    ),
    run4_path: Path = typer.Option(
        DEFAULT_RUN4,
        help="Canonical Run #4 decisions.jsonl.",
    ),
    out_dir: Path = typer.Option(
        DEFAULT_OUT_DIR,
        help="Output directory.",
    ),
    output_stem: str = typer.Option(
        DEFAULT_OUTPUT_STEM,
        help="Output file stem.",
    ),
    target_n: int = typer.Option(49, help="Target sample size."),
    seed: int = typer.Option(42, help="Sampling seed."),
    judge_model: str = typer.Option("gpt-4.1-mini", help="Judge model cache key."),
    policy_aware: bool = typer.Option(True, help="Use v2 policy-aware judge cache."),
    allow_judge_api: bool = typer.Option(
        False,
        help="Allow LLM judge calls for cache misses. Default is cache-only.",
    ),
) -> None:
    """Produce combined blind annotation CSV, sealed key, and manifest."""
    run_paths = [medium_run_path, run4_path]
    if allow_judge_api:
        candidates, judge_failures = _build_candidates_with_api(
            run_paths,
            judge_model=judge_model,
            policy_aware=policy_aware,
            seed=seed,
        )
    else:
        candidates, judge_failures = _build_candidates_cache_only(
            run_paths,
            judge_model=judge_model,
            policy_aware=policy_aware,
        )

    if judge_failures and not allow_judge_api:
        sample = judge_failures[:5]
        typer.echo(
            f"[ERROR] {len(judge_failures)} judge cache misses. "
            f"First misses: {sample}",
            err=True,
        )
        typer.echo(
            "Rerun with --allow-judge-api only if you explicitly want live judge calls.",
            err=True,
        )
        raise typer.Exit(1)

    medium_run = medium_run_path.parent.name
    run4 = run4_path.parent.name
    sampled, manifest = select_sample(
        candidates,
        medium_run=medium_run,
        run4=run4,
        target_n=target_n,
        seed=seed,
    )
    manifest["judge_model"] = judge_model
    manifest["policy_aware"] = policy_aware
    manifest["allow_judge_api"] = allow_judge_api
    manifest["judge_failures"] = judge_failures

    blind_path, key_path, manifest_path = _write_outputs(
        sampled,
        out_dir=out_dir,
        output_stem=output_stem,
        manifest=manifest,
    )

    typer.echo("\n=== OUTPUTS ===")
    typer.echo(f"  ANNOTATE THIS -> {blind_path}")
    typer.echo(f"  KEY (sealed)  -> {key_path}")
    typer.echo(f"  MANIFEST      -> {manifest_path}")
    typer.echo()
    typer.echo("=== MODE D'EMPLOI ===")
    typer.echo(
        "1. Ouvre uniquement blind_annotation_medium_run4.csv et remplis "
        "human_intention=buy/sell/hold."
    )
    typer.echo(
        "2. N'ouvre pas le fichier _KEY.csv avant d'avoir fini toutes les lignes."
    )
    typer.echo(
        f"3. Lance python scripts/reconcile_annotations.py {blind_path} {key_path}"
    )


if __name__ == "__main__":
    app()
