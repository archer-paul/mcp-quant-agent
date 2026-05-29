"""Local LLM usage telemetry for reproducible run accounting."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()

# Prices current at implementation time; used for run-local estimates only.
# Keep DECISIONS.md as the methodological source if pricing changes.
MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}


def usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def estimate_cost_usd(model: str, usage: dict[str, int], *, cache_hit: bool) -> float:
    if cache_hit:
        return 0.0
    pricing = MODEL_PRICING_USD_PER_1M.get(model, MODEL_PRICING_USD_PER_1M["gpt-4.1-mini"])
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    return round(
        prompt * pricing["input"] / 1_000_000
        + completion * pricing["output"] / 1_000_000,
        8,
    )


def log_llm_event(
    path: Path | None,
    *,
    run_id: str | None,
    model: str,
    label: str,
    span_type: str,
    cache_hit: bool,
    prompt_hash: str,
    elapsed_ms: float,
    usage: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return
    usage_payload = dict(usage or {})
    event: dict[str, Any] = {
        "ts_unix": round(time.time(), 3),
        "run_id": run_id,
        "model": model,
        "label": label,
        "span_type": span_type,
        "cache_hit": cache_hit,
        "api_call": not cache_hit,
        "prompt_hash": prompt_hash,
        "elapsed_ms": round(float(elapsed_ms), 1),
        "prompt_tokens": usage_payload.get("prompt_tokens"),
        "completion_tokens": usage_payload.get("completion_tokens"),
        "total_tokens": usage_payload.get("total_tokens"),
        "incremental_cost_usd": estimate_cost_usd(model, usage_payload, cache_hit=cache_hit),
    }
    if extra:
        event.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True)
    with _LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def summarize_usage(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "n_events": 0,
            "n_api_calls": 0,
            "n_cache_hits": 0,
            "cache_hit_rate": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "wall_clock_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "incremental_cost_usd": 0.0,
            "by_span_type": {},
        }
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = len(rows)
    n_cache = sum(1 for row in rows if row.get("cache_hit"))
    by_span: dict[str, dict[str, Any]] = {}
    for row in rows:
        span = str(row.get("span_type") or "unknown")
        bucket = by_span.setdefault(
            span,
            {
                "n_events": 0,
                "n_api_calls": 0,
                "n_cache_hits": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "incremental_cost_usd": 0.0,
                "latencies_ms": [],
            },
        )
        bucket["n_events"] += 1
        bucket["n_api_calls"] += 0 if row.get("cache_hit") else 1
        bucket["n_cache_hits"] += 1 if row.get("cache_hit") else 0
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            bucket[key] += int(row.get(key) or 0)
        bucket["incremental_cost_usd"] += float(row.get("incremental_cost_usd") or 0.0)
        bucket["latencies_ms"].append(float(row.get("elapsed_ms") or 0.0))
    latencies = [float(row.get("elapsed_ms") or 0.0) for row in rows]
    timestamps = [float(row.get("ts_unix") or 0.0) for row in rows if row.get("ts_unix")]
    return {
        "n_events": n,
        "n_api_calls": n - n_cache,
        "n_cache_hits": n_cache,
        "cache_hit_rate": round(n_cache / n, 4) if n else 0.0,
        "latency_ms_p50": round(_percentile(latencies, 0.50), 1),
        "latency_ms_p95": round(_percentile(latencies, 0.95), 1),
        "wall_clock_seconds": round(max(timestamps) - min(timestamps), 1)
        if len(timestamps) >= 2
        else 0.0,
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        "incremental_cost_usd": round(
            sum(float(row.get("incremental_cost_usd") or 0.0) for row in rows),
            6,
        ),
        "by_span_type": {
            key: {
                **{k: v for k, v in value.items() if k != "latencies_ms"},
                "cache_hit_rate": round(
                    value["n_cache_hits"] / value["n_events"], 4
                )
                if value["n_events"]
                else 0.0,
                "latency_ms_p50": round(_percentile(value["latencies_ms"], 0.50), 1),
                "latency_ms_p95": round(_percentile(value["latencies_ms"], 0.95), 1),
                "incremental_cost_usd": round(value["incremental_cost_usd"], 6),
            }
            for key, value in sorted(by_span.items())
        },
    }
