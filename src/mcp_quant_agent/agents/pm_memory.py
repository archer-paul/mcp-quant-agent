"""Append-only PM decision log with causal past-context injection.

Adapted from TradingAgents' TradingMemoryLog pattern (reference/TradingAgents/
tradingagents/agents/utils/memory.py) for our portfolio-manager multi-agent path.

Key differences from the original:
- Portfolio-level decisions (target weights across all tickers) rather than
  per-ticker single decisions.
- Realized returns are recorded per ticker at J+1, keyed by decision date.
- Context window is bounded to the last ``n_recent`` decisions to prevent
  unbounded prompt growth during a full backtest.
- All writes are atomic (temp-file + os.replace) so a crash never corrupts the log.
- Anti-lookahead: ``get_past_context()`` only ever returns entries whose date is
  strictly before the supplied ``t_now``; the caller must pass the current bar date.

File format:
    Each entry is a markdown block separated by an HTML comment delimiter.
    The tag line encodes the key metadata in a bracket-delimited format so it
    can be parsed back without a full JSON decode:

        [2023-02-13 | JPM=0.25,NVDA=0.70 | cash=0.05 | pending]
        ...rationale...
        <!-- PM_ENTRY_END -->

    After J+1 outcomes are known the tag is updated to:

        [2023-02-13 | JPM=0.25,NVDA=0.70 | cash=0.05 | ret=JPM:+1.2%,NVDA:-0.5%]
"""

from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path
from typing import Any


class PMDecisionLog:
    """Append-only log of PM decisions for causal context injection.

    Args:
        log_path: Path to the markdown log file.  The parent directory is
            created on first write.  Pass ``None`` to disable persistence
            (in-memory only, useful for tests).
        max_entries: Maximum number of resolved entries to keep before
            dropping the oldest.  ``None`` disables rotation.
    """

    _SEPARATOR = "\n\n<!-- PM_ENTRY_END -->\n\n"
    _TAG_RE = re.compile(
        r"^\[([^\|]+)\|([^\|]+)\|([^\|]+)\|([^\]]+)\]$"
    )

    def __init__(
        self,
        log_path: str | Path | None = None,
        max_entries: int | None = 30,
    ) -> None:
        self._log_path = Path(log_path).expanduser() if log_path else None
        self._max_entries = max_entries
        # In-memory cache for the current run (avoids re-parsing the file on
        # every get_past_context call when running under PMBacktestEngine).
        self._entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def store_decision(
        self,
        *,
        date: str,
        weights: dict[str, float],
        cash_weight: float,
        rationale: str,
        regimes: dict[str, str | None] | None = None,
    ) -> None:
        """Append one pending PM decision entry.

        Args:
            date: Decision date (ISO format, e.g. "2023-02-13").
            weights: Target portfolio weights, e.g. {"JPM": 0.25, "NVDA": 0.70}.
            cash_weight: Target cash allocation (0..1).
            rationale: PM rationale string (truncated to 2000 chars on store).
            regimes: Optional dict of ticker to regime label.
        """
        weights_str = ",".join(
            f"{ticker}={weight:.4f}"
            for ticker, weight in sorted(weights.items())
        )
        cash_str = f"cash={cash_weight:.4f}"
        tag = f"[{date.strip()} | {weights_str} | {cash_str} | pending]"

        regimes_line = ""
        if regimes:
            regime_str = ", ".join(
                f"{ticker}:{label or 'unknown'}"
                for ticker, label in sorted(regimes.items())
            )
            regimes_line = f"regimes: {regime_str}\n"

        rationale_body = str(rationale or "")[:2000].strip()
        entry_text = f"{tag}\n\n{regimes_line}{rationale_body}"

        # Update in-memory cache
        self._entries.append(
            {
                "date": date.strip(),
                "weights": weights,
                "cash_weight": cash_weight,
                "rationale": rationale_body,
                "regimes": regimes or {},
                "pending": True,
                "realized_returns": {},
            }
        )

        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(entry_text + self._SEPARATOR)

    def update_with_returns(
        self,
        *,
        date: str,
        realized_returns: dict[str, float],
    ) -> None:
        """Update a pending entry with realized per-ticker returns.

        Args:
            date: Decision date matching a previous ``store_decision`` call.
            realized_returns: Dict of ticker to one-period return, e.g.
                {"JPM": 0.012, "NVDA": -0.005}.
        """
        # Update in-memory cache
        for entry in self._entries:
            if entry["date"] == date.strip() and entry["pending"]:
                entry["pending"] = False
                entry["realized_returns"] = realized_returns
                break

        if self._log_path is None or not self._log_path.exists():
            return

        ret_str = ",".join(
            f"{ticker}:{ret:+.3%}"
            for ticker, ret in sorted(realized_returns.items())
        )
        outcome_tag_suffix = f"ret={ret_str}" if ret_str else "ret=none"
        pending_prefix = f"[{date.strip()} |"

        text = self._log_path.read_text(encoding="utf-8")
        blocks = text.split(self._SEPARATOR)

        updated = False
        new_blocks: list[str] = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                new_blocks.append(block)
                continue
            first_line = stripped.splitlines()[0].strip()
            if (
                not updated
                and first_line.startswith(pending_prefix)
                and first_line.endswith("| pending]")
            ):
                # Replace | pending] with the outcome tag
                new_first = first_line[:-len("| pending]")] + f"| {outcome_tag_suffix}]"
                rest = "\n".join(stripped.splitlines()[1:])
                new_blocks.append(f"{new_first}\n{rest}")
                updated = True
            else:
                new_blocks.append(block)

        if not updated:
            return

        new_blocks = self._apply_rotation(new_blocks)
        self._atomic_write(self._SEPARATOR.join(new_blocks))

    # ------------------------------------------------------------------
    # Read path: context injection
    # ------------------------------------------------------------------

    def get_past_context(
        self,
        *,
        t_now: str | None = None,
        n_recent: int = 5,
    ) -> str:
        """Return a formatted context string for injection into PM/analyst prompts.

        Only entries whose date is strictly before ``t_now`` are included.
        When ``t_now`` is ``None``, all resolved entries are eligible.

        Returns an empty string when no past context is available.
        """
        entries = self._get_resolved_entries(t_now=t_now)
        if not entries:
            return ""

        recent = entries[-n_recent:][::-1]  # most recent first

        parts: list[str] = ["--- Past PM Decisions (most recent first) ---"]
        for entry in recent:
            parts.append(self._format_entry(entry))
        parts.append("--- End Past PM Decisions ---")
        return "\n\n".join(parts)

    def _get_resolved_entries(self, t_now: str | None) -> list[dict[str, Any]]:
        """Return resolved in-memory entries optionally filtered by date."""
        resolved = [e for e in self._entries if not e["pending"]]
        if t_now is not None:
            resolved = [e for e in resolved if e["date"] < t_now]
        return resolved

    @staticmethod
    def _format_entry(entry: dict[str, Any]) -> str:
        date = entry["date"]
        weights = entry.get("weights", {})
        cash_weight = entry.get("cash_weight", 0.0)
        rationale = entry.get("rationale", "")[:500]
        realized = entry.get("realized_returns", {})
        regimes = entry.get("regimes", {})

        weights_str = ", ".join(
            f"{t}={w:.1%}" for t, w in sorted(weights.items())
        ) or "all-cash"
        cash_str = f"cash={cash_weight:.1%}"

        lines = [f"[{date}] weights: {weights_str}, {cash_str}"]
        if regimes:
            regime_str = ", ".join(
                f"{ticker}:{label}"
                for ticker, label in sorted(regimes.items())
                if label
            )
            if regime_str:
                lines.append(f"  regimes: {regime_str}")
        if realized:
            ret_str = ", ".join(
                f"{t}:{r:+.2%}" for t, r in sorted(realized.items())
            )
            lines.append(f"  realized: {ret_str}")
        if rationale:
            lines.append(f"  rationale: {rationale}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _atomic_write(self, content: str) -> None:
        if self._log_path is None:
            return
        tmp = self._log_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self._log_path)

    def _apply_rotation(self, blocks: list[str]) -> list[str]:
        if not self._max_entries or self._max_entries <= 0:
            return blocks

        resolved_idx: list[int] = []
        for i, block in enumerate(blocks):
            stripped = block.strip()
            if not stripped:
                continue
            first = stripped.splitlines()[0].strip()
            is_resolved = (
                first.startswith("[")
                and first.endswith("]")
                and not first.endswith("| pending]")
            )
            if is_resolved:
                resolved_idx.append(i)

        to_drop = max(0, len(resolved_idx) - self._max_entries)
        drop_set = set(resolved_idx[:to_drop])
        return [block for i, block in enumerate(blocks) if i not in drop_set]

    def load_from_file(self) -> None:
        """Populate the in-memory cache by parsing the log file.

        Call this when resuming a backtest from a checkpoint so that past
        context is available without replaying all decisions.
        """
        if self._log_path is None or not self._log_path.exists():
            return
        text = self._log_path.read_text(encoding="utf-8")
        raw_entries = [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]
        self._entries = []
        for raw in raw_entries:
            parsed = self._parse_raw_entry(raw)
            if parsed is not None:
                self._entries.append(parsed)

    @staticmethod
    def _parse_raw_entry(raw: str) -> dict[str, Any] | None:
        lines = raw.strip().splitlines()
        if not lines:
            return None
        first = lines[0].strip()
        if not (first.startswith("[") and first.endswith("]")):
            return None

        inner = first[1:-1]
        parts = [p.strip() for p in inner.split("|")]
        if len(parts) < 4:
            return None

        date = parts[0]
        # Parse weights: "JPM=0.2500,NVDA=0.7000" or ""
        weights: dict[str, float] = {}
        for item in parts[1].split(","):
            item = item.strip()
            if "=" in item:
                ticker, val = item.split("=", 1)
                with suppress(ValueError):
                    weights[ticker.strip()] = float(val)
        # Parse cash
        cash_weight = 0.0
        cash_raw = parts[2].strip()
        if cash_raw.startswith("cash="):
            with suppress(ValueError):
                cash_weight = float(cash_raw[5:])

        outcome = parts[3].strip()
        pending = outcome == "pending"
        realized: dict[str, float] = {}
        if not pending and outcome.startswith("ret="):
            for item in outcome[4:].split(","):
                item = item.strip()
                if ":" in item:
                    ticker, val = item.split(":", 1)
                    with suppress(ValueError):
                        realized[ticker.strip()] = float(val.replace("%", "")) / 100.0

        body = "\n".join(lines[1:]).strip()
        return {
            "date": date,
            "weights": weights,
            "cash_weight": cash_weight,
            "rationale": body[:2000],
            "regimes": {},
            "pending": pending,
            "realized_returns": realized,
        }
