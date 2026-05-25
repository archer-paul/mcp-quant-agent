"""Simulation clock — the single source of truth for ``t_now``.

Design
------
:class:`SimulationClock` holds the current simulation timestamp.  All MCP
wrappers and data helpers **MUST** consult ``get_clock().t_now`` before
returning any data; they **MUST** call :meth:`SimulationClock.filter_rows`
(or :meth:`SimulationClock.assert_not_future`) to strip any item whose
timestamp is strictly after ``t_now``.

"MCP-as-time-machine"
---------------------
The key architectural claim of this thesis is that look-ahead bias is
*structurally impossible*, not merely avoided by discipline.  This module
is the enforcement point:

1. ``t_now`` is set by the backtest engine before each bar.
2. Every MCP data tool reads ``t_now`` via ``get_clock()``.
3. ``filter_rows`` strips any bar/news item beyond ``t_now`` before it
   can reach the agent.
4. Anti-look-ahead tests (``tests/test_clock_no_lookahead.py``) FAIL if
   the filter is removed or weakened.

Module-level singleton
----------------------
There is exactly **one** module-level ``_CLOCK`` instance (see
:func:`get_clock` / :func:`set_clock`).  CLAUDE.md explicitly permits this
because MCP tool calls arrive from external processes over stdio/HTTP — there
is no Python call stack through which a clock could be dependency-injected.
The singleton is explicit (not hidden) via :func:`set_clock`, which MUST be
called at server / backtest startup before any tool is invoked.

Anti-look-ahead guarantee
--------------------------
Every data-returning function MUST eventually call
:meth:`SimulationClock.filter_rows` on the rows it is about to yield.
:meth:`SimulationClock.assert_not_future` is the single-item variant for
cases where a future timestamp is unambiguously a bug (use ``filter_rows``
for expected batch trimming).

Both helpers RAISE loudly rather than silently dropping data.  A silent
fallback that masks a look-ahead leak is far worse than a visible crash.

Adjusted-price leakage note
----------------------------
Using split/dividend-adjusted prices derived from *current* adjustment
factors encodes future corporate actions into historical prices — a subtle
form of look-ahead.  See ``mcp_servers/data/yfinance_source.py`` for how
we handle this (``auto_adjust=False``) and ``docs/DECISIONS.md`` for the
rationale.
"""

from __future__ import annotations

import datetime as dt
import threading
from typing import Any

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class FutureDataError(Exception):
    """Raised when data for a timestamp strictly after ``t_now`` is requested.

    This exception is the main anti-look-ahead guard.  It MUST be raised;
    never catch and suppress it unless you are inside a deliberate filtering
    loop (use :meth:`SimulationClock.filter_rows` for that instead).

    Example
    -------
    >>> clock = SimulationClock("2022-06-15")
    >>> clock.assert_not_future("2022-06-16")
    Traceback (most recent call last):
        ...
    mcp_quant_agent.clock.FutureDataError: LOOK-AHEAD VIOLATION ...
    """


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

DateLike = dt.datetime | dt.date | str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_datetime(ts: DateLike) -> dt.datetime:
    """Normalise a timestamp to a UTC-naive :class:`~datetime.datetime`.

    Supported input types:

    - :class:`~datetime.datetime` — timezone info is stripped (all comparisons
      are UTC-naive to avoid ambiguous offset arithmetic in backtests).
    - :class:`~datetime.date` — converted to midnight ``00:00:00``.
    - ``str`` — parsed as ISO-8601.  Accepted formats:
      ``"YYYY-MM-DD"``, ``"YYYY-MM-DDTHH:MM:SS"``, ``"YYYY-MM-DDTHH:MM"``,
      ``"YYYY-MM-DD HH:MM:SS"``, ``"YYYY-MM-DD HH:MM"``.

    Raises
    ------
    ValueError
        If a string cannot be parsed as any of the supported formats.
    TypeError
        If *ts* is not a ``datetime``, ``date``, or ``str``.
    """
    if isinstance(ts, dt.datetime):
        # Strip timezone — we work entirely in UTC-naive datetimes.
        # Reason: yfinance / Finnhub timestamps are often timezone-naive;
        # mixing naive and aware datetimes causes comparison errors.
        return ts.replace(tzinfo=None)
    if isinstance(ts, dt.date):
        # date.date → datetime at midnight
        return dt.datetime(ts.year, ts.month, ts.day)
    if isinstance(ts, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                return dt.datetime.strptime(ts, fmt)
            except ValueError:
                continue
        raise ValueError(
            f"Cannot parse timestamp string: {ts!r}. "
            "Expected ISO-8601 (e.g. '2022-06-15' or '2022-06-15T09:30:00')."
        )
    raise TypeError(
        f"Expected datetime.datetime, datetime.date, or str; got {type(ts).__name__!r}."
    )


# ---------------------------------------------------------------------------
# SimulationClock
# ---------------------------------------------------------------------------


class SimulationClock:
    """Simulation clock that gates all data access by ``t_now``.

    Parameters
    ----------
    t_now:
        Initial simulation time.  Accepts a :class:`~datetime.datetime`,
        :class:`~datetime.date`, or ISO-8601 string.  All internal
        comparisons use UTC-naive datetimes.

    Thread safety
    -------------
    All mutation methods (``advance``, ``advance_to``, ``reset``) acquire
    an internal :class:`threading.Lock`.  Read access (``t_now`` property)
    is a simple attribute read — safe under Python's GIL for single-object
    reads, no explicit locking needed.

    Examples
    --------
    >>> clock = SimulationClock("2022-06-15")
    >>> clock.t_now
    datetime.datetime(2022, 6, 15, 0, 0)
    >>> clock.is_future("2022-06-16")
    True
    >>> clock.is_future("2022-06-15")   # t_now itself is NOT future
    False
    """

    def __init__(self, t_now: DateLike) -> None:
        self._t_now: dt.datetime = _to_datetime(t_now)
        self._lock: threading.Lock = threading.Lock()

    # ── read ──────────────────────────────────────────────────────────────────

    @property
    def t_now(self) -> dt.datetime:
        """Current simulation timestamp (UTC-naive)."""
        return self._t_now

    def is_future(self, ts: DateLike) -> bool:
        """Return ``True`` if *ts* is **strictly after** ``t_now``.

        This is the pure boolean predicate.  For imperative enforcement
        (raising on violation), use :meth:`assert_not_future`.
        """
        return _to_datetime(ts) > self._t_now

    # ── anti-look-ahead guards ────────────────────────────────────────────────

    def assert_not_future(self, ts: DateLike, label: str = "") -> None:
        """Raise :class:`FutureDataError` if *ts* is strictly after ``t_now``.

        This is the **single-item** anti-look-ahead guard.  Call it before
        returning or consuming any time-indexed datum to verify the clock
        contract is honoured.

        The guard **RAISES** rather than returning silently because a future
        timestamp here is a genuine bug, not an expected trim.  Use
        :meth:`filter_rows` for batch trimming where dropping future items
        is the expected behaviour (e.g. slicing a cached price series to
        the backtest window).

        Parameters
        ----------
        ts:
            Timestamp to check.
        label:
            Optional context string included in the error message (e.g.
            a ticker symbol or field name) to aid debugging.

        Raises
        ------
        FutureDataError
            If ``_to_datetime(ts) > t_now``.

        Examples
        --------
        >>> clock = SimulationClock("2022-06-15")
        >>> clock.assert_not_future("2022-06-15")   # t_now — OK
        >>> clock.assert_not_future("2022-06-14")   # past — OK
        >>> clock.assert_not_future("2022-06-16")
        Traceback (most recent call last):
            ...
        FutureDataError: LOOK-AHEAD VIOLATION: ...
        """
        ts_dt = _to_datetime(ts)
        if ts_dt > self._t_now:
            label_str = f" [{label}]" if label else ""
            raise FutureDataError(
                f"LOOK-AHEAD VIOLATION: data timestamp {ts_dt.date()} "
                f"is strictly after t_now={self._t_now.date()}"
                f"{label_str}. "
                "A data source is leaking future data — check the t_now filter "
                "in the relevant MCP data wrapper."
            )

    def filter_rows(
        self,
        rows: list[dict[str, Any]],
        date_key: str = "date",
    ) -> list[dict[str, Any]]:
        """Return only the rows whose *date_key* value is ≤ ``t_now``.

        This is the **batch** filtering operation for price series and news
        feeds.  It silently drops future rows — that is the intended
        anti-look-ahead trim — unlike :meth:`assert_not_future` which raises
        for a single offending item.

        The filter is the last line of defence at the clock layer: even if
        a downstream data source (yfinance, Finnhub, cached parquet) returns
        rows beyond ``t_now``, this method strips them before they reach the
        agent.

        Parameters
        ----------
        rows:
            List of dicts, each containing a date or datetime field.
        date_key:
            The key in each dict whose value is the timestamp.  Defaults to
            ``"date"`` (OHLCV bars); pass ``"datetime"`` for news items.

        Returns
        -------
        list[dict[str, Any]]
            New list containing only rows with timestamp ≤ ``t_now``.
            Order is preserved.

        Raises
        ------
        KeyError
            If any row is missing *date_key*.  Fail loudly — do not silently
            include or exclude rows with unknown timestamps.
        ValueError
            If a row's *date_key* value cannot be parsed as a timestamp.

        Examples
        --------
        >>> clock = SimulationClock("2022-06-15")
        >>> rows = [
        ...     {"date": "2022-06-14", "close": 100.0},
        ...     {"date": "2022-06-15", "close": 101.0},  # t_now — included
        ...     {"date": "2022-06-16", "close": 102.0},  # future — excluded
        ... ]
        >>> [r["date"] for r in clock.filter_rows(rows)]
        ['2022-06-14', '2022-06-15']
        """
        result: list[dict[str, Any]] = []
        for row in rows:
            # Fail loudly if the date key is absent.
            # "Fail loudly" is the project convention — a silent include/exclude
            # of a row with an unknown timestamp is worse than a visible crash.
            if date_key not in row:
                raise KeyError(
                    f"Row missing expected date field {date_key!r}: {row!r}. "
                    f"Available keys: {sorted(row.keys())}"
                )
            raw_ts = row[date_key]
            try:
                ts_dt = _to_datetime(raw_ts)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Cannot parse {date_key!r}={raw_ts!r} in row: {row!r}"
                ) from exc
            # Core invariant: future rows are excluded.
            if ts_dt <= self._t_now:
                result.append(row)
        return result

    # ── mutation ──────────────────────────────────────────────────────────────

    def advance(self, delta: dt.timedelta) -> None:
        """Advance the clock forward by *delta*.

        Raises
        ------
        ValueError
            If *delta* is negative.  Backward time travel could re-expose
            a previously-filtered future bar, reintroducing look-ahead.
            Use :meth:`reset` if you need to explicitly set an earlier time
            (only in test setup or between independent backtests).
        """
        if delta < dt.timedelta(0):
            raise ValueError(
                f"Cannot advance clock backwards by {delta}. "
                "Use SimulationClock.reset() to set an earlier time explicitly."
            )
        with self._lock:
            self._t_now += delta

    def advance_to(self, new_time: DateLike) -> None:
        """Advance the clock to an absolute *new_time*.

        Equivalent to ``advance(new_time - t_now)`` but accepts an absolute
        timestamp, which is more convenient in the backtest event loop.

        Raises
        ------
        ValueError
            If *new_time* is strictly before ``t_now``.
        """
        new_dt = _to_datetime(new_time)
        with self._lock:
            if new_dt < self._t_now:
                raise ValueError(
                    f"Cannot advance clock backwards: "
                    f"{new_dt.date()} < t_now={self._t_now.date()}. "
                    "Use SimulationClock.reset() if you intend to restart."
                )
            self._t_now = new_dt

    def reset(self, t_now: DateLike) -> None:
        """Reset the clock to *t_now* (any direction allowed).

        Intended **only** for test setup and the start of independent
        backtests.  In production / backtest event loops, prefer
        :meth:`advance` / :meth:`advance_to` so that accidental backward
        jumps are caught immediately.
        """
        with self._lock:
            self._t_now = _to_datetime(t_now)

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"SimulationClock(t_now={self._t_now.isoformat()})"


# ---------------------------------------------------------------------------
# Module-level singleton (the one explicitly permitted global — see CLAUDE.md)
# ---------------------------------------------------------------------------
# CLAUDE.md permits a module-level clock because MCP tool calls arrive from
# external processes over stdio/HTTP; there is no Python call stack to
# inject the clock through. The singleton is explicit and testable via
# set_clock() / get_clock(), NOT hidden.
#
# Usage:
#   from mcp_quant_agent.clock import set_clock, get_clock, t_now
#   set_clock(SimulationClock("2022-06-15"))  # at server/backtest startup
#   current = t_now()                         # in data wrappers / hot path

_CLOCK: SimulationClock | None = None
_CLOCK_LOCK: threading.Lock = threading.Lock()


def set_clock(clock: SimulationClock) -> None:
    """Set the module-level clock.

    Call this once at server startup or at the start of a backtest run,
    **before** any MCP tool is invoked.

    In tests, call it inside each test or in a fixture, and use the
    ``autouse`` conftest fixture to reset ``_CLOCK`` to ``None`` after each
    test to ensure isolation.
    """
    global _CLOCK
    with _CLOCK_LOCK:
        _CLOCK = clock


def get_clock() -> SimulationClock:
    """Return the current module-level clock.

    Raises
    ------
    RuntimeError
        If :func:`set_clock` has not been called yet.  This is the loud
        failure mode: a tool that calls ``get_clock()`` before the clock
        is configured crashes immediately rather than silently returning
        stale or unfiltered data.
    """
    if _CLOCK is None:
        raise RuntimeError(
            "No simulation clock configured. "
            "Call clock.set_clock(SimulationClock(t_now)) at startup (or in "
            "the backtest engine before the first bar is processed)."
        )
    return _CLOCK


def t_now() -> dt.datetime:
    """Convenience shorthand for ``get_clock().t_now``.

    Prefer this in hot-path code (e.g. the per-bar loop) to avoid two
    attribute lookups on every call.
    """
    return get_clock().t_now
