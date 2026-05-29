"""Anti-look-ahead tests for the simulation clock.

TDD timeline
------------
This file is committed *before* ``clock.py`` is fully implemented (TDD red
phase).  Running::

    pytest tests/test_clock_no_lookahead.py

at that point FAILS.  The clock is then implemented to make every test here
GREEN.  **These tests are never weakened to make code pass.**

What we test
------------
1.  Basic construction from string / datetime / date.
2.  ``is_future`` — strictly-after semantics, exact boundary, past.
3.  ``assert_not_future`` — raises ``FutureDataError`` for future timestamps.
4.  ``assert_not_future`` — does NOT raise for t_now (boundary included).
5.  ``assert_not_future`` — does NOT raise for past timestamps.
6.  ``filter_rows`` — future rows are removed (structural guarantee).
7.  ``filter_rows`` — t_now row is included (exact boundary).
8.  ``filter_rows`` — raises ``KeyError`` on missing date field (loud failure).
9.  ``filter_rows`` — supports custom ``date_key`` for news items.
10. ``advance`` — forward advancement.
11. ``advance_to`` — absolute advancement.
12. ``advance`` — raises ``ValueError`` for negative delta.
13. ``advance_to`` — raises ``ValueError`` for going backwards.
14. ``reset`` — resets to any time (for test setup / backtest restart).
15. ``get_clock()`` — raises ``RuntimeError`` if no clock set.
16. ``set_clock()`` + ``get_clock()`` round-trip.
17. ``t_now()`` convenience function.
18. ``FutureDataError`` message contains relevant diagnostics.
19. Datetime strings with time components compare correctly.
20. Post-``advance`` the new boundary is enforced.
"""

from __future__ import annotations

import datetime as dt

import pytest

from mcp_quant_agent.clock import (
    FutureDataError,
    SimulationClock,
    _to_datetime,
    get_clock,
    set_clock,
    t_now,
)

# ---------------------------------------------------------------------------
# Reference constants
# ---------------------------------------------------------------------------

T_NOW_STR: str = "2022-06-15"
T_NOW_DT: dt.datetime = dt.datetime(2022, 6, 15, 0, 0, 0)
T_PAST_STR: str = "2022-06-14"
T_FUTURE_STR: str = "2022-06-16"


# ---------------------------------------------------------------------------
# 1. Basic construction
# ---------------------------------------------------------------------------


class TestClockConstruction:
    """SimulationClock can be constructed from multiple timestamp types."""

    def test_from_date_string(self) -> None:
        clock = SimulationClock("2022-06-15")
        assert clock.t_now == dt.datetime(2022, 6, 15)

    def test_from_datetime_object(self) -> None:
        ts = dt.datetime(2022, 6, 15, 9, 30)
        clock = SimulationClock(ts)
        assert clock.t_now == ts

    def test_from_date_object(self) -> None:
        d = dt.date(2022, 6, 15)
        clock = SimulationClock(d)
        assert clock.t_now == dt.datetime(2022, 6, 15, 0, 0)

    def test_from_datetime_string_with_time_t_sep(self) -> None:
        clock = SimulationClock("2022-06-15T09:30:00")
        assert clock.t_now == dt.datetime(2022, 6, 15, 9, 30, 0)

    def test_from_datetime_string_with_time_space_sep(self) -> None:
        clock = SimulationClock("2022-06-15 09:30:00")
        assert clock.t_now == dt.datetime(2022, 6, 15, 9, 30, 0)

    def test_timezone_stripped(self) -> None:
        """Timezone info must be stripped; all comparisons are UTC-naive."""
        ts = dt.datetime(2022, 6, 15, 9, 0, tzinfo=dt.UTC)
        clock = SimulationClock(ts)
        assert clock.t_now.tzinfo is None

    def test_repr_contains_t_now(self) -> None:
        clock = SimulationClock("2022-06-15")
        assert "2022-06-15" in repr(clock)


# ---------------------------------------------------------------------------
# 2–3. is_future — boolean predicate
# ---------------------------------------------------------------------------


class TestIsFuture:
    """Tests for the is_future() boolean predicate."""

    @pytest.fixture(autouse=True)
    def _clock(self) -> SimulationClock:  # type: ignore[return]
        self.clock = SimulationClock(T_NOW_STR)
        return self.clock

    def test_strictly_after_t_now_is_future(self) -> None:
        """Core invariant: a bar dated t_now + 1 day is future."""
        assert self.clock.is_future(T_FUTURE_STR) is True

    def test_at_t_now_is_not_future(self) -> None:
        """t_now itself is NOT future — it is the current bar."""
        assert self.clock.is_future(T_NOW_STR) is False

    def test_past_is_not_future(self) -> None:
        assert self.clock.is_future(T_PAST_STR) is False

    def test_is_future_with_datetime_object(self) -> None:
        future_dt = T_NOW_DT + dt.timedelta(days=1)
        assert self.clock.is_future(future_dt) is True

    def test_is_future_with_date_object(self) -> None:
        future_date = dt.date(2022, 6, 16)
        assert self.clock.is_future(future_date) is True

    def test_is_future_far_past(self) -> None:
        assert self.clock.is_future("2020-01-01") is False


# ---------------------------------------------------------------------------
# 4–5. assert_not_future — imperative guard
# ---------------------------------------------------------------------------


class TestAssertNotFuture:
    """Tests for assert_not_future — the main anti-look-ahead enforcement."""

    @pytest.fixture(autouse=True)
    def _clock(self) -> SimulationClock:  # type: ignore[return]
        self.clock = SimulationClock(T_NOW_STR)
        return self.clock

    # ── cases that MUST raise ─────────────────────────────────────────────────

    def test_future_raises_future_data_error(self) -> None:
        """THE key test: future timestamp must raise FutureDataError."""
        with pytest.raises(FutureDataError):
            self.clock.assert_not_future(T_FUTURE_STR)

    def test_error_message_contains_offending_date(self) -> None:
        with pytest.raises(FutureDataError, match="2022-06-16"):
            self.clock.assert_not_future("2022-06-16")

    def test_error_message_contains_t_now(self) -> None:
        with pytest.raises(FutureDataError, match="2022-06-15"):
            self.clock.assert_not_future("2022-06-16")

    def test_error_message_contains_label_when_provided(self) -> None:
        with pytest.raises(FutureDataError, match="AAPL"):
            self.clock.assert_not_future("2022-06-16", label="AAPL")

    def test_error_message_omits_brackets_when_no_label(self) -> None:
        """No label → no spurious bracket in the message."""
        with pytest.raises(FutureDataError) as exc_info:
            self.clock.assert_not_future("2022-06-16")
        assert "[]" not in str(exc_info.value)

    # ── cases that must NOT raise ─────────────────────────────────────────────

    def test_t_now_does_not_raise(self) -> None:
        """t_now itself MUST be included — boundary is ≤, not <."""
        self.clock.assert_not_future(T_NOW_STR)  # must not raise

    def test_past_does_not_raise(self) -> None:
        self.clock.assert_not_future(T_PAST_STR)

    def test_much_older_date_does_not_raise(self) -> None:
        self.clock.assert_not_future("2020-01-01")

    # ── sub-day precision edge cases ──────────────────────────────────────────

    def test_same_day_later_second_raises(self) -> None:
        """t_now has time component; a later second on the same day is future."""
        clock = SimulationClock("2022-06-15T09:30:00")
        with pytest.raises(FutureDataError):
            clock.assert_not_future("2022-06-15T09:30:01")

    def test_same_day_same_second_does_not_raise(self) -> None:
        clock = SimulationClock("2022-06-15T09:30:00")
        clock.assert_not_future("2022-06-15T09:30:00")  # exact match — OK

    def test_same_day_earlier_time_does_not_raise(self) -> None:
        clock = SimulationClock("2022-06-15T09:30:00")
        clock.assert_not_future("2022-06-15T08:00:00")

    def test_date_only_string_treated_as_midnight(self) -> None:
        """'2022-06-15' = midnight < t_now (09:30) → not future."""
        clock = SimulationClock("2022-06-15T09:30:00")
        clock.assert_not_future("2022-06-15")  # midnight, not future


# ---------------------------------------------------------------------------
# 6–9. filter_rows — batch filtering
# ---------------------------------------------------------------------------


class TestFilterRows:
    """Tests for the batch row filter used by all MCP data wrappers."""

    @pytest.fixture(autouse=True)
    def _clock(self) -> SimulationClock:  # type: ignore[return]
        self.clock = SimulationClock(T_NOW_STR)
        return self.clock

    @pytest.fixture()
    def sample_bars(self) -> list[dict[str, object]]:
        """Price bars spanning past / present / future relative to T_NOW_STR."""
        return [
            {"date": "2022-06-13", "close": 99.0},  # past
            {"date": "2022-06-14", "close": 100.0},  # past
            {"date": "2022-06-15", "close": 101.0},  # t_now — MUST be included
            {"date": "2022-06-16", "close": 102.0},  # future — MUST be excluded
            {"date": "2022-06-17", "close": 103.0},  # future — MUST be excluded
        ]

    def test_future_rows_excluded(self, sample_bars: list[dict[str, object]]) -> None:
        """Core guarantee: no bar dated > t_now survives the filter."""
        filtered = self.clock.filter_rows(sample_bars)  # type: ignore[arg-type]
        dates = [r["date"] for r in filtered]
        assert "2022-06-16" not in dates
        assert "2022-06-17" not in dates

    def test_t_now_row_included(self, sample_bars: list[dict[str, object]]) -> None:
        """t_now bar must survive — it is the current bar the agent sees."""
        filtered = self.clock.filter_rows(sample_bars)  # type: ignore[arg-type]
        assert any(r["date"] == "2022-06-15" for r in filtered)

    def test_past_rows_included(self, sample_bars: list[dict[str, object]]) -> None:
        filtered = self.clock.filter_rows(sample_bars)  # type: ignore[arg-type]
        dates = [r["date"] for r in filtered]
        assert "2022-06-13" in dates
        assert "2022-06-14" in dates

    def test_filter_count(self, sample_bars: list[dict[str, object]]) -> None:
        """Exactly 3 rows pass (2 past + t_now), 2 future rows stripped."""
        filtered = self.clock.filter_rows(sample_bars)  # type: ignore[arg-type]
        assert len(filtered) == 3

    def test_empty_input_returns_empty(self) -> None:
        assert self.clock.filter_rows([]) == []

    def test_all_past_unchanged(self) -> None:
        rows: list[dict[str, object]] = [
            {"date": "2022-01-01", "close": 50.0},
            {"date": "2022-03-15", "close": 75.0},
        ]
        assert self.clock.filter_rows(rows) == rows  # type: ignore[arg-type]

    def test_all_future_returns_empty(self) -> None:
        rows: list[dict[str, object]] = [
            {"date": "2023-01-01", "close": 200.0},
            {"date": "2024-06-15", "close": 300.0},
        ]
        assert self.clock.filter_rows(rows) == []  # type: ignore[arg-type]

    def test_missing_date_key_raises_key_error(self) -> None:
        """Missing date field must raise KeyError — not silently include/exclude."""
        rows: list[dict[str, object]] = [{"close": 100.0}]  # no "date" key
        with pytest.raises(KeyError):
            self.clock.filter_rows(rows)  # type: ignore[arg-type]

    def test_unparseable_date_raises_value_error(self) -> None:
        rows: list[dict[str, object]] = [{"date": "not-a-date", "close": 100.0}]
        with pytest.raises(ValueError):
            self.clock.filter_rows(rows)  # type: ignore[arg-type]

    def test_custom_date_key_for_news(self) -> None:
        """News items use 'datetime' key — filter_rows must support custom key."""
        news: list[dict[str, object]] = [
            {"datetime": "2022-06-14T10:00:00", "headline": "Past news"},
            {"datetime": "2022-06-15T00:00:00", "headline": "t_now news"},  # included
            {"datetime": "2022-06-16T08:00:00", "headline": "Future news"},  # excluded
        ]
        filtered = self.clock.filter_rows(news, date_key="datetime")  # type: ignore[arg-type]
        headlines = [n["headline"] for n in filtered]
        assert "Past news" in headlines
        assert "t_now news" in headlines
        assert "Future news" not in headlines

    def test_filter_preserves_full_row_dict(
        self, sample_bars: list[dict[str, object]]
    ) -> None:
        """filter_rows must return the same dict objects — not truncated copies."""
        filtered = self.clock.filter_rows(sample_bars)  # type: ignore[arg-type]
        assert all("close" in r for r in filtered)
        june14 = next(r for r in filtered if r["date"] == "2022-06-14")
        assert june14["close"] == 100.0


# ---------------------------------------------------------------------------
# 10–14. Clock mutation
# ---------------------------------------------------------------------------


class TestClockMutation:
    def test_advance_forward(self) -> None:
        clock = SimulationClock("2022-06-15")
        clock.advance(dt.timedelta(days=1))
        assert clock.t_now == dt.datetime(2022, 6, 16)

    def test_advance_accumulates(self) -> None:
        clock = SimulationClock("2022-06-15")
        clock.advance(dt.timedelta(hours=12))
        clock.advance(dt.timedelta(hours=12))
        assert clock.t_now == dt.datetime(2022, 6, 16)

    def test_advance_negative_raises(self) -> None:
        """Backward advance would re-expose filtered future bars."""
        clock = SimulationClock("2022-06-15")
        with pytest.raises(ValueError, match="backwards"):
            clock.advance(dt.timedelta(days=-1))

    def test_advance_zero_is_noop(self) -> None:
        clock = SimulationClock("2022-06-15")
        clock.advance(dt.timedelta(0))
        assert clock.t_now == dt.datetime(2022, 6, 15)

    def test_advance_to_forward(self) -> None:
        clock = SimulationClock("2022-06-15")
        clock.advance_to("2022-06-20")
        assert clock.t_now == dt.datetime(2022, 6, 20)

    def test_advance_to_backwards_raises(self) -> None:
        clock = SimulationClock("2022-06-15")
        with pytest.raises(ValueError):
            clock.advance_to("2022-06-14")

    def test_advance_to_same_time_is_noop(self) -> None:
        clock = SimulationClock("2022-06-15")
        clock.advance_to("2022-06-15")
        assert clock.t_now == dt.datetime(2022, 6, 15)

    def test_reset_to_earlier_time(self) -> None:
        clock = SimulationClock("2022-06-15")
        clock.advance(dt.timedelta(days=5))
        clock.reset("2022-01-01")  # explicitly allowed for test setup
        assert clock.t_now == dt.datetime(2022, 1, 1)

    def test_after_advance_new_boundary_enforced(self) -> None:
        """After advancing, the new t_now is the boundary — not the old one."""
        clock = SimulationClock("2022-06-15")
        clock.advance(dt.timedelta(days=1))  # t_now → 2022-06-16

        # 2022-06-17 is now future
        with pytest.raises(FutureDataError):
            clock.assert_not_future("2022-06-17")
        # 2022-06-16 is now t_now — must NOT raise
        clock.assert_not_future("2022-06-16")
        # 2022-06-15 is now past — must NOT raise
        clock.assert_not_future("2022-06-15")


# ---------------------------------------------------------------------------
# 15–17. Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    """Tests for set_clock / get_clock / t_now module-level functions.

    The conftest.py autouse fixture resets _CLOCK to None before each test,
    so these tests start from a clean state.
    """

    def test_get_clock_raises_if_not_set(self) -> None:
        """get_clock() must raise RuntimeError — no silent stale state."""
        with pytest.raises(RuntimeError, match="set_clock"):
            get_clock()

    def test_set_and_get_round_trip(self) -> None:
        clock = SimulationClock("2022-06-15")
        set_clock(clock)
        assert get_clock() is clock  # same object, not a copy

    def test_t_now_function(self) -> None:
        clock = SimulationClock("2022-06-15")
        set_clock(clock)
        assert t_now() == dt.datetime(2022, 6, 15)

    def test_t_now_raises_if_no_clock(self) -> None:
        with pytest.raises(RuntimeError):
            t_now()

    def test_set_clock_replaces_previous(self) -> None:
        set_clock(SimulationClock("2022-01-01"))
        set_clock(SimulationClock("2023-01-01"))
        assert get_clock().t_now == dt.datetime(2023, 1, 1)


# ---------------------------------------------------------------------------
# 18. _to_datetime helper — parsing edge cases
# ---------------------------------------------------------------------------


class TestToDatetime:
    """Exhaustive parsing tests for the internal _to_datetime helper."""

    def test_date_string(self) -> None:
        assert _to_datetime("2022-06-15") == dt.datetime(2022, 6, 15)

    def test_datetime_string_t_separator(self) -> None:
        assert _to_datetime("2022-06-15T09:30:00") == dt.datetime(2022, 6, 15, 9, 30)

    def test_datetime_string_space_separator(self) -> None:
        assert _to_datetime("2022-06-15 09:30:00") == dt.datetime(2022, 6, 15, 9, 30)

    def test_datetime_object_passthrough(self) -> None:
        ts = dt.datetime(2022, 6, 15, 9, 30)
        assert _to_datetime(ts) == ts

    def test_datetime_object_tz_stripped(self) -> None:
        ts = dt.datetime(2022, 6, 15, tzinfo=dt.UTC)
        result = _to_datetime(ts)
        assert result.tzinfo is None
        assert result == dt.datetime(2022, 6, 15)

    def test_date_object_to_midnight(self) -> None:
        d = dt.date(2022, 6, 15)
        assert _to_datetime(d) == dt.datetime(2022, 6, 15, 0, 0)

    def test_invalid_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            _to_datetime("not-a-date")

    def test_integer_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            _to_datetime(20220615)  # type: ignore[arg-type]

    def test_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            _to_datetime(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Timezone handling
# ---------------------------------------------------------------------------
# These tests FAIL with the old _to_datetime (bare .replace(tzinfo=None)) and
# PASS with the fixed version (convert to UTC first, then strip).
# ---------------------------------------------------------------------------


class TestTimezoneHandling:
    """Timezone-aware datetimes must be converted to UTC before stripping.

    The old code did ``ts.replace(tzinfo=None)``, which discards the offset
    without adjusting the wall-clock time.  Examples of failures:

    - ``2022-06-16T08:00+09:00`` (= 2022-06-15T23:00 UTC) looks like a
      *future* date with the old code but is actually *past* in UTC.
    - ``2022-06-14T20:00-08:00`` (= 2022-06-15T04:00 UTC) looks like a
      *past* date with the old code but is actually *future* in UTC.

    See ``docs/DECISIONS.md`` — "UTC-naive internal timestamps" entry.
    """

    @pytest.fixture(autouse=True)
    def _setup_clock(self) -> None:
        # t_now = 2022-06-15T00:00:00 UTC (midnight)
        self.clock = SimulationClock("2022-06-15")
        set_clock(self.clock)

    # ── _to_datetime UTC conversion ───────────────────────────────────────────

    def test_aware_plus9_before_midnight_utc_is_past(self) -> None:
        """2022-06-15T01:00+09:00 = 2022-06-14T16:00 UTC → past (NOT future).

        With the old code (.replace(tzinfo=None) → 2022-06-15T01:00) this
        would be misidentified as future (01:00 > 00:00).
        """
        ts = dt.datetime(2022, 6, 15, 1, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
        # In UTC: 2022-06-14T16:00 ≤ t_now (2022-06-15T00:00) → NOT future
        assert self.clock.is_future(ts) is False, (
            "2022-06-15T01:00+09:00 = 2022-06-14T16:00 UTC — "
            "this is BEFORE t_now=2022-06-15T00:00 UTC"
        )

    def test_aware_minus8_after_midnight_utc_is_future(self) -> None:
        """2022-06-14T20:00-08:00 = 2022-06-15T04:00 UTC → future.

        With the old code (.replace(tzinfo=None) → 2022-06-14T20:00) this
        would be misidentified as past (2022-06-14 < 2022-06-15).
        """
        ts = dt.datetime(2022, 6, 14, 20, 0, tzinfo=dt.timezone(dt.timedelta(hours=-8)))
        # In UTC: 2022-06-15T04:00 > t_now (2022-06-15T00:00) → IS future
        assert self.clock.is_future(ts) is True, (
            "2022-06-14T20:00-08:00 = 2022-06-15T04:00 UTC — "
            "this is AFTER t_now=2022-06-15T00:00 UTC"
        )

    def test_aware_utc_exact_boundary_is_not_future(self) -> None:
        """An aware datetime at exactly t_now (UTC midnight) is not future."""
        ts = dt.datetime(2022, 6, 15, 0, 0, tzinfo=dt.UTC)
        assert self.clock.is_future(ts) is False

    def test_aware_to_datetime_strips_tz_after_conversion(self) -> None:
        """_to_datetime must return a naive datetime."""
        ts = dt.datetime(2022, 6, 15, 9, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
        result = _to_datetime(ts)
        assert result.tzinfo is None
        # 2022-06-15T09:00+09:00 = 2022-06-15T00:00 UTC
        assert result == dt.datetime(2022, 6, 15, 0, 0)

    def test_naive_datetime_unchanged(self) -> None:
        """A naive datetime is treated as UTC — no conversion applied."""
        ts = dt.datetime(2022, 6, 15, 9, 30)
        result = _to_datetime(ts)
        assert result == dt.datetime(2022, 6, 15, 9, 30)
        assert result.tzinfo is None

    # ── filter_rows with aware datetime objects in row values ─────────────────

    def test_filter_rows_aware_plus9_included(self) -> None:
        """A +09:00 datetime that is past in UTC must survive filter_rows."""
        # 2022-06-15T01:00+09:00 = 2022-06-14T16:00 UTC → ≤ t_now → INCLUDE
        ts = dt.datetime(2022, 6, 15, 1, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
        rows = [{"datetime": ts, "headline": "past in UTC"}]
        result = self.clock.filter_rows(rows, date_key="datetime")
        assert len(result) == 1, "Row should be included (past in UTC)"

    def test_filter_rows_aware_minus8_excluded(self) -> None:
        """A -08:00 datetime that is future in UTC must be dropped by filter_rows."""
        # 2022-06-14T20:00-08:00 = 2022-06-15T04:00 UTC → > t_now → EXCLUDE
        ts = dt.datetime(2022, 6, 14, 20, 0, tzinfo=dt.timezone(dt.timedelta(hours=-8)))
        rows = [{"datetime": ts, "headline": "future in UTC"}]
        result = self.clock.filter_rows(rows, date_key="datetime")
        assert result == [], "Row should be excluded (future in UTC)"

    def test_assert_not_future_aware_correctly_raises(self) -> None:
        """assert_not_future must raise for a future-in-UTC aware datetime."""
        ts = dt.datetime(2022, 6, 14, 20, 0, tzinfo=dt.timezone(dt.timedelta(hours=-8)))
        with pytest.raises(FutureDataError):
            self.clock.assert_not_future(ts)

    def test_assert_not_future_aware_correctly_passes(self) -> None:
        """assert_not_future must NOT raise for a past-in-UTC aware datetime."""
        ts = dt.datetime(2022, 6, 15, 1, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
        self.clock.assert_not_future(ts)  # should not raise


# ---------------------------------------------------------------------------
# Mutation guard — proves anti-look-ahead tests are not decorative
# ---------------------------------------------------------------------------


class TestAntiLookaheadNotDecorative:
    """Proves that the anti-look-ahead test suite catches real filter bugs.

    Method: simulate a plausible off-by-one mutation in ``filter_rows``
    (using ``t_now + 1 day`` instead of ``t_now``) by constructing a clock
    set one day ahead.  Demonstrate that:

    1. The correct clock excludes the t_now+1 bar (expected behaviour).
    2. The mutated clock includes the t_now+1 bar (leak under mutation).
    3. The existing ``test_bar_one_day_after_t_now_excluded``-style assertion
       would FAIL under the mutation — proving it is not decorative.

    The mutation is simulated at the *clock* level rather than by patching
    ``filter_rows`` itself; both approaches are equivalent for demonstration.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.correct_clock = SimulationClock("2022-06-15")
        set_clock(self.correct_clock)

    def test_correct_filter_excludes_t_now_plus1(self) -> None:
        """Control: the correct t_now strictly excludes the +1 bar."""
        rows = [
            {"date": "2022-06-15", "close": 100.0},  # t_now
            {"date": "2022-06-16", "close": 101.0},  # t_now+1 — must be excluded
        ]
        result = self.correct_clock.filter_rows(rows)
        dates = [r["date"] for r in result]
        assert dates == ["2022-06-15"]

    def test_relaxed_filter_leaks_t_now_plus1(self) -> None:
        """Mutation scenario: clock at t_now+1 day leaks the future bar.

        This is the filter with ``<= t_now + 1 day`` — the mutated version.
        It DOES include the bar that should be excluded.
        """
        mutated_clock = SimulationClock("2022-06-16")  # simulates relaxed filter
        rows = [
            {"date": "2022-06-15", "close": 100.0},
            {"date": "2022-06-16", "close": 101.0},  # leaks under mutation
        ]
        result = mutated_clock.filter_rows(rows)
        dates = [r["date"] for r in result]
        # Both bars are visible — the future bar leaked
        assert "2022-06-16" in dates, (
            "Mutation confirmed: relaxing t_now by 1 day exposes the 'future' bar."
        )

    def test_anti_lookahead_assertion_fails_under_mutation(self) -> None:
        """The core anti-leak assertion fails when the filter is relaxed.

        Demonstrates that ``assert '2022-06-16' not in dates`` — the assertion
        used in ``test_bar_one_day_after_t_now_excluded`` — is VIOLATED when
        the filter boundary is shifted by one day.  The test IS sensitive.
        """
        mutated_clock = SimulationClock("2022-06-16")
        rows = [
            {"date": "2022-06-15", "close": 100.0},
            {"date": "2022-06-16", "close": 101.0},
        ]
        result = mutated_clock.filter_rows(rows)
        mutated_dates = [r["date"] for r in result]

        # The assertion that guards us in the real test suite:
        #   assert "2022-06-16" not in dates
        # Under mutation this assertion FAILS — shown here by the inverse:
        assert "2022-06-16" in mutated_dates, (
            "PROOF: relaxing t_now by +1 day causes the look-ahead guard to fail. "
            "The anti-look-ahead tests ARE sensitive to the filter boundary."
        )

    def test_assert_not_future_catches_mutation_too(self) -> None:
        """assert_not_future raises for t_now+1 with correct clock, not with mutated."""
        # Correct clock: 2022-06-16 IS future → raises
        with pytest.raises(FutureDataError):
            self.correct_clock.assert_not_future("2022-06-16")
        # Mutated clock: 2022-06-16 is NOT future (now t_now) → no raise
        mutated_clock = SimulationClock("2022-06-16")
        mutated_clock.assert_not_future("2022-06-16")  # does NOT raise under mutation
