"""Unit tests for the ingest-boundary parsing.

These cover the gotchas that would otherwise fail silently. Every case here
was observed in real FastF1 data, not invented.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from app.ingestion import transform as t


class TestParseTrackStatus:
    """The single most important parse in the pipeline."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", [1]),
            ("4", [4]),
            # Multi-flag strings, all observed at the 2024 Brazilian GP.
            ("12", [1, 2]),
            ("14", [1, 4]),
            ("45", [4, 5]),
            ("124", [1, 2, 4]),
            ("671", [1, 6, 7]),
            ("1267", [1, 2, 6, 7]),
        ],
    )
    def test_parses_multi_flag_strings(self, raw, expected):
        assert t.parse_track_status(raw) == expected

    def test_output_is_canonical_regardless_of_input_order(self):
        """'12' and '21' are the same flag set and must compare equal.

        Without sorting, Postgres stores {1,2} and {2,1} as distinct arrays
        and a GROUP BY splits one real condition across two buckets.
        """
        assert t.parse_track_status("12") == t.parse_track_status("21")
        assert t.parse_track_status("1267") == t.parse_track_status("7621")

    def test_deduplicates(self):
        assert t.parse_track_status("111") == [1]

    @pytest.mark.parametrize("raw", [None, "", float("nan"), pd.NaT])
    def test_missing_becomes_none_not_empty(self, raw):
        """None (unknown) must stay distinguishable from [1] (clear)."""
        assert t.parse_track_status(raw) is None

    def test_not_cast_to_int(self):
        """Regression guard for the bug this parser exists to prevent."""
        assert t.parse_track_status("1267") != 1267
        assert isinstance(t.parse_track_status("1267"), list)


class TestIsGreenFlag:
    def test_only_fully_clear_is_green(self):
        assert t.is_green_flag([1]) is True

    @pytest.mark.parametrize("status", [[1, 2], [2], [1, 4], [4], [4, 5], [1, 2, 6, 7], None, []])
    def test_anything_else_is_not_green(self, status):
        """Yellow included: drivers lift for a local caution, so the lap time
        is not representative even with no safety car."""
        assert t.is_green_flag(status) is False


class TestTimedeltaToMs:
    def test_converts_lap_time(self):
        assert t.td_to_ms(pd.Timedelta(minutes=1, seconds=23, milliseconds=880)) == 83_880

    def test_sub_millisecond_truncates(self):
        assert t.td_to_ms(pd.Timedelta("0 days 00:01:23.880400")) == 83_880

    @pytest.mark.parametrize("value", [pd.NaT, None, float("nan")])
    def test_missing_becomes_none(self, value):
        """LapTime is NaT far more often than you would expect: first laps,
        timing dropouts, laps behind a safety car."""
        assert t.td_to_ms(value) is None

    def test_returns_int_not_float(self):
        assert isinstance(t.td_to_ms(pd.Timedelta(seconds=84)), int)


class TestCoercion:
    @pytest.mark.parametrize("value", [None, float("nan"), pd.NaT])
    def test_bool_of_missing_is_none(self, value):
        """FastF1's `Deleted` column is dtype object holding both None and
        NaN, so bool(x) would wrongly yield True for NaN."""
        assert t.to_bool(value) is None

    def test_bool_of_nan_is_not_true(self):
        assert t.to_bool(float("nan")) is not True

    def test_int_from_float_column(self):
        """LapNumber and Stint arrive as float64."""
        assert t.to_int(21.0) == 21
        assert t.to_int(float("nan")) is None

    def test_str_truncates_and_strips(self):
        assert t.to_str("  VER  ", 5) == "VER"
        assert t.to_str("A" * 300, 10) == "A" * 10
        assert t.to_str("   ") is None


class TestDerivePitStops:
    @staticmethod
    def _laps(rows):
        return pd.DataFrame(rows)

    def test_normal_stop_duration_is_kept(self):
        laps = self._laps([
            {"Driver": "VER", "LapNumber": 20.0, "Compound": "MEDIUM",
             "PitInTime": pd.Timedelta(minutes=30), "PitOutTime": pd.NaT},
            {"Driver": "VER", "LapNumber": 21.0, "Compound": "HARD",
             "PitInTime": pd.NaT, "PitOutTime": pd.Timedelta(minutes=30, seconds=25)},
        ])
        stops = t.derive_pit_stops(laps, session_id=1, driver_ids={"VER": 7})
        assert len(stops) == 1
        assert stops[0]["duration_ms_est"] == 25_000
        assert stops[0]["compound_in"] == "MEDIUM"
        assert stops[0]["compound_out"] == "HARD"
        assert stops[0]["lap_number"] == 20

    def test_red_flag_suspension_is_nulled_not_recorded(self):
        """Observed at Brazil 2024: a red flag fell with cars in the pit lane,
        making PitOut - PitIn about 23 minutes. Recording that as a pit stop
        would corrupt Phase 3's pit-loss estimate."""
        laps = self._laps([
            {"Driver": "LEC", "LapNumber": 32.0, "Compound": "INTERMEDIATE",
             "PitInTime": pd.Timedelta("0 days 02:00:37.713"), "PitOutTime": pd.NaT},
            {"Driver": "LEC", "LapNumber": 33.0, "Compound": "INTERMEDIATE",
             "PitInTime": pd.NaT, "PitOutTime": pd.Timedelta("0 days 02:24:06.937")},
        ])
        stops = t.derive_pit_stops(laps, session_id=1, driver_ids={"LEC": 3})
        assert len(stops) == 1, "the stop is still recorded"
        assert stops[0]["duration_ms_est"] is None, "but its duration is not trusted"

    def test_retirement_in_pits_has_no_out_lap(self):
        laps = self._laps([
            {"Driver": "HAM", "LapNumber": 40.0, "Compound": "HARD",
             "PitInTime": pd.Timedelta(minutes=50), "PitOutTime": pd.NaT},
        ])
        stops = t.derive_pit_stops(laps, session_id=1, driver_ids={"HAM": 4})
        assert len(stops) == 1
        assert stops[0]["duration_ms_est"] is None
        assert stops[0]["compound_out"] is None

    def test_stops_are_numbered_per_driver(self):
        laps = self._laps([
            {"Driver": "VER", "LapNumber": float(n), "Compound": "HARD",
             "PitInTime": pd.Timedelta(minutes=n) if n in (10, 30) else pd.NaT,
             "PitOutTime": pd.Timedelta(minutes=n, seconds=24) if n in (11, 31) else pd.NaT}
            for n in (10, 11, 30, 31)
        ])
        stops = t.derive_pit_stops(laps, session_id=1, driver_ids={"VER": 7})
        assert [s["stop_number"] for s in stops] == [1, 2]

    def test_empty_input(self):
        assert t.derive_pit_stops(pd.DataFrame(), 1, {}) == []


class TestSessionDateTimezone:
    """FastF1's `session.date` is naive but already UTC.

    Verified against the schedule: China 2024 gives Timestamp('2024-04-21
    07:00:00') with tzinfo=None, while the schedule reports Session5Date
    15:00+08:00 and Session5DateUtc 07:00.

    Handing a naive datetime to a TIMESTAMPTZ column makes Postgres interpret
    it in the server's local timezone, so the stored instant ends up wrong by
    the local UTC offset -- silently, and by a different amount depending on
    where the code runs. Phase 4 orders races by time for walk-forward
    backtesting, so this is a correctness bug, not a formatting one.
    """

    def test_naive_timestamp_is_treated_as_utc(self):
        from app.ingestion.fastf1_client import _as_utc

        result = _as_utc(pd.Timestamp("2024-04-21 07:00:00"))
        assert result == dt.datetime(2024, 4, 21, 7, 0, tzinfo=dt.UTC)
        assert result.tzinfo is not None, "must be tz-aware before it reaches Postgres"

    def test_does_not_shift_the_instant(self):
        """The hour must not move. A naive->local conversion would shift it by
        the machine's UTC offset, which is how the original bug presented."""
        from app.ingestion.fastf1_client import _as_utc

        assert _as_utc(pd.Timestamp("2024-09-01 13:00:00")).hour == 13

    def test_already_aware_input_is_converted_not_relabelled(self):
        from app.ingestion.fastf1_client import _as_utc

        aware = pd.Timestamp("2024-04-21 15:00:00", tz="Asia/Shanghai")
        assert _as_utc(aware) == dt.datetime(2024, 4, 21, 7, 0, tzinfo=dt.UTC)

    def test_none_passes_through(self):
        from app.ingestion.fastf1_client import _as_utc

        assert _as_utc(None) is None


class TestMissingValueSentinelStrings:
    """FastF1 sometimes stores the literal string 'None' instead of null.

    Observed in 2023 sessions' `Compound` column. `pd.isna("None")` is False,
    so a naive `str()` passes it through as a real value. It reached Postgres
    and was rejected by the compound CHECK constraint -- which is exactly why
    that constraint exists. Without it, "None" becomes a phantom compound
    category that Phase 2 would silently train on.
    """

    @pytest.mark.parametrize(
        "sentinel", ["None", "none", "NONE", "nan", "NaN", "NaT", "NA", "<NA>", "null"]
    )
    def test_sentinel_strings_become_none(self, sentinel):
        assert t.to_str(sentinel) is None

    @pytest.mark.parametrize("real", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "VER"])
    def test_real_values_are_untouched(self, real):
        assert t.to_str(real) == real

    def test_compound_sentinel_does_not_reach_the_database(self):
        """Regression guard for the exact failure: a lap row whose Compound is
        the string 'None' must produce SQL NULL, not a CHECK violation."""
        laps = pd.DataFrame([{
            "Driver": "ALB", "Team": "Williams", "LapNumber": 23.0,
            "LapTime": pd.Timedelta(milliseconds=98_336),
            "Compound": "None",  # the literal string, as FastF1 supplies it
            "TyreLife": 5.0, "FreshTyre": False, "Stint": 6.0,
            "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "1",
            "Position": 10.0, "IsAccurate": True, "Deleted": None,
        }])
        rows = t.transform_laps(laps, 1, {"ALB": 9}, {"Williams": 1})
        assert len(rows) == 1
        assert rows[0]["compound"] is None
        assert rows[0]["compound"] != "None"
