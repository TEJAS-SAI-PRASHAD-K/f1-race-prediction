"""Loader tests: idempotency and upsert semantics.

These matter because FastF1 backfills and corrects session data for a while
after a race, so ingestion is re-run routinely rather than exceptionally.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import func, select

from app.db.models import Driver, Lap, Season, Team
from app.ingestion.fastf1_client import SessionData
from app.ingestion.loader import ingest_session_data, upsert_drivers


def _session_data(lap_time_ms: int = 84_000, n_laps: int = 5) -> SessionData:
    """A tiny but structurally real session payload."""
    event = pd.Series({
        "RoundNumber": 16,
        "Country": "Italy",
        "Location": "Monza",
        "EventName": "Italian Grand Prix",
        "OfficialEventName": "FORMULA 1 GRAND PRIX 2024",
        "EventDate": pd.Timestamp("2024-09-01"),
        "EventFormat": "conventional",
    })

    laps = pd.DataFrame([
        {
            "Driver": "VER", "DriverNumber": "1", "Team": "Red Bull Racing",
            "LapNumber": float(n), "LapTime": pd.Timedelta(milliseconds=lap_time_ms + n * 100),
            "Sector1Time": pd.Timedelta(seconds=28), "Sector2Time": pd.Timedelta(seconds=29),
            "Sector3Time": pd.Timedelta(seconds=27),
            "Compound": "HARD", "TyreLife": float(n), "FreshTyre": True, "Stint": 1.0,
            "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "1",
            "Position": 1.0, "SpeedI1": 300.0, "SpeedI2": 310.0,
            "SpeedFL": 320.0, "SpeedST": 330.0,
            "IsAccurate": True, "Deleted": None, "DeletedReason": None,
        }
        for n in range(1, n_laps + 1)
    ])

    results = pd.DataFrame([{
        "Abbreviation": "VER", "DriverNumber": "1", "TeamName": "Red Bull Racing",
        "FullName": "Max Verstappen", "CountryCode": "NED",
        "Position": 1.0, "GridPosition": 1.0, "Status": "Finished",
        "Points": 25.0, "Time": pd.Timedelta(minutes=74),
    }])

    weather = pd.DataFrame([{
        "Time": pd.Timedelta(minutes=m), "AirTemp": 25.0, "TrackTemp": 40.0,
        "Humidity": 50.0, "Pressure": 1010.0, "Rainfall": False,
        "WindSpeed": 2.0, "WindDirection": 180,
    } for m in range(3)])

    return SessionData(
        year=2024, round_number=16, session_type="R", session_name="Race",
        session_date=dt.datetime(2024, 9, 1, 13, 0, tzinfo=dt.UTC),
        total_laps=53, event=event, laps=laps, results=results, weather=weather,
    )


class TestIdempotency:
    def test_ingesting_twice_does_not_duplicate_rows(self, db):
        data = _session_data()

        first = ingest_session_data(db, data)
        db.flush()
        count_after_first = db.scalar(
            select(func.count()).select_from(Lap).where(Lap.session_id == first["session_id"])
        )

        second = ingest_session_data(db, data)
        db.flush()
        count_after_second = db.scalar(
            select(func.count()).select_from(Lap).where(Lap.session_id == second["session_id"])
        )

        assert count_after_first == count_after_second == 5
        assert first["session_id"] == second["session_id"], "same session reused"

    def test_no_duplicate_natural_keys(self, db):
        data = _session_data()
        ingest_session_data(db, data)
        ingest_session_data(db, data)
        db.flush()

        dupes = db.execute(
            select(Lap.session_id, Lap.driver_id, Lap.lap_number, func.count())
            .group_by(Lap.session_id, Lap.driver_id, Lap.lap_number)
            .having(func.count() > 1)
        ).all()
        assert dupes == []

    def test_re_ingest_updates_corrected_values(self, db):
        """FastF1 revises timing data after a session. A re-run must pick up
        the correction rather than keep the stale value or insert a second row.
        """
        ingest_session_data(db, _session_data(lap_time_ms=84_000))
        db.flush()

        result = ingest_session_data(db, _session_data(lap_time_ms=83_000))
        db.flush()

        times = db.scalars(
            select(Lap.lap_time_ms)
            .where(Lap.session_id == result["session_id"])
            .order_by(Lap.lap_number)
        ).all()
        assert times == [83_100, 83_200, 83_300, 83_400, 83_500]

    def test_fewer_laps_on_re_ingest_leaves_old_rows(self, db):
        """Documents real behaviour: upsert never deletes. If a later fetch
        returns fewer laps, the earlier rows remain. Acceptable here because
        FastF1 corrects data rather than removing laps, but worth knowing."""
        ingest_session_data(db, _session_data(n_laps=5))
        db.flush()
        result = ingest_session_data(db, _session_data(n_laps=3))
        db.flush()
        assert db.scalar(
            select(func.count()).select_from(Lap).where(Lap.session_id == result["session_id"])
        ) == 5


class TestDriverUpsert:
    def test_resolves_abbreviations_to_ids(self, db):
        db.add(Season(year=2024))
        db.flush()
        ids = upsert_drivers(db, _session_data())
        assert set(ids) == {"VER"}

    def test_sparse_later_data_does_not_blank_known_fields(self, db):
        """COALESCE guard: a session whose results lack a full name must not
        erase the name learned from an earlier session."""
        db.add(Season(year=2024))
        db.flush()
        upsert_drivers(db, _session_data())
        db.flush()

        sparse = _session_data()
        sparse.results = pd.DataFrame([{
            "Abbreviation": "VER", "DriverNumber": None, "TeamName": "Red Bull Racing",
            "FullName": None, "CountryCode": None, "Position": 1.0,
            "GridPosition": 1.0, "Status": "Finished", "Points": 25.0,
            "Time": pd.Timedelta(minutes=74),
        }])
        upsert_drivers(db, sparse)
        db.flush()

        driver = db.scalars(select(Driver).where(Driver.abbreviation == "VER")).one()
        assert driver.full_name == "Max Verstappen"
        assert driver.country_code == "NED"
        assert driver.driver_number == 1


class TestTeamUpsert:
    def test_team_is_scoped_to_season(self, db):
        """Team names change between seasons, so the natural key includes the
        year. The same name in two seasons is two rows."""
        ingest_session_data(db, _session_data())
        data_2023 = _session_data()
        data_2023.year = 2023
        ingest_session_data(db, data_2023)
        db.flush()

        teams = db.scalars(
            select(Team).where(Team.name == "Red Bull Racing").order_by(Team.season_year)
        ).all()
        assert [t.season_year for t in teams] == [2023, 2024]


class TestTrackStatusPersistence:
    @pytest.mark.parametrize(
        "raw,expected", [("1", [1]), ("14", [1, 4]), ("1267", [1, 2, 6, 7]), ("21", [1, 2])]
    )
    def test_arrays_survive_the_round_trip(self, db, raw, expected):
        """Proves the smallint[] column stores a real array, canonically
        ordered -- not a string and not an integer."""
        data = _session_data()
        data.laps["TrackStatus"] = raw
        result = ingest_session_data(db, data)
        db.flush()

        stored = db.scalars(
            select(Lap.track_status).where(Lap.session_id == result["session_id"]).limit(1)
        ).one()
        assert stored == expected
