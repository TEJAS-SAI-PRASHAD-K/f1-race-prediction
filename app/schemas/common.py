"""Pydantic response models.

Times are exposed as both integer milliseconds (exact, what is stored) and
seconds (convenient for humans and plots). Clients should compute with ms.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, computed_field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SeasonOut(ORMModel):
    year: int


class CircuitOut(ORMModel):
    id: int
    name: str
    country: str | None = None
    locality: str | None = None
    typical_race_laps: int | None = None
    pit_loss_seconds: float | None = None


class EventOut(ORMModel):
    id: int
    season_year: int
    round: int
    event_name: str
    official_name: str | None = None
    event_date: date | None = None
    event_format: str | None = None
    circuit: CircuitOut | None = None


class SessionOut(ORMModel):
    id: int
    event_id: int
    session_type: str
    session_date: datetime | None = None
    total_laps: int | None = None


class DriverOut(ORMModel):
    id: int
    abbreviation: str
    driver_number: int | None = None
    full_name: str | None = None
    country_code: str | None = None


class LapOut(ORMModel):
    lap_number: int
    driver: str
    team: str | None = None
    lap_time_ms: int | None = None
    sector1_ms: int | None = None
    sector2_ms: int | None = None
    sector3_ms: int | None = None
    compound: str | None = None
    tyre_life: float | None = None
    is_fresh_tyre: bool | None = None
    stint_number: int | None = None
    pit_in_ms: int | None = None
    pit_out_ms: int | None = None
    track_status: list[int] | None = None
    position: float | None = None
    is_accurate: bool | None = None

    @computed_field
    @property
    def lap_time_s(self) -> float | None:
        return round(self.lap_time_ms / 1000.0, 3) if self.lap_time_ms is not None else None

    @computed_field
    @property
    def is_green_flag(self) -> bool:
        """Strictly clear conditions. A yellow-flag lap is not green --
        drivers lifted, so the lap time is not representative."""
        return self.track_status == [1]


class ResultOut(ORMModel):
    driver: str
    full_name: str | None = None
    team: str | None = None
    position: float | None = None
    grid_position: float | None = None
    status: str | None = None
    points: float | None = None
    # Cumulative race time for P1, gap-to-leader for everyone else.
    time_ms: int | None = None


class StintOut(BaseModel):
    """One driver's run on one set of tires, aggregated from laps."""

    driver: str
    stint_number: int
    compound: str | None = None
    lap_start: int
    lap_end: int
    laps_run: int
    tyre_life_start: float | None = None
    tyre_life_end: float | None = None
    is_fresh_tyre: bool | None = None
    # Median over green-flag accurate laps only -- the honest pace measure.
    median_green_lap_time_ms: int | None = None
    green_laps: int


class PitStopOut(ORMModel):
    driver: str
    stop_number: int
    lap_number: int | None = None
    duration_ms_est: int | None = None
    compound_in: str | None = None
    compound_out: str | None = None


class HealthOut(BaseModel):
    status: str
    database: str
    laps_ingested: int | None = None
