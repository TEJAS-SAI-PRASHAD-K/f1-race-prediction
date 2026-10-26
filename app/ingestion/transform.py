"""Convert FastF1 DataFrames into plain dicts ready for bulk insert.

This module is where every data gotcha gets handled, exactly once, at the
boundary. Nothing downstream should ever have to re-parse FastF1's quirks.

Verified against real data (2024 Monza race, 2024 Brazil race) rather than
assumed:

  * `TrackStatus` really does contain multi-flag strings. Brazil 2024 yields
    '12', '45', '124', '671', '1267'. Cast '1267' to an int and you get one
    thousand two hundred sixty-seven. As a flag array it is {1,2,6,7}.
  * `Deleted` has dtype object holding both None and NaN.
  * `LapNumber` and `Stint` arrive as float64 and need careful int coercion.
  * `results.Time` is CUMULATIVE race time for P1 but a GAP TO LEADER for
    everyone else. We store it verbatim; any consumer must know this.
  * `PitOutTime - PitInTime` is NOT stationary time. At Brazil 2024 a red
    flag made that difference 23 minutes. See derive_pit_stops().
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# Track status single-digit flag meanings, per FastF1's API docs.
TRACK_STATUS_MEANINGS = {
    1: "clear",
    2: "yellow",
    3: "unknown",
    4: "safety_car",
    5: "red_flag",
    6: "vsc_deployed",
    7: "vsc_ending",
}

# Flags that definitively compromise a lap: safety car, red flag, VSC.
NON_GREEN_FLAGS = (4, 5, 6, 7)

# Plausible bounds for (pit exit - pit entry) in a real pit stop: pit lane
# transit at the speed limit plus a stationary stop, so roughly 20-30s, with
# headroom for slow stops and served penalties. Anything outside this is a
# red-flag suspension or a car stopping in the pits, not a pit stop, so we
# record NULL rather than a number that would quietly corrupt pit-loss stats.
PIT_DURATION_MIN_MS = 8_000
PIT_DURATION_MAX_MS = 180_000


# --------------------------------------------------------------------------
# scalar coercion
# --------------------------------------------------------------------------


def _isna(value: Any) -> bool:
    """NaN/NaT/None check that is safe for non-scalars."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return bool(getattr(result, "all", lambda: False)())


def td_to_ms(value: Any) -> int | None:
    """pandas.Timedelta -> integer milliseconds.

    Integer ms is exact, sorts correctly, and Postgres rejects a wrong type
    instead of silently coercing it. Never store a Timedelta or float seconds.
    """
    if _isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return int(value.total_seconds() * 1000)
    if isinstance(value, int | float):
        return int(float(value) * 1000)
    try:
        return int(pd.Timedelta(value).total_seconds() * 1000)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if _isna(value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f)


def to_float(value: Any) -> float | None:
    if _isna(value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def to_bool(value: Any) -> bool | None:
    """`Deleted` is dtype object holding None and NaN, so this needs care."""
    if _isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "1", "yes"}:
            return True
        if v in {"false", "f", "0", "no", ""}:
            return False
        return None
    try:
        return bool(value)
    except (TypeError, ValueError):
        return None


# Strings that pandas/FastF1 produce when a value is really missing. Observed
# in the wild: FastF1 puts the literal string "None" in `Compound` for some
# 2023 sessions, which is NOT Python None -- pd.isna("None") is False, so a
# naive str() passes it straight through. That reached Postgres and was
# rejected by the compound CHECK constraint, which is precisely why the
# constraint is there. Without it, "None" would have become a phantom compound
# category downstream and nothing would have complained.
_MISSING_STRINGS = {"none", "nan", "nat", "na", "<na>", "null", "nonetype"}


def to_str(value: Any, max_len: int | None = None) -> str | None:
    """Coerce to a clean string, mapping missing-value sentinels to None."""
    if _isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in _MISSING_STRINGS:
        return None
    return s[:max_len] if max_len else s


# --------------------------------------------------------------------------
# the TrackStatus gotcha
# --------------------------------------------------------------------------


def parse_track_status(value: Any) -> list[int] | None:
    """'1267' -> [1, 2, 6, 7]. The single most important parse in the pipeline.

    FastF1 concatenates every status flag that applied during the lap into one
    string. '14' means the lap was partly clear and partly under safety car.
    Treating that as the number 14 is a silent, destructive bug: it produces a
    plausible-looking integer that poisons every model trained on it.

    The result is SORTED and de-duplicated, which makes it canonical. This
    matters: FastF1 emits both '12' and '21' for the same pair of flags, and
    without sorting they would be stored as the distinct arrays {1,2} and
    {2,1}. Equality comparisons and GROUP BY would then split one real
    condition across two buckets. Order of appearance carries no information
    we need -- the flag is which conditions occurred, not their sequence -- so
    sorting loses nothing and makes the column comparable.

    Returns None for missing input, so "unknown" stays distinguishable from
    "clear" ([1]).
    """
    if _isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    flags = {int(ch) for ch in raw if ch.isdigit()}
    return sorted(flags) or None


def is_green_flag(track_status: list[int] | None) -> bool:
    """True only if the lap ran entirely under clear conditions, i.e. [1].

    This is deliberately strict. A yellow flag ([1, 2]) means drivers lifted
    for a local caution, so the lap time is not representative even though no
    safety car was deployed. Standard practice for F1 pace analysis is to keep
    only fully-clear laps, and for tire-degradation modeling a contaminated lap
    is worse than a missing one.

    Unknown status (None) counts as not-green: better to discard a lap than to
    train on one we cannot vouch for.

    The equivalent SQL predicate is `track_status = ARRAY[1]::smallint[]`. Use
    the looser `NOT (track_status && ARRAY[4,5,6,7])` only if you specifically
    want to keep yellow-flag laps.
    """
    return track_status == [1]


# --------------------------------------------------------------------------
# row builders
# --------------------------------------------------------------------------


def transform_event(event: pd.Series, year: int) -> dict[str, Any]:
    return {
        "season_year": year,
        "round": to_int(event.get("RoundNumber")),
        "event_name": to_str(event.get("EventName"), 128) or f"Round {event.get('RoundNumber')}",
        "official_name": to_str(event.get("OfficialEventName"), 256),
        "event_date": (
            None if _isna(event.get("EventDate")) else pd.Timestamp(event["EventDate"]).date()
        ),
        "event_format": to_str(event.get("EventFormat"), 32),
    }


# FastF1 reports `Location` inconsistently across seasons, producing two rows
# for one physical track: "Monte Carlo" in 2019/2021 but "Monaco" from 2022,
# "Singapore" in 2019 but "Marina Bay" from 2023. Left alone that fragments
# every per-circuit statistic -- Monaco's pit loss split into 34 and 28 stops,
# and the model saw two unrelated categorical levels for one track.
#
# Only names for the SAME physical circuit belong here. Hockenheim/Nurburgring,
# Monza/Imola/Mugello and Austin/Miami/Las Vegas share a country but are
# genuinely different tracks.
CIRCUIT_NAME_ALIASES = {
    "Monte Carlo": "Monaco",
    "Monte-Carlo": "Monaco",
    "Singapore": "Marina Bay",
}


def canonical_circuit_name(name: str | None) -> str | None:
    """Collapse FastF1's naming variants onto one name per physical circuit."""
    if name is None:
        return None
    return CIRCUIT_NAME_ALIASES.get(name, name)


def transform_circuit(event: pd.Series) -> dict[str, Any]:
    """Circuit identity comes from the event's Location/Country.

    The schedule carries no latitude/longitude, so those stay NULL. None of
    the four features need coordinates.

    `name` is canonicalised (see CIRCUIT_NAME_ALIASES) because it is the
    UNIQUE key, so a naming variant would create a second row for one track.
    `locality` keeps whatever FastF1 actually said, so the raw value is not lost.
    """
    location = to_str(event.get("Location"), 128)
    return {
        "name": canonical_circuit_name(location) or "Unknown",
        "country": to_str(event.get("Country"), 64),
        "locality": location[:64] if location else None,
    }


def transform_laps(
    laps: pd.DataFrame,
    session_id: int,
    driver_ids: dict[str, int],
    team_ids: dict[str, int],
) -> list[dict[str, Any]]:
    """One dict per lap, keyed to already-resolved driver/team ids."""
    rows: list[dict[str, Any]] = []
    for lap in laps.itertuples(index=False):
        abbr = to_str(getattr(lap, "Driver", None), 5)
        driver_id = driver_ids.get(abbr) if abbr else None
        lap_number = to_int(getattr(lap, "LapNumber", None))
        if driver_id is None or lap_number is None:
            # Without a driver and lap number the row has no natural key and
            # cannot be upserted, so it is unusable.
            continue

        team_name = to_str(getattr(lap, "Team", None), 128)
        rows.append(
            {
                "session_id": session_id,
                "driver_id": driver_id,
                "team_id": team_ids.get(team_name) if team_name else None,
                "lap_number": lap_number,
                "lap_time_ms": td_to_ms(getattr(lap, "LapTime", None)),
                "sector1_ms": td_to_ms(getattr(lap, "Sector1Time", None)),
                "sector2_ms": td_to_ms(getattr(lap, "Sector2Time", None)),
                "sector3_ms": td_to_ms(getattr(lap, "Sector3Time", None)),
                "compound": to_str(getattr(lap, "Compound", None), 16),
                "tyre_life": to_float(getattr(lap, "TyreLife", None)),
                "is_fresh_tyre": to_bool(getattr(lap, "FreshTyre", None)),
                "stint_number": to_int(getattr(lap, "Stint", None)),
                "pit_in_ms": td_to_ms(getattr(lap, "PitInTime", None)),
                "pit_out_ms": td_to_ms(getattr(lap, "PitOutTime", None)),
                "track_status": parse_track_status(getattr(lap, "TrackStatus", None)),
                "position": to_float(getattr(lap, "Position", None)),
                "speed_i1": to_float(getattr(lap, "SpeedI1", None)),
                "speed_i2": to_float(getattr(lap, "SpeedI2", None)),
                "speed_fl": to_float(getattr(lap, "SpeedFL", None)),
                "speed_st": to_float(getattr(lap, "SpeedST", None)),
                "is_accurate": to_bool(getattr(lap, "IsAccurate", None)),
                "is_deleted": to_bool(getattr(lap, "Deleted", None)),
                "deleted_reason": to_str(getattr(lap, "DeletedReason", None), 128),
            }
        )
    return rows


def transform_results(
    results: pd.DataFrame,
    session_id: int,
    driver_ids: dict[str, int],
    team_ids: dict[str, int],
) -> list[dict[str, Any]]:
    """Session classification.

    Note on `time_ms`: FastF1 reports cumulative race time for the winner but
    gap-to-leader for everyone else. We store it as given; consumers must
    account for that. The feature view uses `position`, not `time_ms`, for exactly
    this reason.
    """
    rows: list[dict[str, Any]] = []
    for res in results.itertuples(index=False):
        abbr = to_str(getattr(res, "Abbreviation", None), 5)
        driver_id = driver_ids.get(abbr) if abbr else None
        if driver_id is None:
            continue
        team_name = to_str(getattr(res, "TeamName", None), 128)
        rows.append(
            {
                "session_id": session_id,
                "driver_id": driver_id,
                "team_id": team_ids.get(team_name) if team_name else None,
                "position": to_float(getattr(res, "Position", None)),
                "grid_position": to_float(getattr(res, "GridPosition", None)),
                "status": to_str(getattr(res, "Status", None), 64),
                "points": to_float(getattr(res, "Points", None)),
                "time_ms": td_to_ms(getattr(res, "Time", None)),
            }
        )
    return rows


def transform_weather(weather: pd.DataFrame, session_id: int) -> list[dict[str, Any]]:
    if weather is None or weather.empty:
        return []
    rows: list[dict[str, Any]] = []
    for w in weather.itertuples(index=False):
        offset = td_to_ms(getattr(w, "Time", None))
        if offset is None:
            continue
        rows.append(
            {
                "session_id": session_id,
                "time_offset_ms": offset,
                "air_temp": to_float(getattr(w, "AirTemp", None)),
                "track_temp": to_float(getattr(w, "TrackTemp", None)),
                "humidity": to_float(getattr(w, "Humidity", None)),
                "pressure": to_float(getattr(w, "Pressure", None)),
                "rainfall": to_bool(getattr(w, "Rainfall", None)),
                "wind_speed": to_float(getattr(w, "WindSpeed", None)),
                "wind_direction": to_int(getattr(w, "WindDirection", None)),
            }
        )
    return rows


def derive_pit_stops(
    laps: pd.DataFrame, session_id: int, driver_ids: dict[str, int]
) -> list[dict[str, Any]]:
    """Reconstruct pit stops from in-lap / out-lap timestamps.

    FastF1 gives no pit stop duration. `PitInTime` is set on the in-lap and
    `PitOutTime` on the following out-lap, both as session-time offsets.

    The naive difference is NOT stationary time, and is sometimes not even a
    pit stop: at Brazil 2024 a red flag fell while cars were in the pit lane,
    making PitOut - PitIn about 23 minutes. We therefore store the difference
    only when it falls in a plausible range and NULL otherwise, so downstream
    never silently averages a race suspension into its pit-loss estimate.

    Even when plausible, this is pit-lane transit plus stationary time, not
    stationary time alone.
    """
    if laps is None or laps.empty:
        return []

    rows: list[dict[str, Any]] = []
    for abbr, group in laps.groupby("Driver", sort=False):
        driver_id = driver_ids.get(to_str(abbr, 5) or "")
        if driver_id is None:
            continue

        g = group.sort_values("LapNumber")
        records = list(g.itertuples(index=False))
        stop_number = 0

        for idx, lap in enumerate(records):
            pit_in = td_to_ms(getattr(lap, "PitInTime", None))
            if pit_in is None:
                continue

            stop_number += 1
            out_lap = records[idx + 1] if idx + 1 < len(records) else None
            pit_out = td_to_ms(getattr(out_lap, "PitOutTime", None)) if out_lap else None

            duration = None
            if pit_out is not None and pit_out > pit_in:
                delta = pit_out - pit_in
                if PIT_DURATION_MIN_MS <= delta <= PIT_DURATION_MAX_MS:
                    duration = delta

            rows.append(
                {
                    "session_id": session_id,
                    "driver_id": driver_id,
                    "stop_number": stop_number,
                    "lap_number": to_int(getattr(lap, "LapNumber", None)),
                    "duration_ms_est": duration,
                    "compound_in": to_str(getattr(lap, "Compound", None), 16),
                    "compound_out": (
                        to_str(getattr(out_lap, "Compound", None), 16) if out_lap else None
                    ),
                }
            )
    return rows
