"""SQLAlchemy ORM models for the `raw` schema.

Design decisions worth understanding:

1. All times are stored as nullable BIGINT **milliseconds**, never as floats or
   text. FastF1 hands you `pandas.Timedelta` values that are `NaT` surprisingly
   often (first laps, timing dropouts). Integer ms is exact, sorts correctly,
   and Postgres rejects a wrong type rather than silently coercing it.

2. `laps.track_status` is a SMALLINT[] array, not a string. FastF1 gives a
   concatenated digit string where '14' means "clear AND safety car during this
   lap". Casting that to an int is a silent, destructive bug. We parse it once
   at the ingest boundary so every downstream query is unambiguous.

3. CHECK constraints instead of native Postgres ENUM types. Same validation,
   but altering a CHECK in Alembic is far less painful than altering an enum.

4. Every table has a natural UNIQUE key so ingestion can use
   INSERT ... ON CONFLICT DO UPDATE and therefore be safely re-runnable.
"""

from datetime import date, datetime

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import SCHEMA_ML, SCHEMA_RAW, Base

# FastF1's documented compound values. It does NOT distinguish C1-C5: "SOFT"
# is only relative to that weekend's allocation, so a Monaco soft and a
# Silverstone soft are physically different rubber. Any model using compound
# must interact it with circuit, never use it alone.
COMPOUNDS = (
    "SOFT",
    "MEDIUM",
    "HARD",
    "INTERMEDIATE",
    "WET",
    "TEST_UNKNOWN",
    "UNKNOWN",
)

SESSION_TYPES = ("FP1", "FP2", "FP3", "Q", "SQ", "S", "R")


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = {"schema": SCHEMA_RAW}

    year: Mapped[int] = mapped_column(Integer, primary_key=True)

    events: Mapped[list["Event"]] = relationship(back_populates="season")


class Circuit(Base):
    __tablename__ = "circuits"
    __table_args__ = (
        UniqueConstraint("name", name="uq_circuits_name"),
        {"schema": SCHEMA_RAW},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    country: Mapped[str | None] = mapped_column(String(64))
    locality: Mapped[str | None] = mapped_column(String(64))
    latitude: Mapped[float | None] = mapped_column(Numeric(9, 6))
    longitude: Mapped[float | None] = mapped_column(Numeric(9, 6))

    # Circuit-level reference values. `pit_loss_seconds` is the time lost
    # by pitting versus staying out -- typically 18-25s. We seed it from
    # observed pit stops once enough data is ingested; NULL means "unknown,
    # fall back to a default".
    typical_race_laps: Mapped[int | None] = mapped_column(SmallInteger)
    pit_loss_seconds: Mapped[float | None] = mapped_column(Numeric(5, 2))

    events: Mapped[list["Event"]] = relationship(back_populates="circuit")


class Event(Base):
    """A race weekend."""

    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("season_year", "round", name="uq_events_season_year"),
        {"schema": SCHEMA_RAW},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season_year: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.seasons.year", ondelete="CASCADE"), nullable=False
    )
    round: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    event_name: Mapped[str] = mapped_column(String(128), nullable=False)
    official_name: Mapped[str | None] = mapped_column(String(256))
    circuit_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.circuits.id", ondelete="SET NULL")
    )
    event_date: Mapped[date | None] = mapped_column(Date)
    # 'conventional' | 'sprint' | 'sprint_shootout' | 'sprint_qualifying' | 'testing'
    event_format: Mapped[str | None] = mapped_column(String(32))

    season: Mapped["Season"] = relationship(back_populates="events")
    circuit: Mapped["Circuit | None"] = relationship(back_populates="events")
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class Session(Base):
    """One session within a weekend: FP1-3, Q, SQ, S (sprint), R (race)."""

    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint("event_id", "session_type", name="uq_sessions_event_id"),
        CheckConstraint(
            "session_type IN " + str(SESSION_TYPES), name="session_type_valid"
        ),
        {"schema": SCHEMA_RAW},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.events.id", ondelete="CASCADE"), nullable=False
    )
    session_type: Mapped[str] = mapped_column(String(8), nullable=False)
    session_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_laps: Mapped[int | None] = mapped_column(SmallInteger)

    event: Mapped["Event"] = relationship(back_populates="sessions")
    laps: Mapped[list["Lap"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    results: Mapped[list["SessionResult"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Driver(Base):
    __tablename__ = "drivers"
    __table_args__ = (
        UniqueConstraint("abbreviation", name="uq_drivers_abbreviation"),
        {"schema": SCHEMA_RAW},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Three-letter code: VER, HAM, NOR. Stable across seasons, so it makes a
    # good natural key.
    abbreviation: Mapped[str] = mapped_column(String(5), nullable=False)
    driver_number: Mapped[int | None] = mapped_column(SmallInteger)
    full_name: Mapped[str | None] = mapped_column(String(128))
    country_code: Mapped[str | None] = mapped_column(String(8))


class Team(Base):
    """Team names change between seasons, so the natural key includes the year."""

    __tablename__ = "teams"
    __table_args__ = (
        UniqueConstraint("name", "season_year", name="uq_teams_name"),
        {"schema": SCHEMA_RAW},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    season_year: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.seasons.year", ondelete="CASCADE"), nullable=False
    )


class SessionResult(Base):
    __tablename__ = "session_results"
    __table_args__ = {"schema": SCHEMA_RAW}

    session_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.sessions.id", ondelete="CASCADE"), primary_key=True
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.drivers.id", ondelete="CASCADE"), primary_key=True
    )
    team_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.teams.id", ondelete="SET NULL")
    )
    position: Mapped[float | None] = mapped_column(Numeric(4, 1))
    grid_position: Mapped[float | None] = mapped_column(Numeric(4, 1))
    status: Mapped[str | None] = mapped_column(String(64))
    points: Mapped[float | None] = mapped_column(Numeric(5, 2))
    time_ms: Mapped[int | None] = mapped_column(BigInteger)

    session: Mapped["Session"] = relationship(back_populates="results")
    driver: Mapped["Driver"] = relationship()
    team: Mapped["Team | None"] = relationship()


class Lap(Base):
    """One timed lap. The core modeling table."""

    __tablename__ = "laps"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "driver_id", "lap_number", name="uq_laps_session_id"
        ),
        CheckConstraint(
            "compound IS NULL OR compound IN " + str(COMPOUNDS),
            name="compound_valid",
        ),
        Index("ix_laps_session_driver_lap", "session_id", "driver_id", "lap_number"),
        # Partial index: pace queries filter on is_accurate every time, and
        # skipping inaccurate laps keeps this index small.
        Index(
            "ix_laps_compound_tyrelife_accurate",
            "compound",
            "tyre_life",
            postgresql_where="is_accurate",
        ),
        {"schema": SCHEMA_RAW},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.sessions.id", ondelete="CASCADE"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.drivers.id", ondelete="CASCADE"), nullable=False
    )
    team_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.teams.id", ondelete="SET NULL")
    )

    lap_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    lap_time_ms: Mapped[int | None] = mapped_column(BigInteger)
    sector1_ms: Mapped[int | None] = mapped_column(BigInteger)
    sector2_ms: Mapped[int | None] = mapped_column(BigInteger)
    sector3_ms: Mapped[int | None] = mapped_column(BigInteger)

    # --- tire state ---
    compound: Mapped[str | None] = mapped_column(String(16))
    # Laps on this tire SET, including laps from earlier sessions. A "new"
    # race tire that did a qualifying run is not at zero.
    tyre_life: Mapped[float | None] = mapped_column(Numeric(5, 1))
    # True only if the set was genuinely new at stint start (TyreLife == 0).
    is_fresh_tyre: Mapped[bool | None] = mapped_column(Boolean)
    stint_number: Mapped[int | None] = mapped_column(SmallInteger)

    # --- pit lane: session-time offsets, on DIFFERENT laps ---
    # pit_in_ms is set on the in-lap, pit_out_ms on the following out-lap.
    # Stationary time is derived from these, not given directly.
    pit_in_ms: Mapped[int | None] = mapped_column(BigInteger)
    pit_out_ms: Mapped[int | None] = mapped_column(BigInteger)

    # Parsed flag set, not a string. 1=clear 2=yellow 4=SC 5=red 6=VSC 7=VSC-end
    # Query green-flag laps with: NOT (track_status && ARRAY[4,5,6,7])
    track_status: Mapped[list[int] | None] = mapped_column(ARRAY(SmallInteger))

    position: Mapped[float | None] = mapped_column(Numeric(4, 1))
    speed_i1: Mapped[float | None] = mapped_column(Numeric(6, 2))
    speed_i2: Mapped[float | None] = mapped_column(Numeric(6, 2))
    speed_fl: Mapped[float | None] = mapped_column(Numeric(6, 2))
    speed_st: Mapped[float | None] = mapped_column(Numeric(6, 2))

    # FastF1's own "this lap is clean enough to compare" flag. It excludes
    # in-laps, out-laps, and laps with timing problems. Filtering on this
    # removes most of the garbage in one step.
    is_accurate: Mapped[bool | None] = mapped_column(Boolean)
    is_deleted: Mapped[bool | None] = mapped_column(Boolean)
    deleted_reason: Mapped[str | None] = mapped_column(String(128))

    session: Mapped["Session"] = relationship(back_populates="laps")
    driver: Mapped["Driver"] = relationship()


class Weather(Base):
    """Per-session weather samples, roughly one per minute."""

    __tablename__ = "weather"
    __table_args__ = (
        UniqueConstraint("session_id", "time_offset_ms", name="uq_weather_session_id"),
        {"schema": SCHEMA_RAW},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.sessions.id", ondelete="CASCADE"), nullable=False
    )
    time_offset_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    air_temp: Mapped[float | None] = mapped_column(Numeric(5, 2))
    track_temp: Mapped[float | None] = mapped_column(Numeric(5, 2))
    humidity: Mapped[float | None] = mapped_column(Numeric(5, 2))
    pressure: Mapped[float | None] = mapped_column(Numeric(7, 2))
    rainfall: Mapped[bool | None] = mapped_column(Boolean)
    wind_speed: Mapped[float | None] = mapped_column(Numeric(5, 2))
    wind_direction: Mapped[int | None] = mapped_column(SmallInteger)


class PitStop(Base):
    """Derived, not ingested directly -- see transform.derive_pit_stops().

    `duration_ms_est` is an ESTIMATE. FastF1 gives pit entry/exit timestamps,
    not stationary time, so this includes pit lane transit.
    """

    __tablename__ = "pit_stops"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "driver_id", "stop_number", name="uq_pit_stops_session_id"
        ),
        {"schema": SCHEMA_RAW},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.sessions.id", ondelete="CASCADE"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_RAW}.drivers.id", ondelete="CASCADE"), nullable=False
    )
    stop_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    lap_number: Mapped[int | None] = mapped_column(SmallInteger)
    duration_ms_est: Mapped[int | None] = mapped_column(BigInteger)
    compound_in: Mapped[str | None] = mapped_column(String(16))
    compound_out: Mapped[str | None] = mapped_column(String(16))


class ModelVersion(Base):
    """Model registry.

    `backend` matters: XGBoost models with categorical features MUST be saved
    as JSON via save_model(), because joblib/pickle silently loses the category
    information and the model then mispredicts without erroring. sklearn models
    use joblib. The registry therefore dispatches on this column.
    """

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name"),
        CheckConstraint(
            "backend IN ('xgboost', 'sklearn')", name="backend_valid"
        ),
        {"schema": SCHEMA_ML},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    backend: Mapped[str] = mapped_column(String(16), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    # Pinned category lists live here. XGBoost treats unseen categories as a
    # hard error, so prediction must map anything unseen to NaN.
    categories_path: Mapped[str | None] = mapped_column(String(512))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    train_season_min: Mapped[int | None] = mapped_column(SmallInteger)
    train_season_max: Mapped[int | None] = mapped_column(SmallInteger)
    n_train_rows: Mapped[int | None] = mapped_column(Integer)
    metrics_json: Mapped[str | None] = mapped_column(String(4096))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str | None] = mapped_column(String(1024))
