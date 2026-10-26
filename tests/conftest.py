"""Test fixtures.

Tests run against a real Postgres database (`f1_race_prediction_test`), not
SQLite. That is deliberate: the tests exercise the same engine, the same array
types and the same ON CONFLICT semantics as production, so a test passing here
means something.

THE TEST DATABASE IS BUILT BY THE MIGRATIONS
--------------------------------------------
Not by `Base.metadata.create_all`. The difference matters: `create_all` knows
only about ORM tables, so it cannot create `features.stint_laps` or
`features.race_features`, and anything depending on those views would have to
read the developer's real database instead. Running the migrations means the
test database is the same shape as production, and the migrations themselves
get exercised on every run.

Schemas are dropped with CASCADE rather than `alembic downgrade base`, because
CASCADE also removes `ml.alembic_version`. Without that, a database left
stamped at an old revision makes `upgrade head` a silent no-op against tables
that were just dropped -- which fails later, confusingly, and far from here.
"""

from __future__ import annotations

import datetime as dt

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from alembic import command
from app.config import PROJECT_ROOT, get_settings  # noqa: I001
from app.db.base import SCHEMA_FEATURES, SCHEMA_ML, SCHEMA_RAW
from app.db.deps import get_db, get_engine
from app.db.models import Driver, Event, Lap, Season, Session, SessionResult, Team
from app.main import app
from tests.factories import seed_synthetic_seasons


@pytest.fixture(scope="session")
def engine():
    url = get_settings().test_database_url
    eng = create_engine(url, future=True)

    with eng.begin() as conn:
        for schema in (SCHEMA_RAW, SCHEMA_FEATURES, SCHEMA_ML):
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def synthetic(engine) -> dict:
    """Committed synthetic seasons, plus the ground truth that was planted.

    Session-scoped and committed, because the feature views read from the
    database rather than from a session -- an uncommitted transaction is
    invisible to them.
    """
    return seed_synthetic_seasons(engine)


@pytest.fixture
def db(engine) -> OrmSession:
    """Session bound to a transaction that is always rolled back."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db, engine) -> TestClient:
    """TestClient wired to the rolled-back session AND the test engine.

    Overriding `get_db` alone is not enough: routes that read the feature views
    take an engine too, and without this override they would reach whatever
    `DATABASE_URL` names -- the developer's real database, on their machine, and
    nothing at all on anyone else's.
    """
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_engine] = lambda: engine
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def seeded(db) -> dict[str, int]:
    """A minimal but realistic session: 2 drivers, 2 stints, a safety car.

    Lap times loosely mimic Monza: ~84s, degrading within a stint. This is the
    fixture for per-row ingestion and API-shape tests; anything that needs
    rolling history across races wants `synthetic` instead.
    """
    db.add(Season(year=2024))
    db.flush()

    event = Event(season_year=2024, round=16, event_name="Italian Grand Prix",
                  event_date=dt.date(2024, 9, 1), event_format="conventional")
    db.add(event)
    db.flush()

    session = Session(event_id=event.id, session_type="R",
                      session_date=dt.datetime(2024, 9, 1, 13, 0, tzinfo=dt.UTC),
                      total_laps=53)
    db.add(session)

    team = Team(name="Ferrari", season_year=2024)
    ver = Driver(abbreviation="VER", driver_number=1, full_name="Max Verstappen")
    lec = Driver(abbreviation="LEC", driver_number=16, full_name="Charles Leclerc")
    db.add_all([team, ver, lec])
    db.flush()

    laps: list[Lap] = []
    for driver in (ver, lec):
        for lap_number in range(1, 11):
            stint = 1 if lap_number <= 5 else 2
            tyre_life = lap_number if stint == 1 else lap_number - 5
            # Lap 7 ran under safety car -- must be excluded from pace stats.
            status = [1, 4] if lap_number == 7 else [1]
            laps.append(
                Lap(
                    session_id=session.id,
                    driver_id=driver.id,
                    team_id=team.id,
                    lap_number=lap_number,
                    lap_time_ms=84_000 + tyre_life * 100 + (20_000 if lap_number == 7 else 0),
                    compound="MEDIUM" if stint == 1 else "HARD",
                    tyre_life=float(tyre_life),
                    is_fresh_tyre=True,
                    stint_number=stint,
                    track_status=status,
                    position=1.0 if driver is lec else 2.0,
                    is_accurate=True,
                )
            )
    db.add_all(laps)

    db.add_all([
        SessionResult(session_id=session.id, driver_id=lec.id, team_id=team.id,
                      position=1.0, grid_position=4.0, status="Finished", points=25.0),
        SessionResult(session_id=session.id, driver_id=ver.id, team_id=team.id,
                      position=6.0, grid_position=7.0, status="Finished", points=8.0),
    ])
    db.flush()

    return {"event_id": event.id, "session_id": session.id,
            "ver_id": ver.id, "lec_id": lec.id}
