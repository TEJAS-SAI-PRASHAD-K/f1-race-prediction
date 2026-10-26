"""FastAPI dependency for database sessions."""

from collections.abc import Iterator

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.db.base import SessionLocal, engine


def get_engine() -> Engine:
    """The engine, as a dependency.

    Injectable for the same reason `get_db` is: binding it at module import
    would make every route read whatever `DATABASE_URL` happened to name when
    the app was imported, which no test override could then redirect.
    """
    return engine


def get_db() -> Iterator[Session]:
    """Request-scoped session. FastAPI closes it when the request ends."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
