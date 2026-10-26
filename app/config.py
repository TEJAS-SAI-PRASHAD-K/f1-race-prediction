"""Application settings, loaded from environment / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = (
        "postgresql+psycopg://postgres@localhost:5432/f1_race_prediction"
    )
    test_database_url: str = (
        "postgresql+psycopg://postgres@localhost:5432/f1_race_prediction_test"
    )

    # FastF1 caches raw HTTP responses here. Without a cache, every re-run
    # re-downloads session data, which is slow and rude to the upstream API.
    fastf1_cache_dir: str = "data/fastf1_cache"

    log_level: str = "INFO"

    # Browser origins allowed to call the API, comma-separated in the env var
    # (CORS_ORIGINS=https://a.example,https://b.example). The default is the
    # Vite dev server on both spellings of localhost, because a browser treats
    # http://localhost:5173 and http://127.0.0.1:5173 as different origins.
    #
    # Deliberately NOT "*": with a wildcard any page the developer visits can
    # read this API. A deployment is expected to set the variable; if it does
    # not, the dev default fails closed rather than opening up.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Echo SQL to stdout. Useful when learning what the ORM actually emits.
    sql_echo: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        """`cors_origins` split into the list CORSMiddleware wants.

        Blank entries are dropped so a trailing comma in a .env file is not read
        as an empty origin, which would never match anything.
        """
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def fastf1_cache_path(self) -> Path:
        p = PROJECT_ROOT / self.fastf1_cache_dir
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is read once per process."""
    return Settings()
