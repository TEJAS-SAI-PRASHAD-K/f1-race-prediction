"""FastAPI application.

Interactive docs at http://localhost:8000/docs once running.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import catalog, health, prediction, sessions
from app.config import get_settings

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="F1 Race Prediction API",
        version="0.1.0",
        description=(
            "Predicts Formula 1 race finishing positions from the starting grid, "
            "driver and team form, and circuit history, using FastF1 timing data "
            "from 2018 onward.\n\n"
            "**What each group needs.** `catalog` and `sessions` are read-only "
            "views of ingested data and need nothing else. `prediction` serves a "
            "trained model -- run `python -m app.ml.prediction.train` if an "
            "endpoint reports no registered model.\n\n"
            "**Every feature is as-of.** The rolling form windows end at "
            "`1 PRECEDING` and use `GROUPS`, not `ROWS`, so a driver's own "
            "teammate in the race being predicted cannot leak into a team-level "
            "feature. See the `features.race_features` migration.\n\n"
            "Two things to know when reading lap data:\n"
            "- `track_status` is an array of flags, not a number. `[1]` is clear; "
            "`[1,4]` means the lap was partly clear and partly under safety car.\n"
            "- `compound` does not distinguish C1-C5. SOFT/MEDIUM/HARD are relative "
            "to each weekend's allocation, so a Monza soft and a Monaco soft are "
            "different rubber."
        ),
    )

    # Without this a browser on any other origin cannot read a response at all,
    # which is how a separate frontend is served in development. The allowed
    # origins come from CORS_ORIGINS (comma-separated) so a deployment can name
    # its own; the default covers only the local Vite dev server. See
    # Settings.cors_origins for why the default is not "*".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(catalog.router, prefix=API_PREFIX)
    app.include_router(sessions.router, prefix=API_PREFIX)
    app.include_router(prediction.router, prefix=API_PREFIX)
    return app


app = create_app()
