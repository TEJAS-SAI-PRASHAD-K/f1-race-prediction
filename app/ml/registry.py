"""Model registry: save, load and record trained models.

THE SERIALISATION TRAP
----------------------
XGBoost's docs state that a model using categorical features must be saved as
JSON or UBJSON, because otherwise "the information is lost". Pickling such a
booster with joblib APPEARS to work -- no exception, a file is written, it loads
back -- and then mispredicts, because the category mappings are gone.

That is the worst class of bug: a silent wrong answer. So this registry
dispatches on backend:

    xgboost  -> booster.save_model("*.json")   (JSON, never pickle)
    sklearn  -> joblib.dump("*.joblib")

and `ml.model_versions.backend` records which, so loading cannot guess wrong.

Every save also writes the pinned categorical vocabulary next to the artifact
(see app/ml/categories.py). A model without its vocabulary is unusable for
prediction, so they are treated as one unit.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import select, update
from sqlalchemy.orm import Session as OrmSession

from app.config import PROJECT_ROOT
from app.db.models import ModelVersion
from app.ml.categories import load_categories, save_categories

MODEL_ROOT = PROJECT_ROOT / "models"

BACKEND_XGBOOST = "xgboost"
BACKEND_SKLEARN = "sklearn"


@dataclass
class LoadedModel:
    """A model plus everything needed to use it correctly."""

    model: Any
    backend: str
    categories: dict[str, list[str]]
    feature_names: list[str]
    version: str
    metrics: dict[str, Any]


def _artifact_path(name: str, version: str, backend: str) -> Path:
    suffix = "json" if backend == BACKEND_XGBOOST else "joblib"
    return MODEL_ROOT / name / f"{version}.{suffix}"


def _categories_path(name: str, version: str) -> Path:
    return MODEL_ROOT / name / f"{version}.categories.json"


def _features_path(name: str, version: str) -> Path:
    return MODEL_ROOT / name / f"{version}.features.json"


def save_model(
    db: OrmSession,
    *,
    name: str,
    version: str,
    backend: str,
    model: Any,
    categories: dict[str, list[str]],
    feature_names: list[str],
    metrics: dict[str, Any],
    train_season_min: int | None = None,
    train_season_max: int | None = None,
    n_train_rows: int | None = None,
    notes: str | None = None,
    make_active: bool = True,
) -> ModelVersion:
    """Persist a model and record it in `ml.model_versions`.

    `feature_names` is saved because a saved booster expects its columns in the
    exact training order. A reordered frame at predict time yields plausible,
    wrong numbers rather than an error, so the order is pinned on disk.
    """
    if backend not in (BACKEND_XGBOOST, BACKEND_SKLEARN):
        raise ValueError(f"unknown backend {backend!r}")

    artifact = _artifact_path(name, version, backend)
    artifact.parent.mkdir(parents=True, exist_ok=True)

    if backend == BACKEND_XGBOOST:
        # JSON, not pickle. See the module docstring.
        model.save_model(str(artifact))
    else:
        joblib.dump(model, artifact)

    save_categories(categories, _categories_path(name, version))
    _features_path(name, version).write_text(json.dumps(feature_names, indent=2))

    if make_active:
        # Exactly one active version per model name.
        db.execute(
            update(ModelVersion)
            .where(ModelVersion.name == name, ModelVersion.is_active.is_(True))
            .values(is_active=False)
        )

    row = db.execute(
        select(ModelVersion).where(
            ModelVersion.name == name, ModelVersion.version == version
        )
    ).scalar_one_or_none()

    fields = {
        "backend": backend,
        "artifact_path": str(artifact.relative_to(PROJECT_ROOT)),
        "categories_path": str(
            _categories_path(name, version).relative_to(PROJECT_ROOT)
        ),
        "trained_at": dt.datetime.now(dt.UTC),
        "train_season_min": train_season_min,
        "train_season_max": train_season_max,
        "n_train_rows": n_train_rows,
        "metrics_json": json.dumps(metrics)[:4096],
        "is_active": make_active,
        "notes": notes,
    }

    if row is None:
        row = ModelVersion(name=name, version=version, **fields)
        db.add(row)
    else:
        for key, value in fields.items():
            setattr(row, key, value)

    db.flush()
    return row


def load_model(db: OrmSession, name: str, version: str | None = None) -> LoadedModel:
    """Load a model by name; defaults to the active version.

    Raises `FileNotFoundError` with the registry row's path if the artifact is
    missing, which is the common failure after a fresh clone (models/ is
    gitignored because artifacts are build output, not source).
    """
    stmt = select(ModelVersion).where(ModelVersion.name == name)
    stmt = (
        stmt.where(ModelVersion.version == version)
        if version
        else stmt.where(ModelVersion.is_active.is_(True))
    )
    row = db.execute(stmt).scalar_one_or_none()
    if row is None:
        raise LookupError(
            f"no {'active ' if version is None else ''}model registered as {name!r}"
            f"{'' if version is None else f' version {version!r}'}"
        )

    artifact = PROJECT_ROOT / row.artifact_path
    if not artifact.exists():
        raise FileNotFoundError(
            f"registry row {name}/{row.version} points at {artifact}, which is missing. "
            "Re-run training: python -m app.ml.prediction.train"
        )

    if row.backend == BACKEND_XGBOOST:
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(str(artifact))
    else:
        model = joblib.load(artifact)

    categories = load_categories(PROJECT_ROOT / row.categories_path)
    feature_names = json.loads(_features_path(name, row.version).read_text())

    return LoadedModel(
        model=model,
        backend=row.backend,
        categories=categories,
        feature_names=feature_names,
        version=row.version,
        metrics=json.loads(row.metrics_json) if row.metrics_json else {},
    )


def list_models(db: OrmSession, name: str | None = None) -> list[ModelVersion]:
    stmt = select(ModelVersion)
    if name:
        stmt = stmt.where(ModelVersion.name == name)
    return list(
        db.execute(stmt.order_by(ModelVersion.trained_at.desc())).scalars().all()
    )
