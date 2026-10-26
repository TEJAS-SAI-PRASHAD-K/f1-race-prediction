"""Tests for the model registry and category pinning.

These cover the two XGBoost traps from the project plan, both of which fail
SILENTLY rather than raising.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from app.ml import registry
from app.ml.categories import (
    apply_categories,
    extract_categories,
    load_categories,
    save_categories,
)


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tyre_life": [1.0, 2.0, 3.0, 4.0],
            "circuit": ["Monza", "Monza", "Monaco", "Monaco"],
            "compound": ["SOFT", "HARD", "SOFT", "HARD"],
        }
    )


class TestExtractCategories:
    def test_vocabulary_is_sorted_for_determinism(self, tiny_frame):
        """Sorted so the same training data yields a byte-identical vocabulary.
        Otherwise a model and its category file can silently disagree about
        which integer code means which circuit."""
        vocab = extract_categories(tiny_frame, ["circuit", "compound"])
        assert vocab["circuit"] == ["Monaco", "Monza"]
        assert vocab["compound"] == ["HARD", "SOFT"]

    def test_ignores_nulls(self):
        df = pd.DataFrame({"circuit": ["Monza", None, "Monza"]})
        assert extract_categories(df, ["circuit"]) == {"circuit": ["Monza"]}


class TestApplyCategories:
    def test_unseen_category_becomes_nan_not_an_error(self, tiny_frame):
        """XGBoost's docs say an unseen category CANNOT be re-coded at
        inference -- it is a hard error. A new circuit on the calendar or a
        rookie driver must degrade gracefully, not take the endpoint down."""
        vocab = extract_categories(tiny_frame, ["circuit"])
        new = pd.DataFrame({"circuit": ["Monza", "Las Vegas"]})
        out, unseen = apply_categories(new, vocab)

        assert pd.isna(out.circuit.iloc[1])
        assert out.circuit.iloc[0] == "Monza"
        assert unseen["circuit"] == 1

    def test_reports_unseen_counts_so_callers_can_warn(self, tiny_frame):
        """Silent degradation is how a model ends up predicting the global
        average for a whole race weekend and nobody notices."""
        vocab = extract_categories(tiny_frame, ["circuit"])
        new = pd.DataFrame({"circuit": ["Vegas", "Miami", "Monza"]})
        _, unseen = apply_categories(new, vocab)
        assert unseen["circuit"] == 2

    def test_dtype_is_category_with_the_pinned_order(self, tiny_frame):
        vocab = extract_categories(tiny_frame, ["circuit"])
        out, _ = apply_categories(pd.DataFrame({"circuit": ["Monza"]}), vocab)
        assert str(out.circuit.dtype) == "category"
        assert list(out.circuit.cat.categories) == ["Monaco", "Monza"]

    def test_genuine_nulls_are_not_counted_as_unseen(self, tiny_frame):
        vocab = extract_categories(tiny_frame, ["circuit"])
        _, unseen = apply_categories(pd.DataFrame({"circuit": [None]}), vocab)
        assert unseen["circuit"] == 0


class TestCategoriesRoundTrip:
    def test_save_and_load(self, tmp_path, tiny_frame):
        vocab = extract_categories(tiny_frame, ["circuit", "compound"])
        path = save_categories(vocab, tmp_path / "v1.categories.json")
        assert load_categories(path) == vocab


class TestRegistryBackendDispatch:
    def test_sklearn_model_round_trips_through_joblib(self, db, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
        # registry.py binds PROJECT_ROOT at import, so patch ITS name, not
        # app.config's -- patching the source module would not reach it.
        monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)

        model = Ridge(alpha=1.0).fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))
        row = registry.save_model(
            db,
            name="t_sklearn",
            version="v1",
            backend=registry.BACKEND_SKLEARN,
            model=model,
            categories={"circuit": ["Monza"]},
            feature_names=["tyre_life"],
            metrics={"test": {"mae_ms": 1.0}},
        )
        assert row.backend == "sklearn"
        assert row.artifact_path.endswith(".joblib")

    def test_xgboost_artifact_is_json_never_pickle(self, db, tmp_path, monkeypatch):
        """THE serialisation trap. XGBoost's docs: a model with categorical
        features must be saved as JSON/UBJSON or "the information is lost".
        joblib.dump appears to work, then mispredicts without raising. So the
        registry must dispatch on backend and write .json for XGBoost.
        """
        import xgboost as xgb

        monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
        # registry.py binds PROJECT_ROOT at import, so patch ITS name, not
        # app.config's -- patching the source module would not reach it.
        monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)

        X = pd.DataFrame(
            {
                "tyre_life": [1.0, 2.0, 3.0, 4.0],
                "circuit": pd.Categorical(["Monza", "Monza", "Monaco", "Monaco"]),
            }
        )
        y = np.array([0.0, 100.0, 50.0, 150.0])
        model = xgb.XGBRegressor(
            n_estimators=5, tree_method="hist", enable_categorical=True
        ).fit(X, y)

        row = registry.save_model(
            db,
            name="t_xgb",
            version="v1",
            backend=registry.BACKEND_XGBOOST,
            model=model,
            categories={"circuit": ["Monaco", "Monza"]},
            feature_names=["tyre_life", "circuit"],
            metrics={},
        )

        assert row.backend == "xgboost"
        assert row.artifact_path.endswith(".json"), "must be JSON, not a pickle"
        saved = tmp_path / "t_xgb" / "v1.json"
        assert saved.exists()
        # A real XGBoost JSON model, not a pickled blob.
        assert "learner" in json.loads(saved.read_text())

    def test_rejects_unknown_backend(self, db):
        with pytest.raises(ValueError, match="unknown backend"):
            registry.save_model(
                db,
                name="bad",
                version="v1",
                backend="tensorflow",
                model=object(),
                categories={},
                feature_names=[],
                metrics={},
            )


class TestRegistryActiveVersion:
    def test_saving_a_new_version_deactivates_the_old(self, db, tmp_path, monkeypatch):
        """Exactly one active version per model name, so the API never has to
        guess which one to serve."""
        monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
        # registry.py binds PROJECT_ROOT at import, so patch ITS name, not
        # app.config's -- patching the source module would not reach it.
        monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)

        model = Ridge().fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
        for version in ("v1", "v2"):
            registry.save_model(
                db, name="t_active", version=version,
                backend=registry.BACKEND_SKLEARN, model=model,
                categories={}, feature_names=["x"], metrics={},
            )
        db.flush()

        rows = registry.list_models(db, "t_active")
        active = [r for r in rows if r.is_active]
        assert len(active) == 1
        assert active[0].version == "v2"

    def test_load_unregistered_model_raises_lookup_error(self, db):
        with pytest.raises(LookupError, match="no active model registered"):
            registry.load_model(db, "does_not_exist")


class TestModelVersionConstraints:
    def test_backend_check_constraint_is_enforced(self, db):
        """The DB rejects a bad backend even if application code lets one
        through -- the same defence-in-depth as the Phase 1 compound CHECK."""
        from sqlalchemy.exc import IntegrityError

        from app.db.models import ModelVersion

        db.add(
            ModelVersion(
                name="x", version="v1", backend="not_a_backend",
                artifact_path="p", trained_at=dt.datetime.now(dt.UTC), is_active=False,
            )
        )
        with pytest.raises(IntegrityError):
            db.flush()
