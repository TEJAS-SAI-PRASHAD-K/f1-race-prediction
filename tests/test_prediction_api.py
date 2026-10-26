"""Tests for the prediction endpoints.

Runs entirely against the synthetic seasons in `tests/factories.py`, so a fresh
clone passes with nothing ingested. Metrics are COMPUTED from that data rather
than hardcoded: a literal copied from a real training run is false here, and a
test comparing a literal against a freshly computed baseline tests nothing but
the literal.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

from app.ml.prediction import predict as race_predict


@pytest.fixture(scope="module")
def late_event_id(engine, synthetic) -> int:
    """A race late in the final season, so every rolling feature has history."""
    df = pd.read_sql(
        text("SELECT id FROM raw.events WHERE season_year = :y ORDER BY round DESC LIMIT 1"),
        engine,
        params={"y": synthetic["last_season"]},
    )
    if df.empty:
        pytest.skip("no synthetic events")
    return int(df.iloc[0, 0])


@pytest.fixture(scope="module")
def known_drivers(synthetic) -> list[str]:
    """Driver codes the synthetic data actually contains."""
    return sorted(synthetic["driver_skill"])[:3]


def _podium_metrics(df) -> dict:
    """Walk-forward Brier score for the podium classifier.

    Mirrors what `train.train_all` reports, so the registered metrics are the
    same shape the endpoint expects. `brier_base_rate` is the score of always
    predicting the base rate -- the thing a probability has to beat to have any
    skill at all.
    """
    import numpy as np
    from sklearn.metrics import brier_score_loss

    from app.ml.prediction import dataset as ds
    from app.ml.prediction import train as T

    frames = []
    for train, test in ds.walk_forward_splits(df, min_train_races=40):
        clf = T.build_podium_classifier()
        clf.fit(ds.feature_matrix(train), train[ds.TARGET_PODIUM].astype(int))
        proba = clf.predict_proba(ds.feature_matrix(test))[:, 1]
        frames.append(test.assign(podium_proba=proba))

    preds = pd.concat(frames, ignore_index=True)
    outcomes = preds[ds.TARGET_PODIUM].astype(int).to_numpy()
    probabilities = preds["podium_proba"].to_numpy()
    base_rate = float(outcomes.mean())
    return {
        "brier": float(brier_score_loss(outcomes, probabilities)),
        "brier_base_rate": float(np.mean((base_rate - outcomes) ** 2)),
        "base_rate": base_rate,
        "n": int(len(preds)),
    }


@pytest.fixture(scope="module")
def fitted(engine, synthetic) -> dict:
    """Build the dataset and fit the models ONCE for the whole module.

    `ds.build_dataset` refits the per-stint slopes per season to derive the
    degradation feature, which costs several seconds. Rebuilding it per test took this file
    from 15s to 207s, so the expensive work is module-scoped and only the
    registry row -- which must live inside each test's rolled-back transaction
    -- is per-test.
    """
    from app.ml.categories import extract_categories
    from app.ml.prediction import dataset as ds
    from app.ml.prediction import train as T

    df = ds.build_dataset(engine)
    if df.empty:
        pytest.skip("no race data")

    X = ds.feature_matrix(df)
    return {
        "df": df,
        "position_model": T.build_ridge().fit(
            X, df[ds.TARGET_POSITION].to_numpy(dtype=float)
        ),
        "podium_model": T.build_podium_classifier().fit(
            X, df[ds.TARGET_PODIUM].astype(int).to_numpy()
        ),
        # 40 matches what `python -m app.ml.prediction.train` uses, so the
        # hardcoded model metrics below stay consistent with these baselines.
        # Raising it shrinks the tested set to later races whose grid baseline
        # happens to be stronger, which broke these assertions once.
        "baselines": T.backtest_baselines(df, min_train_races=40),
        "vocab": extract_categories(df, ds.IDENTITY_FEATURES),
        # Real walk-forward numbers on the synthetic data. Computed, never
        # copied: the endpoint reports whatever was registered, so registering
        # a literal would make every "beats the baseline" test vacuous.
        "walk_forward": T.walk_forward_backtest(T.build_ridge, df, min_train_races=40)[0],
        "podium": _podium_metrics(df),
    }


@pytest.fixture
def registered_models(db, tmp_path, monkeypatch, fitted):
    """Register the module-scoped models INSIDE this test's transaction.

    The conftest `engine` points at `f1_intelligence_test`, while
    `app.ml.prediction.train` commits to the real database, so models trained by
    the CLI are invisible here. Registering inside the transaction keeps the
    tests self-contained and leaves nothing behind.

    Features still come from the real database's `features.race_features`, since
    the test database has only ORM tables and no views. That mix is deliberate:
    the model is the thing under test, the feature data is a fixture.
    """
    from app.ml import registry
    from app.ml.prediction import dataset as ds
    from app.ml.prediction import train as T

    monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)

    df = fitted["df"]
    position_model = fitted["position_model"]
    podium_model = fitted["podium_model"]
    baseline = fitted["baselines"]
    vocab = fitted["vocab"]

    walk_forward = fitted["walk_forward"]
    # The three candidates the endpoint expects to list. Only ridge is actually
    # backtested here -- the other two exist so the response shape is real;
    # nothing asserts their values against each other.
    metrics = {
        **walk_forward,
        "n": len(df),
        "n_races": int(df.event_id.nunique()),
        "algorithm": "ridge",
        "baselines": baseline,
        "all_models": {
            "ridge": walk_forward,
            "random_forest": walk_forward,
            "xgboost": walk_forward,
        },
    }
    registry.save_model(
        db, name=T.MODEL_NAME, version="test_v1", backend=registry.BACKEND_SKLEARN,
        model=position_model, categories=vocab, feature_names=ds.feature_columns(),
        metrics=metrics, train_season_min=int(df.season_year.min()),
        train_season_max=int(df.season_year.max()), n_train_rows=len(df),
        notes="test fixture",
    )
    registry.save_model(
        db, name=T.PODIUM_MODEL_NAME, version="test_v1", backend=registry.BACKEND_SKLEARN,
        model=podium_model, categories=vocab, feature_names=ds.feature_columns(),
        metrics=fitted["podium"],
        n_train_rows=len(df), notes="test fixture",
    )
    db.flush()
    race_predict.clear_cache()
    yield
    race_predict.clear_cache()


class TestPredictRace:
    def test_returns_a_full_grid_ranked(self, client, registered_models, late_event_id):
        body = client.get(f"/api/v1/prediction/race/{late_event_id}").json()
        assert len(body["drivers"]) >= 18
        ranks = [d["predicted_rank"] for d in body["drivers"]]
        assert ranks == sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_states_that_it_is_in_sample(self, client, registered_models, late_event_id):
        """The active model was trained on all seasons including this race, so
        these numbers are not an accuracy estimate. Saying so is not optional."""
        body = client.get(f"/api/v1/prediction/race/{late_event_id}").json()
        assert any("in-sample" in w.lower() for w in body["warnings"])

    def test_podium_probabilities_are_valid(self, client, registered_models, late_event_id):
        body = client.get(f"/api/v1/prediction/race/{late_event_id}").json()
        for d in body["drivers"]:
            assert 0.0 <= d["podium_probability"] <= 1.0

    def test_front_runners_get_higher_podium_probability(
        self, client, registered_models, late_event_id
    ):
        body = client.get(f"/api/v1/prediction/race/{late_event_id}").json()
        top = body["drivers"][0]["podium_probability"]
        back = body["drivers"][-1]["podium_probability"]
        assert top > back

    def test_includes_actual_results_for_a_past_race(
        self, client, registered_models, late_event_id
    ):
        body = client.get(f"/api/v1/prediction/race/{late_event_id}").json()
        assert any(d["actual_position"] is not None for d in body["drivers"])

    def test_unknown_event_is_404(self, client, registered_models):
        assert client.get("/api/v1/prediction/race/999999").status_code == 404


class TestPredictUpcoming:
    def test_predicts_from_a_supplied_grid(self, client, registered_models):
        body = client.post(
            "/api/v1/prediction/race",
            json={
                "circuit": "Circuit 00",
                "grid": [
                    {"driver": "D00", "grid_position": 1},
                    {"driver": "D01", "grid_position": 2},
                    {"driver": "D02", "grid_position": 3},
                ],
            },
        ).json()
        assert len(body["drivers"]) == 3
        assert body["drivers"][0]["predicted_rank"] == 1

    def test_pole_sitter_usually_predicted_near_the_front(self, client, registered_models):
        body = client.post(
            "/api/v1/prediction/race",
            json={
                "circuit": "Circuit 00",
                "grid": [
                    {"driver": "D00", "grid_position": 1},
                    {"driver": "D01", "grid_position": 10},
                    {"driver": "D02", "grid_position": 18},
                ],
            },
        ).json()
        by_driver = {d["driver"]: d["predicted_rank"] for d in body["drivers"]}
        assert by_driver["D00"] < by_driver["D02"]

    def test_unknown_driver_is_reported_not_imputed(self, client, registered_models):
        """A rookie has no form history. Their prediction rests on grid position
        alone, and that is said out loud rather than hidden behind a median."""
        body = client.post(
            "/api/v1/prediction/race",
            json={
                "circuit": "Circuit 00",
                "grid": [
                    {"driver": "D00", "grid_position": 1},
                    {"driver": "ZZZ", "grid_position": 20},
                ],
            },
        ).json()
        rookie = next(d for d in body["drivers"] if d["driver"] == "ZZZ")
        assert len(rookie["missing_features"]) > 5
        assert any("ZZZ" in w and "no race history" in w for w in body["warnings"])

    def test_duplicate_driver_rejected(self, client, registered_models):
        response = client.post(
            "/api/v1/prediction/race",
            json={
                "circuit": "Circuit 00",
                "grid": [
                    {"driver": "D00", "grid_position": 1},
                    {"driver": "D00", "grid_position": 2},
                ],
            },
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "grid",
        [
            [{"driver": "D00", "grid_position": 0}],
            [{"driver": "D00", "grid_position": 99}],
            [{"driver": "D00", "grid_position": 1}],  # too few entries
        ],
    )
    def test_invalid_grid_rejected(self, client, registered_models, grid):
        assert (
            client.post(
                "/api/v1/prediction/race", json={"circuit": "Circuit 00", "grid": grid}
            ).status_code
            == 422
        )


class TestBacktest:
    def test_reports_metrics_beside_the_baseline(self, client, registered_models):
        """An MAE of 3.14 positions is unreadable without knowing that
        predicting grid order scores 3.52."""
        body = client.get("/api/v1/prediction/backtest").json()
        assert body["walk_forward"]["mae_positions"] > 0
        assert body["baseline_grid"]["mae_positions"] > 0
        assert body["improvement_over_grid_mae"] > 0

    def test_model_beats_the_grid_baseline(self, client, registered_models):
        body = client.get("/api/v1/prediction/backtest").json()
        assert (
            body["walk_forward"]["mae_positions"] < body["baseline_grid"]["mae_positions"]
        )
        assert body["walk_forward"]["spearman"] > body["baseline_grid"]["spearman"]

    def test_lists_every_candidate_model(self, client, registered_models):
        body = client.get("/api/v1/prediction/backtest").json()
        assert set(body["all_models"]) == {"ridge", "random_forest", "xgboost"}

    def test_podium_classifier_has_skill(self, client, registered_models):
        """Brier score beating the base rate. A calibrated probability is the
        point: '70%' has to mean 70% if a strategy is going to rest on it."""
        body = client.get("/api/v1/prediction/backtest").json()
        assert body["podium_skill"] is not None
        assert body["podium_skill"] > 0.2

    def test_states_the_honest_caveats(self, client, registered_models):
        body = client.get("/api/v1/prediction/backtest").json()
        assert any("SMALL" in n for n in body["notes"])
        assert any("grid order" in n for n in body["notes"])
