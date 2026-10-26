"""Tests for race prediction.

The heart of this file is the leakage suite. On a small dataset built from
rolling history, leakage is the failure mode that matters, and it produces no
error -- just a model that looks better than it is.

Everything here runs against the synthetic seasons in `tests/factories.py`, so
it passes on a fresh clone with nothing ingested. The planted signal (fixed
driver skill, fixed per-circuit degradation) is what lets the model assertions
be statements about recovering a known truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.prediction import dataset as ds
from app.ml.prediction import train as T


@pytest.fixture(scope="module")
def race_df(engine, synthetic) -> pd.DataFrame:
    df = ds.load_race_features(engine)
    if df.empty:
        pytest.skip("features.race_features is empty")
    for col in ds.IDENTITY_FEATURES:
        df[col] = df[col].astype("category")
    df[ds.DEGRADATION_FEATURE] = np.nan
    return df


class TestLeakageGuards:
    """The window frames end at `1 PRECEDING`. These prove it."""

    def test_audit_passes(self, race_df):
        report = ds.audit_leakage(race_df)
        assert report["passes"], report

    def test_teammates_agree_on_team_form(self, race_df):
        """THE subtlest leak in the project.

        A team-level feature computed over PREVIOUS races must be identical for
        both of a team's cars in the same race. If the two disagree, the window
        reached into the race being predicted -- which is exactly what a `ROWS`
        frame does and a `GROUPS` frame does not. A team has two cars per race,
        so "1 row preceding" is a driver's own teammate in that same race.
        """
        disagreements = []
        for (event_id, team), group in race_df.groupby(["event_id", "team"], observed=True):
            if len(group) < 2:
                continue
            values = group["team_form_3"].dropna().unique()
            if len(values) > 1:
                disagreements.append((event_id, team, values.tolist()))
        assert not disagreements, f"team_form_3 leaked into the current race: {disagreements[:3]}"

    def test_first_race_has_no_form(self, race_df):
        """A driver's debut cannot have rolling form. A value here means the
        window included the race itself."""
        first = race_df.sort_values("event_date").groupby("driver", observed=True).head(1)
        assert first["driver_form_3"].notna().sum() == 0
        assert first["driver_form_5"].notna().sum() == 0

    def test_first_visit_to_a_circuit_has_no_circuit_history(self, race_df):
        first_visits = (
            race_df.sort_values("event_date")
            .groupby(["driver", "circuit"], observed=True)
            .head(1)
        )
        assert first_visits["driver_circuit_mean_finish"].notna().sum() == 0

    def test_no_in_race_measurements_among_the_features(self):
        """Nothing measured DURING the race may be a feature. Reviewed by name,
        because this is the leak that is obvious once stated and easy to commit
        by accident while joining tables."""
        banned = {
            "finish_position", "podium", "points_finish", "classified", "status",
            "points", "lap_time_ms", "fastest_lap", "pit_stops",
        }
        assert not (set(ds.feature_columns(include_identity=True)) & banned)

    def test_degradation_feature_is_as_of_season(self, engine, synthetic):
        """The degradation rate for a race in season Y must be measured only
        from seasons < Y. The earliest season therefore has none, which is
        correct: nothing was known yet."""
        df = ds.build_dataset(engine)
        earliest = int(df["season_year"].min())
        first_season = df[df["season_year"] == earliest]
        assert first_season[ds.DEGRADATION_FEATURE].notna().sum() == 0


class TestSplitting:
    def test_time_split_by_season(self, race_df):
        train, test = ds.time_split(race_df)
        assert train["season_year"].max() < test["season_year"].min()

    def test_no_event_spans_both_sides(self, race_df):
        train, test = ds.time_split(race_df)
        assert not set(train["event_id"]) & set(test["event_id"])

    def test_walk_forward_trains_only_on_the_past(self, race_df):
        """The defining property. Any training row dated at or after the test
        race would be using the future to predict the past."""
        splits = ds.walk_forward_splits(race_df, min_train_races=40)
        assert len(splits) > 20
        for train, test in splits[:10]:
            assert train["event_date"].max() < test["event_date"].min()

    def test_walk_forward_tests_one_race_at_a_time(self, race_df):
        for _, test in ds.walk_forward_splits(race_df, min_train_races=40)[:5]:
            assert test["event_id"].nunique() == 1


class TestBaselines:
    def test_grid_baseline_is_strong(self, race_df):
        """It is strong because overtaking in modern F1 is hard. A model that
        cannot beat it has learned nothing qualifying did not already say."""
        rho = T.spearman_by_race(race_df, ds.baseline_grid(race_df))
        assert rho > 0.5

    def test_grid_baseline_beats_form_baseline(self, race_df):
        grid = T.evaluate_positions(race_df, ds.baseline_grid(race_df))
        form = T.evaluate_positions(race_df, ds.baseline_driver_form(race_df))
        assert grid["mae_positions"] < form["mae_positions"]

    def test_form_baseline_falls_back_to_grid_when_unknown(self, race_df):
        predicted = ds.baseline_driver_form(race_df)
        assert not np.isnan(predicted).any()


class TestMetrics:
    def test_spearman_is_one_for_a_perfect_order(self):
        df = pd.DataFrame(
            {"event_id": [1] * 5, "finish_position": [1.0, 2, 3, 4, 5], "driver": list("abcde")}
        )
        assert T.spearman_by_race(df, np.array([1.0, 2, 3, 4, 5])) == pytest.approx(1.0)

    def test_spearman_is_minus_one_for_a_reversed_order(self):
        df = pd.DataFrame(
            {"event_id": [1] * 5, "finish_position": [1.0, 2, 3, 4, 5], "driver": list("abcde")}
        )
        assert T.spearman_by_race(df, np.array([5.0, 4, 3, 2, 1])) == pytest.approx(-1.0)

    def test_spearman_is_averaged_per_race_not_pooled(self):
        """Pooling would be inflated by a good driver beating a backmarker in
        every race, which is not the question being asked. Ranking 20 cars
        within one race is."""
        df = pd.DataFrame(
            {
                "event_id": [1, 1, 1, 2, 2, 2],
                "finish_position": [1.0, 2, 3, 3, 2, 1],
                "driver": list("abcabc"),
            }
        )
        # Perfect in race 1, exactly reversed in race 2 -> mean of +1 and -1.
        assert T.spearman_by_race(df, np.array([1.0, 2, 3, 1, 2, 3])) == pytest.approx(0.0)

    def test_top_k_hit_rate(self):
        df = pd.DataFrame(
            {
                "event_id": [1] * 5,
                "finish_position": [1.0, 2, 3, 4, 5],
                "driver": list("abcde"),
            }
        )
        # Predicted top-3 = a,b,d; actual top-3 = a,b,c -> 2 of 3.
        assert T.top_k_hit_rate(df, np.array([1.0, 2, 4, 3, 5]), k=3) == pytest.approx(2 / 3)

    def test_calibration_table_buckets(self):
        rng = np.random.default_rng(0)
        probabilities = rng.random(500)
        outcomes = (rng.random(500) < probabilities).astype(int)
        table = T.calibration_table(probabilities, outcomes, bins=5)
        assert len(table) <= 5
        # A well-calibrated generator: predicted should track observed.
        assert abs(table["predicted"].mean() - table["observed"].mean()) < 0.15


class TestModelsOnSyntheticData:
    """The generator planted a recoverable signal, so these are assertions about
    whether the pipeline finds it -- not hopes about whatever is ingested."""

    @pytest.fixture(scope="class")
    def full_df(self, engine, synthetic) -> pd.DataFrame:
        df = ds.build_dataset(engine)
        if df.empty:
            pytest.skip("no race data")
        return df

    def test_ridge_beats_the_grid_baseline(self, full_df):
        """Modest but real. The honest characterisation of this problem."""
        metrics, _ = T.walk_forward_backtest(T.build_ridge, full_df, min_train_races=60)
        baselines = T.backtest_baselines(full_df, min_train_races=60)
        assert metrics["mae_positions"] < baselines["baseline_grid"]["mae_positions"]
        assert metrics["spearman"] > baselines["baseline_grid"]["spearman"]

    def test_simple_model_is_competitive_on_small_data(self, full_df):
        """At lap-level scale a booster wins; at race-level scale Ridge does.
        The amount of data decides the model."""
        ridge, _ = T.walk_forward_backtest(T.build_ridge, full_df, min_train_races=60)
        xgb, _ = T.walk_forward_backtest(T.build_xgboost, full_df, min_train_races=60)
        assert ridge["mae_positions"] <= xgb["mae_positions"] * 1.05

    def test_predictions_cover_every_tested_race(self, full_df):
        _, preds = T.walk_forward_backtest(T.build_ridge, full_df, min_train_races=60)
        assert preds["predicted"].notna().all()
        assert preds["event_id"].nunique() > 20
