"""Tests for the Phase 2 dataset: fuel correction, target design, splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.features import fuel
from app.ml.features import laps as ds


def _stint_frame(
    n_laps: int = 20,
    race_laps: int = 53,
    first_lap: int = 1,
    base_ms: float = 85_000.0,
    degradation_ms_per_lap: float = 60.0,
    session_id: int = 1,
    driver_id: int = 7,
    stint_number: int = 1,
    season_year: int = 2024,
) -> pd.DataFrame:
    """A synthetic stint with a KNOWN degradation rate and a real fuel effect.

    Raw lap time = base + degradation*tyre_life + fuel_penalty(lap_number).
    Because we build it, we know the right answer and can check the pipeline
    recovers it.
    """
    tyre_life = np.arange(1, n_laps + 1, dtype=float)
    lap_number = first_lap + tyre_life - 1
    penalty = np.array([fuel.fuel_penalty_ms(int(n), race_laps) for n in lap_number])
    lap_time = base_ms + degradation_ms_per_lap * tyre_life + penalty

    return pd.DataFrame(
        {
            "lap_id": np.arange(len(tyre_life)) + session_id * 1000,
            "session_id": session_id,
            "driver_id": driver_id,
            "season_year": season_year,
            "round": 1,
            "event_date": pd.Timestamp("2024-09-01"),
            "circuit": "Monza",
            "driver": "VER",
            "team": "Red Bull Racing",
            "compound": "HARD",
            "stint_number": stint_number,
            "lap_number": lap_number.astype(int),
            "tyre_life": tyre_life,
            "is_fresh_tyre": True,
            "race_laps": race_laps,
            "lap_time_ms": lap_time,
            "stint_median_ms": float(np.median(lap_time)),
            "stint_green_laps": n_laps,
            "stint_first_lap": first_lap,
            "stint_last_lap": int(lap_number[-1]),
            "air_temp": 25.0,
            "track_temp": 38.0,
            "any_rainfall": False,
        }
    )


class TestFuelCorrection:
    def test_vectorised_matches_the_scalar_reference(self):
        """`add_fuel_correction` is a vectorised reimplementation of
        `fuel.fuel_penalty_ms`. Two implementations of one formula is exactly
        where silent drift happens, so pin them together."""
        df = ds.add_fuel_correction(_stint_frame())
        expected = [
            fuel.fuel_penalty_ms(int(n), int(r))
            for n, r in zip(df.lap_number, df.race_laps, strict=True)
        ]
        assert df.fuel_penalty_ms.to_numpy() == pytest.approx(expected)

    def test_correction_recovers_the_known_degradation_rate(self):
        """The headline property. Build a stint with 60 ms/lap of degradation
        hidden under a real fuel effect, then check the corrected times reveal
        60 ms/lap rather than the much smaller raw slope."""
        df = ds.add_fuel_correction(_stint_frame(degradation_ms_per_lap=60.0))

        raw_slope = np.polyfit(df.tyre_life, df.lap_time_ms, 1)[0]
        corrected_slope = np.polyfit(df.tyre_life, df.lap_time_fuel_corr_ms, 1)[0]

        assert corrected_slope == pytest.approx(60.0, abs=1e-6)
        # The raw slope understates it by roughly the fuel effect.
        assert raw_slope < corrected_slope
        assert corrected_slope - raw_slope == pytest.approx(
            fuel.fuel_effect_seconds_per_lap(53) * 1000, rel=1e-6
        )

    def test_a_flat_raw_stint_is_really_degrading(self):
        """If raw lap times are flat, degradation exactly cancelled fuel gain.
        Uncorrected analysis would report zero degradation, which is wrong."""
        effect_ms = fuel.fuel_effect_seconds_per_lap(53) * 1000
        df = ds.add_fuel_correction(_stint_frame(degradation_ms_per_lap=effect_ms))
        raw_slope = np.polyfit(df.tyre_life, df.lap_time_ms, 1)[0]
        corrected_slope = np.polyfit(df.tyre_life, df.lap_time_fuel_corr_ms, 1)[0]
        assert raw_slope == pytest.approx(0.0, abs=1e-6)
        assert corrected_slope == pytest.approx(effect_ms, rel=1e-6)


class TestCollinearity:
    def test_tyre_life_and_lap_number_are_perfectly_collinear_in_one_stint(self):
        """The reason Stage A must IMPOSE the fuel coefficient rather than fit
        it. Within a stint these two move together exactly, so their
        coefficients are not identifiable at any sample size."""
        df = _stint_frame(n_laps=30)
        assert np.corrcoef(df.tyre_life, df.lap_number)[0, 1] == pytest.approx(1.0)

    def test_pooling_across_stints_breaks_the_collinearity(self):
        """And the reason Stages B and C can estimate it. The same tire age
        appearing at different lap numbers is what makes both effects
        separable."""
        early = _stint_frame(n_laps=15, first_lap=1, stint_number=1)
        late = _stint_frame(n_laps=15, first_lap=35, stint_number=2)
        pooled = pd.concat([early, late], ignore_index=True)
        correlation = np.corrcoef(pooled.tyre_life, pooled.lap_number)[0, 1]
        assert correlation < 0.8


class TestTarget:
    def test_target_is_deviation_from_the_stint_median(self):
        df = ds.add_target(ds.add_fuel_correction(_stint_frame()))
        assert df[ds.TARGET].median() == pytest.approx(0.0, abs=1e-6)

    def test_target_preserves_the_slope_it_removes_the_level(self):
        """The natural worry about subtracting a stint baseline is that it
        removes the degradation signal. It does not: the LEVEL goes, the SLOPE
        survives, and the slope is the quantity of interest."""
        df = ds.add_target(ds.add_fuel_correction(_stint_frame(degradation_ms_per_lap=60.0)))
        target_slope = np.polyfit(df.tyre_life, df[ds.TARGET], 1)[0]
        assert target_slope == pytest.approx(60.0, abs=1e-6)

    def test_stint_key_separates_drivers_and_stints(self):
        a = _stint_frame(driver_id=1, stint_number=1)
        b = _stint_frame(driver_id=2, stint_number=1)
        c = _stint_frame(driver_id=1, stint_number=2)
        out = ds.add_target(ds.add_fuel_correction(pd.concat([a, b, c], ignore_index=True)))
        assert out.stint_key.nunique() == 3

    def test_extreme_outlier_is_dropped(self):
        """A lap 19s off its stint median is not tire degradation -- it is a
        driver on slicks when the rain arrived. Observed at Zandvoort 2023."""
        df = _stint_frame(n_laps=20)
        df.loc[5, "lap_time_ms"] += 19_000
        out = ds.add_target(ds.add_fuel_correction(df))
        assert len(out) == 19
        assert out[ds.TARGET].abs().max() < ds.MAX_ABS_TARGET_MS

    def test_median_is_recomputed_after_dropping(self):
        """Pass 2 matters. The baseline must be computed from the rows the
        model actually sees, not from a set that still included the outlier."""
        df = _stint_frame(n_laps=20, degradation_ms_per_lap=0.0)
        df.loc[0, "lap_time_ms"] += 19_000
        out = ds.add_target(ds.add_fuel_correction(df))
        survivors = out["lap_time_fuel_corr_ms"]
        assert out["stint_corr_median_ms"].iloc[0] == pytest.approx(
            float(np.median(survivors))
        )


class TestFeatures:
    def test_categoricals_get_category_dtype(self):
        """XGBoost needs `category` dtype with enable_categorical=True, and
        HistGBR reads the same dtype via categorical_features='from_dtype'."""
        out = ds.add_features(ds.add_target(ds.add_fuel_correction(_stint_frame())))
        for col in ds.CATEGORICAL_FEATURES:
            assert str(out[col].dtype) == "category"

    def test_quadratic_term_present(self):
        out = ds.add_features(ds.add_target(ds.add_fuel_correction(_stint_frame())))
        assert out.tyre_life_sq.to_numpy() == pytest.approx(
            (out.tyre_life**2).to_numpy()
        )

    def test_feature_matrix_order_is_fixed(self):
        """A saved booster expects its training column order. A reordered frame
        gives plausible, wrong predictions rather than an error."""
        out = ds.add_features(ds.add_target(ds.add_fuel_correction(_stint_frame())))
        assert list(ds.feature_matrix(out).columns) == ds.FEATURES


class TestTimeSplit:
    def test_splits_on_season_not_randomly(self):
        old = _stint_frame(season_year=2023, session_id=1)
        new = _stint_frame(season_year=2024, session_id=2)
        df = ds.add_features(ds.add_target(ds.add_fuel_correction(
            pd.concat([old, new], ignore_index=True))))

        train, test = ds.time_split(df)
        assert set(train.season_year.unique()) == {2023}
        assert set(test.season_year.unique()) == {2024}

    def test_no_stint_appears_on_both_sides(self):
        """The property a random split would violate. Laps in one stint are
        near-duplicates, so sharing a stint across the split inflates the
        score."""
        old = _stint_frame(season_year=2023, session_id=1)
        new = _stint_frame(season_year=2024, session_id=2)
        df = ds.add_features(ds.add_target(ds.add_fuel_correction(
            pd.concat([old, new], ignore_index=True))))
        train, test = ds.time_split(df)
        assert not set(train.stint_key) & set(test.stint_key)


class TestBaseline:
    def test_baseline_is_mean_absolute_target(self):
        """Because the target is already a deviation from the stint median,
        the do-nothing prediction is literally zero."""
        df = ds.add_target(ds.add_fuel_correction(_stint_frame()))
        assert ds.baseline_mae_ms(df) == pytest.approx(df[ds.TARGET].abs().mean())

    def test_baseline_is_positive_for_a_degrading_stint(self):
        df = ds.add_target(ds.add_fuel_correction(_stint_frame(degradation_ms_per_lap=60.0)))
        assert ds.baseline_mae_ms(df) > 0


class TestFilteredLoad:
    """The filtered read must not change the answer.

    `add_degradation_feature` only needs one circuit's prior stints, and reading
    the whole 104k-row view to keep ~2.8k of them was most of the ~8.7s a
    single-race request took. Pushing the filter into SQL is only safe because every step
    after the read is per-row or per-stint, so these tests assert that property
    directly instead of trusting the reasoning.

    Two filtered reads are compared rather than a filtered read against the full
    one: the full-view read plus pipeline costs several seconds, and the
    invariant under test -- that neighbouring circuits do not influence each
    other's fits -- is visible without it.
    """

    @pytest.fixture(scope="class")
    def circuits(self, engine, synthetic) -> list[str]:
        from sqlalchemy import text

        found = pd.read_sql(
            text("SELECT DISTINCT circuit FROM features.stint_laps ORDER BY circuit"),
            engine,
        )["circuit"].tolist()
        if len(found) < 2:
            pytest.skip("needs at least two circuits ingested")
        return found[:2]

    def test_slopes_do_not_depend_on_which_other_circuits_were_read(self, engine, circuits):
        from app.ml.features import degradation

        first, second = circuits

        together = degradation.fit_all_stints(
            ds.build_dataset(engine, circuits=[first, second])
        )
        alone = degradation.fit_all_stints(ds.build_dataset(engine, circuits=[first]))

        from_pair = (
            together[together["circuit"] == first]
            .set_index("stint_key")["slope"]
            .sort_index()
        )
        from_single = alone.set_index("stint_key")["slope"].sort_index()

        assert len(from_single) > 0
        assert list(from_pair.index) == list(from_single.index)
        # Bit-identical, not approximately equal: the rows are the same rows.
        assert (from_pair.to_numpy() == from_single.to_numpy()).all()

    def test_before_season_excludes_the_named_season(self, engine, circuits):
        """The as-of degradation feature for season Y may only see seasons < Y;
        a boundary that leaked by one would leak the race being predicted."""
        all_seasons = ds.build_dataset(engine, circuits=circuits)["season_year"]
        cutoff = int(all_seasons.max())

        prior = ds.build_dataset(engine, circuits=circuits, before_season=cutoff)
        assert not prior.empty
        assert int(prior["season_year"].max()) < cutoff

    def test_circuit_filter_returns_only_that_circuit(self, engine, circuits):
        rows = ds.load_stint_laps(engine, circuits=[circuits[0]])
        assert set(rows["circuit"].unique()) == {circuits[0]}
