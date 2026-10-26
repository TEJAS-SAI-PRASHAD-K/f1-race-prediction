"""Tests for the fuel-load correction.

The fuel model is an ASSUMPTION imposed from physics (110 kg at 0.03 s/kg), not
something fitted. That makes it exactly the kind of thing that needs tests: if
someone changes a constant, these pin what the change implies.
"""

from __future__ import annotations

import pytest

from app.ml.features import fuel


class TestFuelBurnPerLap:
    def test_short_race_burns_more_per_lap(self):
        """Derived from race length rather than fixed, so a 44-lap race burns
        more per lap than a 78-lap race from a similar tank."""
        assert fuel.fuel_burn_per_lap_kg(44) > fuel.fuel_burn_per_lap_kg(78)

    def test_known_values(self):
        assert fuel.fuel_burn_per_lap_kg(53) == pytest.approx(110.0 / 53)
        assert fuel.fuel_burn_per_lap_kg(78) == pytest.approx(110.0 / 78)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_non_positive_race_laps(self, bad):
        with pytest.raises(ValueError, match="race_laps must be positive"):
            fuel.fuel_burn_per_lap_kg(bad)


class TestFuelEffectPerLap:
    def test_magnitude_is_comparable_to_degradation(self):
        """The whole reason this module exists: the fuel effect is the same
        order of magnitude as tire degradation (measured median +0.058 s/lap),
        so ignoring it does not produce a small error -- it produces a sign
        error."""
        effect = fuel.fuel_effect_seconds_per_lap(53)
        assert 0.04 < effect < 0.09

    def test_monza_and_monaco(self):
        assert fuel.fuel_effect_seconds_per_lap(53) == pytest.approx(0.0623, abs=1e-4)
        assert fuel.fuel_effect_seconds_per_lap(78) == pytest.approx(0.0423, abs=1e-4)


class TestFuelPenalty:
    def test_final_lap_has_no_penalty(self):
        """The reference point is an empty tank on the last lap, so that lap is
        left untouched."""
        assert fuel.fuel_penalty_seconds(53, 53) == 0.0

    def test_first_lap_carries_the_most(self):
        assert fuel.fuel_penalty_seconds(1, 53) > fuel.fuel_penalty_seconds(27, 53)

    def test_penalty_decreases_monotonically(self):
        penalties = [fuel.fuel_penalty_seconds(n, 53) for n in range(1, 54)]
        # Pairwise: the second sequence is intentionally one shorter, so no strict=.
        assert all(a >= b for a, b in zip(penalties, penalties[1:]))  # noqa: B905

    def test_first_lap_magnitude_is_plausible(self):
        """Lap 1 of a 53-lap race carries 52 laps of fuel. Around 3s is the
        commonly quoted figure for a full tank."""
        assert fuel.fuel_penalty_seconds(1, 53) == pytest.approx(3.24, abs=0.05)

    def test_laps_beyond_race_distance_clamp_to_zero(self):
        """Lapped cars and timing quirks can produce lap_number > race_laps;
        a negative fuel load is not physical."""
        assert fuel.fuel_penalty_seconds(60, 53) == 0.0

    def test_ms_matches_seconds(self):
        assert fuel.fuel_penalty_ms(10, 53) == pytest.approx(
            fuel.fuel_penalty_seconds(10, 53) * 1000
        )


class TestCorrectLapTime:
    def test_early_laps_become_faster(self):
        """A heavy car was slowed by fuel, so removing the penalty must make an
        early lap faster, not slower. Getting this sign backwards would double
        the confound instead of removing it."""
        raw = 86_170.0
        corrected = fuel.correct_lap_time_ms(raw, lap_number=2, race_laps=53)
        assert corrected < raw

    def test_final_lap_is_unchanged(self):
        raw = 83_000.0
        assert fuel.correct_lap_time_ms(raw, 53, 53) == raw

    def test_correction_removes_the_within_stint_trend(self):
        """The point of the correction, stated as a test.

        Build a stint whose RAW lap times are perfectly flat. Physically that
        means the tire was degrading at exactly the rate the car was getting
        lighter. After correction the trend must appear as POSITIVE
        degradation, because the fuel gain has been taken back out.
        """
        import numpy as np

        race_laps = 53
        laps = np.arange(10, 31)
        raw = np.full(len(laps), 85_000.0)

        corrected = np.array(
            [fuel.correct_lap_time_ms(t, int(n), race_laps) for t, n in zip(raw, laps, strict=True)]
        )
        raw_slope = np.polyfit(laps, raw, 1)[0]
        corrected_slope = np.polyfit(laps, corrected, 1)[0]

        assert raw_slope == pytest.approx(0.0, abs=1e-6), "raw stint is flat by construction"
        assert corrected_slope > 0, "correction must reveal the hidden degradation"
        assert corrected_slope == pytest.approx(
            fuel.fuel_effect_seconds_per_lap(race_laps) * 1000, rel=1e-6
        )
