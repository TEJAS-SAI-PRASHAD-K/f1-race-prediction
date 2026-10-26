"""Fuel-load correction for lap times.

WHY THIS MODULE EXISTS
----------------------
A lap time is the sum of several effects, only one of which is tire degradation:

    lap_time = base_pace
             + degradation x tyre_life        (tires get slower)
             - fuel_effect  x laps_completed  (car gets lighter, so faster)
             + track_evolution
             + noise

Degradation and fuel push in OPPOSITE directions and are of similar magnitude,
so a naive fit measures the net and reports it as degradation. You underestimate
tire wear, and nothing warns you.

THE PART THAT IS EASY TO MISS
-----------------------------
Within a single stint the two effects are not merely correlated, they are
PERFECTLY collinear. Verified on 2024 Monza: for every stint,
corr(tyre_life, lap_number) = 1.000 exactly, because a stint advances both by
one every lap.

Perfect collinearity means the two coefficients are mathematically
unidentifiable from one stint. It is not a data-quantity problem; no number of
laps from a single stint can separate them. The design matrix is singular.

There are exactly two ways out:

  1. IMPOSE the fuel coefficient from physics and subtract it, leaving
     degradation as the only remaining slope. That is what this module does,
     and what the per-stint fit uses.

  2. POOL ACROSS STINTS, where the collinearity breaks: on 2024 Monza a
     tyre_life of 5 occurs at lap 4 and at lap 46 (sd 13.9 laps), so the model
     can see the same tire age at very different fuel loads and estimate both.
     That is what Stages B and C do.

Option 1 trades a modeling assumption for identifiability. The assumption is
visible and testable here rather than buried in a fitted coefficient, which is
the point.
"""

from __future__ import annotations

# Maximum race fuel allowed under current F1 regulations. Teams start a race
# with close to this when the circuit demands it, and finish near empty.
FUEL_LOAD_KG = 110.0

# Lap-time penalty per kilogram carried. The commonly cited range is
# 0.03-0.035 s/kg; we take the conservative end. This is an ASSUMPTION, not a
# measurement, and it is the one number here worth perturbing to see how much
# the answer moves.
SECONDS_PER_KG = 0.03


def fuel_burn_per_lap_kg(race_laps: int) -> float:
    """Average fuel burned per lap, in kg.

    Derived rather than fixed: a 44-lap race burns ~2.5 kg/lap while a 78-lap
    race burns ~1.4 kg/lap, because both start with a similar tank and must
    reach the end. Using one global constant would over-correct short races
    and under-correct long ones.
    """
    if race_laps <= 0:
        raise ValueError(f"race_laps must be positive, got {race_laps}")
    return FUEL_LOAD_KG / race_laps


def fuel_effect_seconds_per_lap(race_laps: int) -> float:
    """How much time the car gains per lap purely from burning fuel.

    Monza (53 laps) -> (110/53) * 0.03 = 0.062 s/lap
    Monaco (78 laps) -> (110/78) * 0.03 = 0.042 s/lap

    Compare that to typical tire degradation of 0.02-0.10 s/lap and the scale
    of the problem is clear: the two effects are the same size.
    """
    return fuel_burn_per_lap_kg(race_laps) * SECONDS_PER_KG


def fuel_penalty_seconds(lap_number: int, race_laps: int) -> float:
    """Seconds this lap was slowed by the fuel still in the tank.

    The reference point is an EMPTY tank at the final lap, so:
      - the last lap has a penalty of 0 and is left unchanged
      - lap 1 of a 53-lap race carries 52 laps of fuel -> ~3.2 s of penalty

    Subtracting this from a raw lap time expresses every lap as if the car
    were running on fumes, which makes laps from different points in a race
    directly comparable.
    """
    laps_of_fuel_remaining = max(race_laps - lap_number, 0)
    return laps_of_fuel_remaining * fuel_effect_seconds_per_lap(race_laps)


def fuel_penalty_ms(lap_number: int, race_laps: int) -> float:
    """`fuel_penalty_seconds` in milliseconds, matching how we store times."""
    return fuel_penalty_seconds(lap_number, race_laps) * 1000.0


def correct_lap_time_ms(lap_time_ms: float, lap_number: int, race_laps: int) -> float:
    """Remove the fuel penalty from a raw lap time.

    Returns the lap time the car would have set on an empty tank. Early,
    heavy-fuel laps become faster; the final lap is unchanged.
    """
    return lap_time_ms - fuel_penalty_ms(lap_number, race_laps)
