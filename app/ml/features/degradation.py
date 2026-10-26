"""Per-stint ordinary least squares: the circuit degradation rate.

The simplest model that is still useful. For each stint, fit

    lap_time_seconds = intercept + slope * tyre_life

and read the slope directly as the degradation rate in seconds per lap. Then
aggregate slopes into a (circuit, compound) lookup table.

This is a COMPLETE, usable estimate on its own. Its median over prior seasons
is what `circuit_degradation_s_per_lap` carries into the race-prediction model,
so it is a feature here rather than an end in itself.

WHAT THIS CANNOT DO, AND WHY
----------------------------
Within one stint, tyre_life and lap_number are PERFECTLY collinear --
corr = 1.000 exactly, verified on real stints -- because a stint advances both
by one every lap. So the fuel effect and the degradation effect cannot be
separated here at any sample size: the design matrix is singular.

It therefore works on lap times that have ALREADY been fuel-corrected by
app/ml/features/fuel.py, which imposes the fuel coefficient from physics rather than
estimating it. That is the price of fitting one stint at a time.

Fit the raw times instead and the slope you get is (degradation - fuel_effect),
which is frequently negative -- implying tires get faster as they age.
`fit_all_stints` can do both so the difference is visible rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many laps a slope is dominated by noise: two laps define a line
# exactly, which tells you nothing.
MIN_LAPS_FOR_FIT = 8


def fit_stint(tyre_life: np.ndarray, lap_time_s: np.ndarray) -> dict[str, float]:
    """Least-squares line through one stint.

    Returns slope (s/lap), intercept (s), r2, and n. `r2` is reported because a
    slope without a goodness-of-fit is not interpretable: a confident-looking
    -0.3 s/lap with r2 = 0.02 is noise, not a finding.
    """
    n = len(tyre_life)
    if n < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": float(n)}

    slope, intercept = np.polyfit(tyre_life, lap_time_s, 1)
    predicted = slope * tyre_life + intercept
    residual_ss = float(((lap_time_s - predicted) ** 2).sum())
    total_ss = float(((lap_time_s - lap_time_s.mean()) ** 2).sum())
    r2 = 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n": float(n),
    }


def fit_all_stints(
    df: pd.DataFrame, min_laps: int = MIN_LAPS_FOR_FIT, use_fuel_correction: bool = True
) -> pd.DataFrame:
    """Fit every stint in the dataset.

    `use_fuel_correction=False` fits RAW lap times and is there to demonstrate
    the confound: compare the two outputs and the slope distribution shifts by
    roughly the fuel effect, with many raw slopes landing below zero.

    Returns one row per stint with its identifiers and fit statistics.
    """
    time_col = "lap_time_fuel_corr_ms" if use_fuel_correction else "lap_time_ms"

    records: list[dict] = []
    for stint_key, group in df.groupby("stint_key", observed=True):
        if len(group) < min_laps:
            continue
        group = group.sort_values("tyre_life")
        fit = fit_stint(
            group["tyre_life"].to_numpy(dtype=float),
            (group[time_col].to_numpy(dtype=float) / 1000.0),
        )
        first = group.iloc[0]
        records.append(
            {
                "stint_key": stint_key,
                "season_year": int(first["season_year"]),
                "round": int(first["round"]),
                "circuit": first["circuit"],
                "driver": first["driver"],
                "team": first["team"],
                "compound": first["compound"],
                "stint_number": int(first["stint_number"]),
                "tyre_life_min": float(group["tyre_life"].min()),
                "tyre_life_max": float(group["tyre_life"].max()),
                **fit,
            }
        )

    return pd.DataFrame(records)


def degradation_lookup(fits: pd.DataFrame, min_stints: int = 3) -> pd.DataFrame:
    """Aggregate per-stint slopes into a (circuit, compound) table.

    MEDIAN rather than mean: a handful of stints still carry contamination the
    filters did not catch, and a median ignores them where a mean would not.

    `min_stints` suppresses combinations with too little support -- a "circuit
    degradation rate" from one stint is an anecdote.
    """
    grouped = fits.groupby(["circuit", "compound"], observed=True)["slope"]
    table = grouped.agg(
        slope_median="median",
        slope_p25=lambda s: s.quantile(0.25),
        slope_p75=lambda s: s.quantile(0.75),
        n_stints="count",
    ).reset_index()
    return table[table["n_stints"] >= min_stints].sort_values(
        ["circuit", "compound"]
    )


def predict_from_lookup(
    table: pd.DataFrame, circuit: str, compound: str, tyre_life: np.ndarray
) -> np.ndarray | None:
    """Predict the lap-time delta curve from the degradation lookup table.

    Returns `slope * tyre_life` relative to a new tire, or None when the
    combination is absent from the table. Returning None rather than a global
    fallback keeps "we have no data for this" distinguishable from a real
    prediction -- the same reasoning as NULL vs 0 in the schema.
    """
    match = table[(table.circuit == circuit) & (table.compound == compound)]
    if match.empty:
        return None
    slope = float(match.iloc[0]["slope_median"])
    return slope * np.asarray(tyre_life, dtype=float)
