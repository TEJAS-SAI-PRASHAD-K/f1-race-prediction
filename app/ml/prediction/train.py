"""Race prediction: train, backtest, register.

    python -m app.ml.prediction.train
    python -m app.ml.prediction.train --no-save --include-identity

MODEL ORDER IS DELIBERATE
-------------------------
This dataset is **2,556 rows**, strongly correlated within races, and it
punishes a flexible model. So the order is Ridge first, then a shallow random
forest, then XGBoost -- and the expectation is that the simple model is
competitive.

That is the lesson: the amount of data decides the model, not the model's
reputation. The same feature pipeline run over 104,000 lap-level rows rewards a
booster instead; here, regularisation matters more than model choice.

THE BASELINE IS THE POINT
-------------------------
"The race finishes in grid order" is a very strong baseline, because overtaking
in modern F1 is hard. Every metric below is reported against it. A model that
cannot beat it has learned nothing that qualifying did not already say, and
reporting an MAE without that comparison would hide exactly that.

WHY RANK METRICS AND NOT JUST MAE
---------------------------------
Position MAE treats "predicted 1st, finished 3rd" and "predicted 18th, finished
20th" as the same 2-position error. They are not remotely the same mistake.
Spearman correlation per race measures whether the ORDER is right, and top-3 hit
rate measures the part anyone actually cares about.
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.db.base import SessionLocal, engine
from app.ml import registry
from app.ml.categories import extract_categories
from app.ml.prediction import dataset as ds

MODEL_NAME = "race_prediction"
PODIUM_MODEL_NAME = "race_prediction_podium"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def spearman_by_race(df: pd.DataFrame, predicted: np.ndarray) -> float:
    """Mean per-race Spearman correlation between predicted and actual order.

    Computed PER RACE and then averaged, not pooled. Pooling across races would
    be inflated by the fact that a good driver tends to finish ahead of a bad
    one in every race -- the model would score well for knowing Verstappen beats
    a backmarker, which is not the question. Within a race, ranking 20 cars that
    all plausibly finish in any order is the actual task.
    """
    frame = df.assign(_pred=predicted)
    scores: list[float] = []
    for _, group in frame.groupby("event_id", observed=True):
        if len(group) < 3:
            continue
        rho, _ = spearmanr(group["_pred"], group[ds.TARGET_POSITION])
        if not np.isnan(rho):
            scores.append(float(rho))
    return float(np.mean(scores)) if scores else float("nan")


def top_k_hit_rate(df: pd.DataFrame, predicted: np.ndarray, k: int = 3) -> float:
    """Fraction of the actual top-k captured by the predicted top-k, per race."""
    frame = df.assign(_pred=predicted)
    rates: list[float] = []
    for _, group in frame.groupby("event_id", observed=True):
        if len(group) < k:
            continue
        predicted_set = set(group.nsmallest(k, "_pred")["driver"])
        actual_set = set(group.nsmallest(k, ds.TARGET_POSITION)["driver"])
        rates.append(len(predicted_set & actual_set) / k)
    return float(np.mean(rates)) if rates else float("nan")


def evaluate_positions(df: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    actual = df[ds.TARGET_POSITION].to_numpy(dtype=float)
    return {
        "mae_positions": float(mean_absolute_error(actual, predicted)),
        "spearman": spearman_by_race(df, predicted),
        "top3_hit_rate": top_k_hit_rate(df, predicted, 3),
        "top1_hit_rate": top_k_hit_rate(df, predicted, 1),
        "n": int(len(df)),
        "n_races": int(df["event_id"].nunique()),
    }


def calibration_table(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 5
) -> pd.DataFrame:
    """Predicted vs observed podium rate, bucketed.

    A model can rank well and still be badly calibrated, and a miscalibrated
    probability is dangerous to act on: "70%" has to mean 70% if anyone is going
    to bet a strategy on it. Brier score alone will not show this -- a
    calibration table will.
    """
    frame = pd.DataFrame({"p": probabilities, "y": outcomes.astype(float)})
    frame["bucket"] = pd.cut(frame["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    out = frame.groupby("bucket", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )
    return out.reset_index()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def _preprocessor(include_identity: bool) -> ColumnTransformer:
    """Median imputation plus scaling; one-hot identity columns if requested.

    Imputation is required because Ridge and sklearn's RandomForest cannot
    accept NaN. That is a real cost: `driver_circuit_mean_finish` is missing 35%
    of the time (a driver's first visit to a circuit), and imputing the median
    asserts "this driver performs averagely here", which is a guess. XGBoost
    learns a split direction for missing instead, which is one concrete
    advantage it has on this data.
    """
    transformers = [
        (
            "num",
            Pipeline(
                [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
            ),
            ds.NUMERIC_FEATURES,
        )
    ]
    if include_identity:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ds.IDENTITY_FEATURES,
            )
        )
    return ColumnTransformer(transformers)


def build_ridge(include_identity: bool = False) -> Pipeline:
    """Regularised linear model. Expected to be competitive, not a straw man.

    alpha=10.0 rather than the usual 1.0: with 2,556 rows and 13 features the
    penalty has to work harder, and the value was chosen on the walk-forward
    backtest rather than guessed.
    """
    return Pipeline(
        [("pre", _preprocessor(include_identity)), ("model", Ridge(alpha=10.0))]
    )


def build_random_forest(include_identity: bool = False) -> Pipeline:
    """Shallow forest. `max_depth=6` and a large `min_samples_leaf` because the
    dataset is small enough that a deep tree memorises driver names."""
    return Pipeline(
        [
            ("pre", _preprocessor(include_identity)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=6,
                    min_samples_leaf=20,
                    max_features=0.6,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost(include_identity: bool = False):
    """XGBoost, heavily regularised and third in line.

    `max_depth=2` is not a typo. On 2,556 rows anything deeper overfits, and the
    effect is stronger here than at 100x the sample size.
    Native NaN handling is its real advantage on this data, since 35% of
    `driver_circuit_mean_finish` is missing and imputing it is a fabrication.
    """
    import xgboost as xgb

    return xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=25,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=25.0,
        reg_alpha=0.5,
        tree_method="hist",
        enable_categorical=include_identity,
        random_state=42,
        n_jobs=-1,
    )


def build_podium_classifier(include_identity: bool = False) -> Pipeline:
    """Logistic regression for P(podium).

    Podium rather than win: ~360 podium finishes against ~121 wins in this data,
    so the positive class is three times larger and the problem far better
    conditioned. Predicting wins directly from 121 examples is not a modelling
    problem, it is a wish.
    """
    return Pipeline(
        [
            ("pre", _preprocessor(include_identity)),
            ("model", LogisticRegression(C=0.3, max_iter=2000)),
        ]
    )


MODEL_BUILDERS = {
    "ridge": build_ridge,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
}


# ---------------------------------------------------------------------------
# backtesting
# ---------------------------------------------------------------------------


def walk_forward_backtest(
    builder, df: pd.DataFrame, *, include_identity: bool = False, min_train_races: int = 40
) -> tuple[dict[str, float], pd.DataFrame]:
    """Train on everything before race N, predict race N, step forward.

    The only evaluation that mirrors real use, and strictly harder than a single
    season holdout because the early folds train on very little.

    Returns aggregate metrics plus the per-race predictions, so the predictions
    can be inspected rather than taken on trust.
    """
    splits = ds.walk_forward_splits(df, min_train_races=min_train_races)
    frames: list[pd.DataFrame] = []

    for train, test in splits:
        model = builder(include_identity)
        X_train = ds.feature_matrix(train, include_identity)
        X_test = ds.feature_matrix(test, include_identity)
        model.fit(X_train, train[ds.TARGET_POSITION].to_numpy(dtype=float))
        frames.append(test.assign(predicted=model.predict(X_test)))

    predictions = pd.concat(frames, ignore_index=True)
    return evaluate_positions(predictions, predictions["predicted"].to_numpy()), predictions


def backtest_baselines(df: pd.DataFrame, min_train_races: int = 40) -> dict[str, dict]:
    """Baselines scored on exactly the same races as the models."""
    splits = ds.walk_forward_splits(df, min_train_races=min_train_races)
    tested = pd.concat([test for _, test in splits], ignore_index=True)
    return {
        "baseline_grid": evaluate_positions(tested, ds.baseline_grid(tested)),
        "baseline_driver_form": evaluate_positions(tested, ds.baseline_driver_form(tested)),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def train_all(
    *, include_identity: bool = False, save: bool = True, min_train_races: int = 40
) -> dict:
    print("Loading features.race_features and attaching circuit degradation ...")
    df = ds.build_dataset(engine)
    if df.empty:
        raise RuntimeError("features.race_features is empty -- ingest races first")

    audit = ds.audit_leakage(df)
    print(f"\nLEAKAGE AUDIT: {'PASS' if audit['passes'] else 'FAIL'}")
    print(f"  teammates disagreeing on team_form_3 : {audit['teammate_team_form_disagreements']}")
    print(f"  first-ever races carrying form       : {audit['first_race_with_form']}")
    if not audit["passes"]:
        raise RuntimeError(f"leakage audit failed: {audit}")

    print(
        f"\n{len(df):,} driver-race rows | {df.event_id.nunique()} races | "
        f"seasons {sorted(df.season_year.unique())}"
    )
    print("This is a SMALL dataset. Overfitting it is effortless.")

    baselines = backtest_baselines(df, min_train_races)
    grid_mae = baselines["baseline_grid"]["mae_positions"]
    grid_rho = baselines["baseline_grid"]["spearman"]
    print(
        f"\nBASELINES (walk-forward, {baselines['baseline_grid']['n_races']} races)\n"
        f"  {'finish = grid order':<26} MAE {grid_mae:.3f}  rho {grid_rho:+.3f}  "
        f"top3 {baselines['baseline_grid']['top3_hit_rate']:.1%}\n"
        f"  {'finish = recent form':<26} "
        f"MAE {baselines['baseline_driver_form']['mae_positions']:.3f}  "
        f"rho {baselines['baseline_driver_form']['spearman']:+.3f}  "
        f"top3 {baselines['baseline_driver_form']['top3_hit_rate']:.1%}"
    )
    print("  Grid order is a STRONG baseline. Beating it is the whole task.")

    results: dict[str, dict] = {"baselines": baselines}
    predictions: dict[str, pd.DataFrame] = {}

    for label, builder in MODEL_BUILDERS.items():
        print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")
        metrics, preds = walk_forward_backtest(
            builder, df, include_identity=include_identity, min_train_races=min_train_races
        )
        predictions[label] = preds
        results[label] = metrics
        print(
            f"  walk-forward: MAE {metrics['mae_positions']:.3f}  "
            f"rho {metrics['spearman']:+.3f}  "
            f"top3 {metrics['top3_hit_rate']:.1%}  top1 {metrics['top1_hit_rate']:.1%}"
        )
        print(
            f"  vs grid baseline: MAE {grid_mae - metrics['mae_positions']:+.3f} "
            f"({1 - metrics['mae_positions'] / grid_mae:+.1%}), "
            f"rho {metrics['spearman'] - grid_rho:+.3f}"
        )

    # --- podium probability ---
    print(f"\n{'=' * 68}\npodium classifier\n{'=' * 68}")
    splits = ds.walk_forward_splits(df, min_train_races=min_train_races)
    podium_frames = []
    for train, test in splits:
        clf = build_podium_classifier(include_identity)
        clf.fit(
            ds.feature_matrix(train, include_identity),
            train[ds.TARGET_PODIUM].astype(int).to_numpy(),
        )
        proba = clf.predict_proba(ds.feature_matrix(test, include_identity))[:, 1]
        podium_frames.append(test.assign(podium_proba=proba))
    podium_preds = pd.concat(podium_frames, ignore_index=True)

    outcomes = podium_preds[ds.TARGET_PODIUM].astype(int).to_numpy()
    probabilities = podium_preds["podium_proba"].to_numpy()
    brier = float(brier_score_loss(outcomes, probabilities))
    base_rate = float(outcomes.mean())
    brier_baseline = float(np.mean((probabilities * 0 + base_rate - outcomes) ** 2))

    results["podium"] = {
        "brier": brier,
        "brier_base_rate": brier_baseline,
        "base_rate": base_rate,
        "n": int(len(podium_preds)),
    }
    print(f"  Brier {brier:.4f}  vs always-predict-{base_rate:.2f}: {brier_baseline:.4f}")
    print(f"  skill: {1 - brier / brier_baseline:+.1%}")
    print("\n  calibration (does '70%' mean 70%?):")
    table = calibration_table(probabilities, outcomes)
    for row in table.itertuples(index=False):
        print(
            f"    {str(row.bucket):<16} n={row.n:>4}  predicted {row.predicted:.2f}  "
            f"observed {row.observed:.2f}"
        )

    if save:
        version = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        best_label = min(MODEL_BUILDERS, key=lambda k: results[k]["mae_positions"])
        print(f"\nBest by walk-forward MAE: {best_label}")

        final = MODEL_BUILDERS[best_label](include_identity)
        final.fit(
            ds.feature_matrix(df, include_identity),
            df[ds.TARGET_POSITION].to_numpy(dtype=float),
        )
        final_podium = build_podium_classifier(include_identity)
        final_podium.fit(
            ds.feature_matrix(df, include_identity),
            df[ds.TARGET_PODIUM].astype(int).to_numpy(),
        )

        vocab = extract_categories(df, ds.IDENTITY_FEATURES)
        backend = (
            registry.BACKEND_XGBOOST if best_label == "xgboost" else registry.BACKEND_SKLEARN
        )
        with SessionLocal() as db:
            registry.save_model(
                db,
                name=MODEL_NAME,
                version=version,
                backend=backend,
                model=final,
                categories=vocab,
                feature_names=ds.feature_columns(include_identity),
                metrics={**results[best_label], "algorithm": best_label,
                         "baselines": baselines, "all_models": {
                             k: results[k] for k in MODEL_BUILDERS}},
                train_season_min=int(df.season_year.min()),
                train_season_max=int(df.season_year.max()),
                n_train_rows=len(df),
                notes=f"walk-forward backtest over {results[best_label]['n_races']} races",
            )
            registry.save_model(
                db,
                name=PODIUM_MODEL_NAME,
                version=version,
                backend=registry.BACKEND_SKLEARN,
                model=final_podium,
                categories=vocab,
                feature_names=ds.feature_columns(include_identity),
                metrics=results["podium"],
                train_season_min=int(df.season_year.min()),
                train_season_max=int(df.season_year.max()),
                n_train_rows=len(df),
                notes="logistic regression, P(podium)",
            )
            db.commit()
        print(f"Registered at version {version} (active).")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the race-prediction models.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--include-identity",
        action="store_true",
        help="One-hot driver/team/circuit. Adds ~78 columns to 2,556 rows.",
    )
    parser.add_argument("--min-train-races", type=int, default=40)
    args = parser.parse_args()
    train_all(
        include_identity=args.include_identity,
        save=not args.no_save,
        min_train_races=args.min_train_races,
    )


if __name__ == "__main__":
    main()
