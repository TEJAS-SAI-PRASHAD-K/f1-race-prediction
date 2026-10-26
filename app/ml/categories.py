"""Pinned categorical vocabularies for XGBoost.

WHY THIS EXISTS
---------------
XGBoost's official docs are explicit: a category absent from training "cannot
be re-coded at inference", and training and prediction categorical dtypes must
match strictly or it raises. It is a HARD ERROR, not a fallback.

For this project that is not hypothetical. Predict a tire curve for a circuit
that joined the calendar after training, or for a rookie driver, and prediction
fails outright. The same problem bites race prediction harder, where every new
race may bring a debutant.

The fix is to pin the vocabulary at train time, persist it beside the model, and
map anything unseen to NaN at predict time. XGBoost handles NaN natively by
learning a default split direction, so an unknown circuit degrades to "the
average circuit" rather than crashing. That is a far better failure mode: a
slightly worse prediction instead of no prediction.

NOTE the second XGBoost gotcha, handled in app/ml/registry.py: a model with
categorical features MUST be serialised with `save_model()` to JSON. Pickling it
with joblib silently loses the category information, and the model then
mispredicts without ever raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def extract_categories(df: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    """Record the sorted, unique values of each categorical column.

    Sorted so the vocabulary is deterministic: the same training data must
    produce byte-identical category files, otherwise a model and its categories
    can silently disagree about which integer code means which circuit.
    """
    vocab: dict[str, list[str]] = {}
    for col in columns:
        values = df[col].dropna().astype(str).unique().tolist()
        vocab[col] = sorted(values)
    return vocab


def apply_categories(
    df: pd.DataFrame, vocab: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Re-apply a pinned vocabulary; unseen values become NaN.

    Returns the frame plus a count of how many rows were mapped to NaN per
    column, so a caller can WARN rather than silently degrade. Silent
    degradation is how a model ends up predicting the global average for an
    entire race weekend and nobody notices.
    """
    out = df.copy()
    unseen: dict[str, int] = {}
    for col, allowed in vocab.items():
        if col not in out.columns:
            continue
        as_str = out[col].astype(object).where(out[col].notna()).astype(object)
        is_known = as_str.isin(allowed)
        unseen[col] = int((~is_known & out[col].notna()).sum())
        out[col] = pd.Categorical(as_str.where(is_known), categories=allowed)
    return out, unseen


def save_categories(vocab: dict[str, list[str]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(vocab, indent=2, sort_keys=True))
    return p


def load_categories(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())
