"""merge duplicate circuit rows

FastF1 reports `Location` inconsistently across seasons, so one physical track
ended up as two rows in raw.circuits:

    "Monte Carlo" (2019, 2021)  and  "Monaco" (2022-2024)
    "Singapore"   (2019)        and  "Marina Bay" (2023-2024)

That fragments every per-circuit statistic. Monaco had its 62 green-flag stops
split into two samples of 34 and 28, giving two different answers (35.0s and
21.1s) for one pit lane -- and the race-prediction features partition on
circuit name, so a split circuit silently splits its own history.

The ingestion fix lives in app/ingestion/transform.py (CIRCUIT_NAME_ALIASES), so
new ingests are canonical. This migration repairs the rows already stored, by
repointing events at the surviving circuit and deleting the orphan.

NOTE: `circuit` is a model feature, so its categorical vocabulary changes.
Retrain after this runs:

    python -m app.ml.prediction.train

Revision ID: 819c4ea913e1
Revises: e455366cd113
"""

from collections.abc import Sequence

from alembic import op

revision: str = "819c4ea913e1"
down_revision: str | None = "e455366cd113"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# alias -> canonical. Must mirror transform.CIRCUIT_NAME_ALIASES.
MERGES = [("Monte Carlo", "Monaco"), ("Monte-Carlo", "Monaco"), ("Singapore", "Marina Bay")]


def upgrade() -> None:
    for alias, canonical in MERGES:
        # Repoint events only when BOTH rows exist; if the canonical row is
        # absent, just rename the alias row so no data is stranded.
        op.execute(
            f"""
            UPDATE raw.events e
               SET circuit_id = canon.id
              FROM raw.circuits canon, raw.circuits dup
             WHERE canon.name = '{canonical}'
               AND dup.name   = '{alias}'
               AND e.circuit_id = dup.id
            """
        )
        op.execute(
            f"""
            UPDATE raw.circuits
               SET name = '{canonical}'
             WHERE name = '{alias}'
               AND NOT EXISTS (SELECT 1 FROM raw.circuits c2 WHERE c2.name = '{canonical}')
            """
        )
        # Now unreferenced.
        op.execute(
            f"""
            DELETE FROM raw.circuits
             WHERE name = '{alias}'
               AND NOT EXISTS (SELECT 1 FROM raw.events e WHERE e.circuit_id = raw.circuits.id)
            """
        )


def downgrade() -> None:
    # Not reversible: which events used which spelling is not recoverable once
    # merged. Re-ingesting the affected seasons would restore the raw values.
    pass
