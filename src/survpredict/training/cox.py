"""Cox PH (time-varying) trainer -- interpretability partner to the RSF.

Used to generate human-readable explanations (§7.2). Runs on the same dataset
as the RSF, in parallel. When the proportional hazards assumption is badly
violated, the coefficients are still useful relative signals even if absolute
interpretation weakens.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from lifelines import CoxPHFitter

from survpredict.common.logging import get_logger
from survpredict.labels.survival import SurvivalDataset

log = get_logger(__name__)


@dataclass
class TrainedCox:
    model: CoxPHFitter
    feature_columns: list[str]
    entity_class: str
    trained_at: datetime
    training_size: int


def train_cox(
    dataset: SurvivalDataset,
    penalizer: float = 0.01,
    l1_ratio: float = 0.0,
) -> TrainedCox:
    if len(dataset) == 0:
        raise ValueError("empty training dataset")

    frame = dataset.X.copy()
    frame["_duration"] = dataset.durations
    frame["_event"] = dataset.events
    frame = frame.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    variances = frame[dataset.feature_columns].var()
    keep = variances[variances > 0].index.tolist()
    frame = frame[[*keep, "_duration", "_event"]]

    fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    fitter.fit(frame, duration_col="_duration", event_col="_event", robust=True)

    log.info("cox_trained", entity_class=dataset.entity_class, features=len(keep))

    return TrainedCox(
        model=fitter,
        feature_columns=keep,
        entity_class=dataset.entity_class,
        trained_at=datetime.utcnow(),
        training_size=len(dataset),
    )


def explain_row(trained: TrainedCox, row: pd.Series) -> list[tuple[str, float]]:
    """Return sorted (feature, partial_hazard_contribution) attributions."""
    coefs = trained.model.params_
    contributions: list[tuple[str, float]] = []
    for feat, beta in coefs.items():
        x = float(row.get(feat, 0.0))
        contributions.append((feat, float(beta * x)))
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)
    return contributions
