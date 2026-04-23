"""Random Survival Forest trainer.

Primary model per design doc §7.1: handles non-proportional hazards, captures
interactions, works with mixed feature types out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from survpredict.common.logging import get_logger
from survpredict.labels.survival import SurvivalDataset

log = get_logger(__name__)


@dataclass
class TrainedRSF:
    model: RandomSurvivalForest
    feature_columns: list[str]
    entity_class: str
    trained_at: datetime
    training_size: int
    event_rate: float
    params: dict[str, Any] = field(default_factory=dict)


def train_rsf(
    dataset: SurvivalDataset,
    n_estimators: int = 200,
    min_samples_leaf: int = 15,
    max_features: str | float | None = "sqrt",
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1,
    use_float32: bool = True,
) -> TrainedRSF:
    if len(dataset) == 0:
        raise ValueError("empty training dataset")

    log.info(
        "rsf_fit_start",
        entity_class=dataset.entity_class,
        n=len(dataset),
        n_features=dataset.X.shape[1],
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        dtype="float32" if use_float32 else "float64",
    )

    dtype = np.float32 if use_float32 else np.float64
    X = dataset.X.values.astype(dtype, copy=False)
    y = Surv.from_arrays(event=dataset.events.astype(bool), time=dataset.durations)
    params = dict(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )
    model = RandomSurvivalForest(**params)
    model.fit(X, y)

    log.info(
        "rsf_trained",
        entity_class=dataset.entity_class,
        n=len(dataset),
        event_rate=float(dataset.events.mean()),
    )

    return TrainedRSF(
        model=model,
        feature_columns=list(dataset.X.columns),
        entity_class=dataset.entity_class,
        trained_at=datetime.now(timezone.utc),
        training_size=len(dataset),
        event_rate=float(dataset.events.mean()),
        params=params,
    )


def predict_survival(
    trained: TrainedRSF, X: pd.DataFrame, horizons_minutes: list[int]
) -> dict[str, np.ndarray]:
    """Return per-row hazard and survival probability at each horizon."""
    X_aligned = _align(X, trained.feature_columns)
    risk = trained.model.predict(X_aligned.values)
    surv_fns = trained.model.predict_survival_function(X_aligned.values, return_array=False)
    out: dict[str, np.ndarray] = {"hazard": np.asarray(risk, dtype="float64")}
    for h in horizons_minutes:
        out[f"survival_{h}min"] = np.asarray(
            [float(fn(h)) for fn in surv_fns], dtype="float64"
        )
    return out


def _align(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X[columns].astype("float64").fillna(0.0)
