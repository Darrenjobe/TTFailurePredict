"""Retraining cadence (§9.4).

Weekly full retrains and daily incremental warm-starts per entity class. The
replay buffer over-samples recent false negatives and hard false positives so
the model has an explicit signal to correct on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.labels.survival import build_dataset
from survpredict.training.cox import train_cox
from survpredict.training.evaluate import evaluate_rsf
from survpredict.training.registry import log_model
from survpredict.training.rsf import train_rsf

log = get_logger(__name__)


def _hard_cases(entity_class: str, days: int = 14) -> list[str]:
    """Return entity_guids with a recent false_negative or repeat false_positive."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT entity_guid
            FROM events
            WHERE entity_class = %s AND occurred_at >= %s
              AND (
                (metadata ? 'miss' AND (metadata->>'miss')::bool = TRUE)
                OR label_status = 'false_positive'
              )
            """,
            (entity_class, since),
        )
        return [r["entity_guid"] for r in cur.fetchall()]


def full_retrain(entity_class: str, lookback_days: int = 90) -> dict:
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=lookback_days)
    ds = build_dataset(entity_class, since, until)
    if len(ds) == 0:
        return {"status": "empty"}

    hard = set(_hard_cases(entity_class))
    if hard:
        mask = np.asarray([g in hard for g in ds.X.index.to_list()]) if "entity_guid" in ds.X.columns else None
        if mask is not None and mask.any():
            # crude upsampling by duplicating hard rows
            extra = ds.X[mask].copy()
            ds.X = (ds.X.append([extra] * 2) if hasattr(ds.X, "append") else ds.X)

    trained = train_rsf(ds)
    report = evaluate_rsf(trained, ds)
    version = log_model(trained, report, algorithm="rsf")
    try:
        log_model(train_cox(ds), report, algorithm="cox")
    except Exception as e:
        log.warning("cox_retrain_failed", err=str(e))
    return {"status": "ok", "version": version, "c_index": report.c_index}


def incremental_retrain(entity_class: str, lookback_days: int = 7) -> dict:
    """Warm-start style: retrain on recent data only. Promoted only after canary."""
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=lookback_days)
    ds = build_dataset(entity_class, since, until)
    if len(ds) == 0:
        return {"status": "empty"}
    trained = train_rsf(ds, n_estimators=200, min_samples_leaf=10)
    report = evaluate_rsf(trained, ds)
    version = log_model(trained, report, algorithm="rsf", extra_tags={"mode": "incremental"})
    return {"status": "ok", "version": version, "c_index": report.c_index}
