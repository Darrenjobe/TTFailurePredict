"""End-to-end training pipeline.

Wires dataset assembly -> RSF -> Cox -> eval -> registry. Runs per entity
class; intended to be scheduled weekly with a daily incremental warm-start
(design doc §9.4).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np

from survpredict.common.logging import get_logger
from survpredict.labels.survival import SurvivalDataset, build_dataset
from survpredict.training.cox import train_cox
from survpredict.training.evaluate import evaluate_rsf
from survpredict.training.registry import log_model
from survpredict.training.rsf import train_rsf

log = get_logger(__name__)


def _stratified_subsample(
    dataset: SurvivalDataset, max_samples: int, random_state: int = 42
) -> SurvivalDataset:
    """Cap dataset size while preserving every event=1 row.

    Survival models care about event-positive cases far more than censored
    ones; if we have to drop rows, drop censored. Keeps event_rate up too.
    """
    if len(dataset) <= max_samples:
        return dataset
    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(dataset.events == 1)
    neg_idx = np.flatnonzero(dataset.events == 0)
    n_pos = len(pos_idx)
    n_neg_target = max(0, max_samples - n_pos)
    if n_neg_target >= len(neg_idx):
        idx = np.concatenate([pos_idx, neg_idx])
    else:
        neg_keep = rng.choice(neg_idx, n_neg_target, replace=False)
        idx = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(idx)
    return SurvivalDataset(
        X=dataset.X.iloc[idx].reset_index(drop=True),
        durations=dataset.durations[idx],
        events=dataset.events[idx],
        feature_columns=dataset.feature_columns,
        entity_class=dataset.entity_class,
        created_at=dataset.created_at,
    )


def run_training(
    entity_class: str,
    lookback_days: int = 30,
    max_duration_minutes: int = 60,
    sample_every_minutes: int = 15,
    max_samples: int | None = 50_000,
    n_estimators: int = 200,
    train_cox_model: bool = False,
) -> dict:
    pipeline_t0 = time.perf_counter()
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=lookback_days)
    log.info(
        "training_start",
        entity_class=entity_class,
        since=since.isoformat(),
        until=until.isoformat(),
        sample_every_minutes=sample_every_minutes,
        max_samples=max_samples,
        n_estimators=n_estimators,
    )

    # ---- 1. dataset assembly ------------------------------------------------
    t0 = time.perf_counter()
    dataset = build_dataset(
        entity_class=entity_class,
        since=since,
        until=until,
        max_duration_minutes=max_duration_minutes,
        sample_every_minutes=sample_every_minutes,
    )
    log.info("stage_done", stage="build_dataset", seconds=round(time.perf_counter() - t0, 2))
    if len(dataset) == 0:
        return {"status": "empty_dataset"}

    # ---- 2. subsample if huge ----------------------------------------------
    if max_samples and len(dataset) > max_samples:
        t0 = time.perf_counter()
        before = len(dataset)
        dataset = _stratified_subsample(dataset, max_samples)
        log.info(
            "stage_done",
            stage="subsample",
            from_size=before,
            to_size=len(dataset),
            seconds=round(time.perf_counter() - t0, 2),
        )

    n_events = int(dataset.events.sum())
    log.info(
        "dataset_built",
        entity_class=entity_class,
        n=len(dataset),
        n_features=dataset.X.shape[1],
        n_events=n_events,
        event_rate=float(dataset.events.mean()),
    )
    if n_events == 0:
        return {
            "status": "all_censored",
            "entity_class": entity_class,
            "n": len(dataset),
            "hint": (
                "Features landed but no event hit a sampling grid point within "
                "max_duration_minutes. Check: (a) events table has rows for the "
                "same GUIDs as features; (b) event timestamps fall inside the "
                "backfill window; (c) events.entity_class or entities.entity_class "
                "matches the class being trained."
            ),
        }

    # ---- 3. fit RSF ---------------------------------------------------------
    t0 = time.perf_counter()
    trained_rsf = train_rsf(dataset, n_estimators=n_estimators)
    log.info("stage_done", stage="train_rsf", seconds=round(time.perf_counter() - t0, 2))

    # ---- 4. evaluate --------------------------------------------------------
    t0 = time.perf_counter()
    report = evaluate_rsf(trained_rsf, dataset)
    log.info(
        "stage_done",
        stage="evaluate_rsf",
        c_index=report.c_index,
        ibs=report.ibs,
        seconds=round(time.perf_counter() - t0, 2),
    )

    # ---- 5. log model -------------------------------------------------------
    t0 = time.perf_counter()
    rsf_version = log_model(trained_rsf, report, algorithm="rsf")
    log.info("stage_done", stage="log_model_rsf", version=rsf_version,
             seconds=round(time.perf_counter() - t0, 2))

    # ---- 6. (optional) Cox --------------------------------------------------
    cox_version = None
    if train_cox_model:
        t0 = time.perf_counter()
        try:
            trained_cox = train_cox(dataset)
            cox_version = log_model(trained_cox, report, algorithm="cox")
            log.info("stage_done", stage="train_cox", version=cox_version,
                     seconds=round(time.perf_counter() - t0, 2))
        except Exception as e:
            log.warning("cox_training_failed", err=str(e))

    log.info(
        "training_complete",
        entity_class=entity_class,
        rsf_version=rsf_version,
        cox_version=cox_version,
        total_seconds=round(time.perf_counter() - pipeline_t0, 2),
    )
    return {
        "status": "ok",
        "entity_class": entity_class,
        "n": len(dataset),
        "n_events": n_events,
        "event_rate": float(dataset.events.mean()),
        "rsf_version": rsf_version,
        "cox_version": cox_version,
        "c_index": report.c_index,
        "ibs": report.ibs,
    }
