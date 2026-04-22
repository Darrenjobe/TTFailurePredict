"""End-to-end training pipeline.

Wires dataset assembly -> RSF -> Cox -> eval -> registry. Runs per entity
class; intended to be scheduled weekly with a daily incremental warm-start
(design doc §9.4).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from survpredict.common.logging import get_logger
from survpredict.labels.survival import build_dataset
from survpredict.training.cox import train_cox
from survpredict.training.evaluate import evaluate_rsf
from survpredict.training.registry import log_model
from survpredict.training.rsf import train_rsf

log = get_logger(__name__)


def run_training(
    entity_class: str,
    lookback_days: int = 90,
    max_duration_minutes: int = 60,
    sample_every_minutes: int = 5,
    train_cox_model: bool = True,
) -> dict:
    until = datetime.utcnow()
    since = until - timedelta(days=lookback_days)
    log.info("training_start", entity_class=entity_class, since=since.isoformat())

    dataset = build_dataset(
        entity_class=entity_class,
        since=since,
        until=until,
        max_duration_minutes=max_duration_minutes,
        sample_every_minutes=sample_every_minutes,
    )
    if len(dataset) == 0:
        return {"status": "empty_dataset"}

    trained_rsf = train_rsf(dataset)
    report = evaluate_rsf(trained_rsf, dataset)
    rsf_version = log_model(trained_rsf, report, algorithm="rsf")

    cox_version = None
    if train_cox_model:
        try:
            trained_cox = train_cox(dataset)
            cox_version = log_model(trained_cox, report, algorithm="cox")
        except Exception as e:
            log.warning("cox_training_failed", err=str(e))

    return {
        "status": "ok",
        "entity_class": entity_class,
        "n": len(dataset),
        "event_rate": float(dataset.events.mean()),
        "rsf_version": rsf_version,
        "cox_version": cox_version,
        "c_index": report.c_index,
        "ibs": report.ibs,
    }
