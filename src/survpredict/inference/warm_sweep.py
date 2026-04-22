"""Warm-path fleet sweeper (design doc §8.2).

Runs continuously. Every `WARM_SWEEP_SECONDS` it:
  - Re-aggregates feature vectors for all active entities (from Redis set).
  - Scores every entity and persists the prediction.
  - Invokes graph propagation for any entity crossing the threshold.

Design target: <30s for 10k entities. We run scoring sequentially in v1 and
will move to a worker pool if profiling justifies it.
"""

from __future__ import annotations

import time

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.config import settings
from survpredict.features.aggregator import aggregate_for_entity
from survpredict.features.online_store import active_entities, drain_dirty
from survpredict.inference.propagation import propagate
from survpredict.inference.scorer import score_entity
from survpredict.publish.newrelic_events import publish_prediction

log = get_logger(__name__)


def list_all_entities() -> list[str]:
    with pg_cursor() as cur:
        cur.execute("SELECT entity_guid FROM entities")
        return [r["entity_guid"] for r in cur.fetchall()]


def one_sweep() -> dict[str, int]:
    dirty = set(drain_dirty()) | set(active_entities())
    if not dirty:
        dirty = set(list_all_entities())

    scored = 0
    propagated = 0
    published = 0
    started = time.perf_counter()

    for guid in dirty:
        aggregate_for_entity(guid)
        result = score_entity(guid, explain=False)
        if result is None:
            continue
        scored += 1
        if result.hazard_score >= settings().propagation_hazard_threshold:
            propagate(guid, result.hazard_score)
            propagated += 1
        try:
            publish_prediction(result)
            published += 1
        except Exception as e:
            log.warning("publish_failed", err=str(e))

    elapsed = time.perf_counter() - started
    log.info(
        "warm_sweep_done",
        scored=scored,
        propagated=propagated,
        published=published,
        seconds=round(elapsed, 2),
    )
    return {"scored": scored, "propagated": propagated, "published": published}


def run_forever() -> None:
    s = settings()
    while True:
        try:
            one_sweep()
        except Exception as e:  # keep the loop alive
            log.exception("warm_sweep_failed", err=str(e))
        time.sleep(s.warm_sweep_seconds)


if __name__ == "__main__":
    run_forever()
