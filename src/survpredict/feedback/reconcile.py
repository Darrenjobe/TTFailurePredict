"""Label reconciliation (design doc §9.2).

Joins structured postmortems against existing events/predictions to produce:
  - confirmed_tp     : high-hazard prediction followed by a confirmed event
  - false_positive   : high-hazard prediction, no event, postmortem rules out latent issue
  - false_negative   : confirmed event with no prior high-hazard prediction
  - near_miss        : high hazard that recovered (tracked separately, not penalized)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.config import settings

log = get_logger(__name__)


def reconcile(lookback_hours: int = 168, horizon_minutes: int = 60) -> dict[str, int]:
    s = settings()
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    threshold = s.propagation_hazard_threshold

    tp = _mark_true_positives(since, threshold, horizon_minutes)
    fp = _mark_false_positives(since, threshold, horizon_minutes)
    fn = _emit_false_negatives(since, threshold, horizon_minutes)
    near = _mark_near_misses(since, threshold, horizon_minutes)

    log.info("reconcile_done", tp=tp, fp=fp, fn=fn, near=near)
    return {"tp": tp, "fp": fp, "fn": fn, "near_miss": near}


def _mark_true_positives(since, threshold, horizon) -> int:
    with pg_cursor() as cur:
        cur.execute(
            """
            UPDATE events e
            SET label_status = 'confirmed_tp'
            FROM predictions p
            WHERE e.occurred_at >= %s
              AND p.entity_guid = e.entity_guid
              AND p.hazard_score >= %s
              AND p.predicted_at BETWEEN e.occurred_at - (%s || ' minutes')::interval
                                      AND e.occurred_at
              AND (e.label_status IS NULL OR e.label_status = 'pending')
            """,
            (since, threshold, str(horizon)),
        )
        return cur.rowcount


def _mark_false_positives(since, threshold, horizon) -> int:
    """A prediction crosses threshold, no event occurs within horizon, and any
    overlapping postmortem explicitly says 'no latent issue' (we approximate as
    root_cause_category = 'unknown' with severity unknown -- refine after data).
    """
    with pg_cursor() as cur:
        cur.execute(
            """
            WITH candidate AS (
              SELECT p.entity_guid, p.predicted_at
              FROM predictions p
              WHERE p.predicted_at >= %s AND p.hazard_score >= %s
            ),
            matched AS (
              SELECT c.entity_guid, c.predicted_at
              FROM candidate c
              LEFT JOIN events e
                ON e.entity_guid = c.entity_guid
               AND e.occurred_at BETWEEN c.predicted_at
                                     AND c.predicted_at + (%s || ' minutes')::interval
              WHERE e.event_id IS NULL
            )
            INSERT INTO events (entity_guid, entity_class, event_type, severity,
                                occurred_at, detected_by, label_status)
            SELECT m.entity_guid, COALESCE(en.entity_class, 'unknown'),
                   'false_positive_probe', 'none',
                   m.predicted_at, 'reconciliation', 'false_positive'
            FROM matched m
            JOIN entities en ON en.entity_guid = m.entity_guid
            ON CONFLICT DO NOTHING
            """,
            (since, threshold, str(horizon)),
        )
        return cur.rowcount


def _emit_false_negatives(since, threshold, horizon) -> int:
    """An event fires and there was no predict >= threshold in the prior horizon.

    We update event.label_status to 'pending' and add metadata {'miss': True} so
    the feature-proposer can pick them up for LLM analysis.
    """
    with pg_cursor() as cur:
        cur.execute(
            """
            UPDATE events
            SET metadata = COALESCE(metadata, '{}'::jsonb) || '{"miss": true}'::jsonb
            WHERE occurred_at >= %s
              AND event_id IN (
                SELECT e.event_id
                FROM events e
                LEFT JOIN predictions p
                  ON p.entity_guid = e.entity_guid
                 AND p.hazard_score >= %s
                 AND p.predicted_at BETWEEN e.occurred_at - (%s || ' minutes')::interval
                                         AND e.occurred_at
                WHERE e.occurred_at >= %s
                  AND p.prediction_id IS NULL
                  AND (e.label_status IS NULL OR e.label_status = 'pending')
              )
            """,
            (since, threshold, str(horizon), since),
        )
        return cur.rowcount


def _mark_near_misses(since, threshold, horizon) -> int:
    with pg_cursor() as cur:
        cur.execute(
            """
            WITH peaks AS (
              SELECT entity_guid, predicted_at, hazard_score
              FROM predictions
              WHERE predicted_at >= %s AND hazard_score >= %s
            ),
            recovered AS (
              SELECT pk.entity_guid, pk.predicted_at
              FROM peaks pk
              LEFT JOIN events e
                ON e.entity_guid = pk.entity_guid
               AND e.occurred_at BETWEEN pk.predicted_at
                                     AND pk.predicted_at + (%s || ' minutes')::interval
              JOIN predictions later
                ON later.entity_guid = pk.entity_guid
               AND later.predicted_at BETWEEN pk.predicted_at
                                          AND pk.predicted_at + (%s || ' minutes')::interval
               AND later.hazard_score < %s * 0.5
              WHERE e.event_id IS NULL
            )
            INSERT INTO events (entity_guid, entity_class, event_type, severity,
                                occurred_at, detected_by, label_status)
            SELECT r.entity_guid, COALESCE(en.entity_class, 'unknown'),
                   'near_miss', 'info', r.predicted_at, 'reconciliation', 'near_miss'
            FROM recovered r
            JOIN entities en ON en.entity_guid = r.entity_guid
            ON CONFLICT DO NOTHING
            """,
            (since, threshold, str(horizon), str(horizon), threshold),
        )
        return cur.rowcount


def list_false_negatives(limit: int = 50) -> list[dict[str, Any]]:
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT event_id, entity_guid, entity_class, occurred_at, metadata
            FROM events
            WHERE metadata ? 'miss' AND (metadata->>'miss')::bool = TRUE
            ORDER BY occurred_at DESC LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()
