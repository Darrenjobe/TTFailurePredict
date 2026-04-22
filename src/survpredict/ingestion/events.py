"""Event (label) ingestion from NR alert incidents.

Pulls NRQL-addressable signals (NrAiIncident, TransactionError, SystemSample
threshold crossings, etc.) and writes them as survival endpoints in the
``events`` table.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.common.time import utcnow
from survpredict.ingestion.entity_graph import normalize_entity_class
from survpredict.ingestion.newrelic_client import NewRelicClient

log = get_logger(__name__)


def _build_nrql(
    since_minutes: int,
    entity_type: str | None,
    limit: int,
) -> str:
    type_clause = f" AND entity.type = '{entity_type}'" if entity_type else ""
    return (
        "SELECT incidentId, entity.guid, entity.type, priority, "
        "openedAt, closedAt, event "
        f"FROM NrAiIncident WHERE entity.guid IS NOT NULL{type_clause} "
        f"SINCE {max(since_minutes, 1)} MINUTES AGO "
        f"LIMIT {limit}"
    )


def pull_incidents(
    since: datetime | None = None,
    entity_type: str | None = "APPLICATION",
    limit: int = 5000,
) -> int:
    """Pull NR alert incidents into the events table.

    Args:
      since: datetime lower bound (default: 24h ago).
      entity_type: NR entityType filter. Defaults to APPLICATION (APM apps).
        Pass None to pull incidents for any entity type.
      limit: NRQL LIMIT. NR caps at 5000 per query; increase lookback or
        run multiple times if you're brushing that ceiling.
    """
    since = since or (utcnow() - timedelta(hours=24))
    minutes = int((utcnow() - since).total_seconds() // 60)
    nrql = _build_nrql(minutes, entity_type, limit)
    log.info("incidents_pull_start", lookback_minutes=minutes, entity_type=entity_type, limit=limit)

    with NewRelicClient() as client:
        results = client.nrql(nrql)

    if results:
        log.debug("incidents_sample_row", keys=list(results[0].keys()))

    # NrAiIncident yields one row per state transition (open/activated/close).
    # Dedupe in memory on (entity_guid, openedAt) so one logical incident lands
    # as a single events row regardless of which transitions NR returned.
    seen: set[tuple[str, int]] = set()
    inserted = 0
    with pg_cursor() as cur:
        for r in results:
            guid = r.get("entity.guid")
            opened = r.get("openedAt")
            if not guid or opened is None:
                continue
            key = (guid, int(opened))
            if key in seen:
                continue
            seen.add(key)
            occurred_at = datetime.fromtimestamp(opened / 1000, tz=timezone.utc)
            klass = normalize_entity_class(r.get("entity.type", "unknown"))
            cur.execute(
                """
                INSERT INTO events (entity_guid, entity_class, event_type, severity,
                                    occurred_at, detected_by)
                VALUES (%s, %s, 'confirmed_incident', %s, %s, 'threshold_alert')
                ON CONFLICT DO NOTHING
                """,
                (guid, klass, r.get("priority"), occurred_at),
            )
            inserted += cur.rowcount
    log.info(
        "incidents_pulled",
        returned=len(results),
        unique_incidents=len(seen),
        inserted=inserted,
    )
    return inserted
