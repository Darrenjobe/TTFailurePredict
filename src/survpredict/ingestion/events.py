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


def _get_path(row: dict, path: str):
    """Fetch a dotted NRQL attribute whether NR returned it flat or nested.

    NerdGraph sometimes emits ``{"entity.guid": "..."}`` and sometimes
    ``{"entity": {"guid": "..."}}``. Try flat first, then walk the path.
    """
    if path in row:
        return row[path]
    cur = row
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


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
        log.info("incidents_sample_row", keys=list(results[0].keys()), sample=results[0])

    # NrAiIncident yields one row per state transition (open/activated/close).
    # Dedupe in memory on (entity_guid, openedAt) so one logical incident lands
    # as a single events row regardless of which transitions NR returned.
    seen: set[tuple[str, int]] = set()
    inserted = 0
    skipped_no_guid = 0
    skipped_no_opened = 0
    with pg_cursor() as cur:
        for r in results:
            guid = _get_path(r, "entity.guid")
            opened = _get_path(r, "openedAt") or _get_path(r, "timestamp")
            if not guid:
                skipped_no_guid += 1
                continue
            if opened is None:
                skipped_no_opened += 1
                continue
            key = (guid, int(opened))
            if key in seen:
                continue
            seen.add(key)
            occurred_at = datetime.fromtimestamp(opened / 1000, tz=timezone.utc)
            klass = normalize_entity_class(_get_path(r, "entity.type") or "unknown")
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
        skipped_no_guid=skipped_no_guid,
        skipped_no_opened=skipped_no_opened,
    )
    return inserted
