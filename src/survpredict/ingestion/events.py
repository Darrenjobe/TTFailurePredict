"""Event (label) ingestion from NR alert incidents.

Pulls NRQL-addressable signals (NrAiIncident, TransactionError, SystemSample
threshold crossings, etc.) and writes them as survival endpoints in the
``events`` table.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.common.time import utcnow
from survpredict.ingestion.newrelic_client import NewRelicClient

log = get_logger(__name__)

INCIDENTS_NRQL = (
    "SELECT incidentId, entity.guid, entity.type, priority, openedAt, closedAt "
    "FROM NrAiIncident WHERE entity.guid IS NOT NULL SINCE {since_minutes} MINUTES AGO"
)


def pull_incidents(since: datetime | None = None) -> int:
    since = since or (utcnow() - timedelta(hours=24))
    minutes = int((utcnow() - since).total_seconds() // 60)
    nrql = INCIDENTS_NRQL.format(since_minutes=max(minutes, 1))

    with NewRelicClient() as client:
        results = client.nrql(nrql)

    with pg_cursor() as cur:
        for r in results:
            guid = r.get("entity.guid")
            opened = r.get("openedAt")
            if not guid or not opened:
                continue
            occurred_at = datetime.fromtimestamp(opened / 1000, tz=datetime.now().astimezone().tzinfo)
            cur.execute(
                """
                INSERT INTO events (entity_guid, entity_class, event_type, severity,
                                    occurred_at, detected_by)
                VALUES (%s, %s, 'confirmed_incident', %s, %s, 'threshold_alert')
                ON CONFLICT DO NOTHING
                """,
                (guid, r.get("entity.type", "unknown"), r.get("priority"), occurred_at),
            )
    log.info("incidents_pulled", count=len(results))
    return len(results)
