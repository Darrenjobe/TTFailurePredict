"""Scheduled NRQL puller.

Drives the v1 ingestion path (design doc §4.2). For each feature spec that
declares a ``source: nr_metric`` and an NRQL template, this module:

1. Resolves the set of entity GUIDs to query (from the ``entities`` table).
2. Substitutes the GUID and window into the NRQL template.
3. Writes raw results into the offline feature store.

The streaming export path is deliberately deferred; this puller is sufficient
for prototype-grade freshness (1 minute) and trivially reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.common.time import utcnow
from survpredict.features.spec import FeatureSpec, load_feature_specs
from survpredict.features.offline_store import write_feature_rows
from survpredict.ingestion.newrelic_client import NewRelicClient

log = get_logger(__name__)


@dataclass
class FeatureRow:
    entity_guid: str
    entity_class: str
    feature_name: str
    window_seconds: int
    value: float | None
    computed_at: datetime


def _entities_for_class(entity_class: str) -> list[tuple[str, str]]:
    with pg_cursor() as cur:
        cur.execute(
            "SELECT entity_guid, entity_class FROM entities WHERE entity_class = %s",
            (entity_class,),
        )
        return [(r["entity_guid"], r["entity_class"]) for r in cur.fetchall()]


def _render_nrql(template: str, guid: str, window_seconds: int, since: datetime) -> str:
    """Substitute placeholders. Templates use ``?`` for guid and ``:window`` / ``:since``."""
    q = template.replace("?", f"'{guid}'")
    q = q.replace(":window", f"{window_seconds} SECONDS")
    q = q.replace(":since", since.strftime("%Y-%m-%d %H:%M:%S"))
    if "SINCE" not in q.upper():
        q = f"{q} SINCE '{since.strftime('%Y-%m-%d %H:%M:%S')}'"
    return q


def pull_feature_for_entity(
    client: NewRelicClient,
    spec: FeatureSpec,
    entity_guid: str,
    entity_class: str,
    now: datetime,
) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    for window_seconds in spec.windows:
        since = now - timedelta(seconds=window_seconds)
        nrql = _render_nrql(spec.nrql or "", entity_guid, window_seconds, since)
        try:
            results = client.nrql(nrql)
        except Exception as e:  # retryable failures exhausted; skip and move on
            log.warning("nrql_failed", feature=spec.name, guid=entity_guid, err=str(e))
            continue
        value = _extract_scalar(results)
        rows.append(
            FeatureRow(
                entity_guid=entity_guid,
                entity_class=entity_class,
                feature_name=spec.name,
                window_seconds=window_seconds,
                value=value,
                computed_at=now,
            )
        )
    return rows


def _extract_scalar(results: list[dict]) -> float | None:
    """NRQL returns a list of dicts; collapse to a single scalar where possible."""
    if not results:
        return None
    first = results[-1]  # last time bucket wins for TIMESERIES queries
    for v in first.values():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def run_pull(entity_classes: Iterable[str] | None = None) -> int:
    """Pull all configured features for the given entity classes. Returns row count."""
    specs = load_feature_specs()
    nr_specs = [s for s in specs if s.source == "nr_metric"]
    log.info(
        "nrql_pull_start",
        total_specs=len(specs),
        nr_metric_specs=len(nr_specs),
        filter_classes=list(entity_classes) if entity_classes else "all",
    )
    now = utcnow()
    total = 0
    with NewRelicClient() as client:
        for spec in nr_specs:
            targets = spec.entity_classes
            if entity_classes is not None:
                targets = [c for c in targets if c in entity_classes]
            for klass in targets:
                entities = _entities_for_class(klass)
                log.info(
                    "nrql_pull_spec",
                    spec=spec.name,
                    entity_class=klass,
                    entity_count=len(entities),
                    windows=spec.windows,
                )
                for guid, _ in entities:
                    rows = pull_feature_for_entity(client, spec, guid, klass, now)
                    write_feature_rows(rows)
                    total += len(rows)
    log.info("nrql_pull_complete", rows_written=total)
    return total


# ---------------------------------------------------------------------------
# Historical backfill
# ---------------------------------------------------------------------------
# One NRQL call per (entity, feature) with TIMESERIES <bucket> SINCE N DAYS
# AGO. Every returned bucket becomes one feature row (per configured window).
# Makes a real training set possible without waiting for the puller to tick.


_TIMESERIES_RE = re.compile(r"TIMESERIES\s+\d+\s+\w+", re.IGNORECASE)


def _render_nrql_backfill(template: str, guid: str, days: int, bucket_minutes: int) -> str:
    q = template.replace("?", f"'{guid}'")
    q = _TIMESERIES_RE.sub(f"TIMESERIES {bucket_minutes} minutes", q)
    if "TIMESERIES" not in q.upper():
        q = f"{q} TIMESERIES {bucket_minutes} minutes"
    # Strip any existing SINCE clause and append our own.
    q = re.sub(r"SINCE\s+[^A-Z]*?(?=\s+(?:UNTIL|LIMIT|TIMESERIES|$))", "", q, flags=re.IGNORECASE)
    return f"{q.strip()} SINCE {days} days ago"


def _bucket_ts(bucket: dict) -> datetime | None:
    ts = bucket.get("beginTimeSeconds") or bucket.get("endTimeSeconds")
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def _bucket_value(bucket: dict) -> float | None:
    for k, v in bucket.items():
        if k in ("beginTimeSeconds", "endTimeSeconds", "inspectedCount"):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def backfill_feature_for_entity(
    client: NewRelicClient,
    spec: FeatureSpec,
    entity_guid: str,
    entity_class: str,
    days: int,
    bucket_minutes: int,
) -> list[FeatureRow]:
    nrql = _render_nrql_backfill(spec.nrql or "", entity_guid, days, bucket_minutes)
    try:
        results = client.nrql(nrql)
    except Exception as e:
        log.warning("backfill_nrql_failed", feature=spec.name, guid=entity_guid, err=str(e))
        return []

    rows: list[FeatureRow] = []
    for bucket in results:
        ts = _bucket_ts(bucket)
        if ts is None:
            continue
        value = _bucket_value(bucket)
        for window_seconds in spec.windows:
            rows.append(
                FeatureRow(
                    entity_guid=entity_guid,
                    entity_class=entity_class,
                    feature_name=spec.name,
                    window_seconds=window_seconds,
                    value=value,
                    computed_at=ts,
                )
            )
    return rows


def run_backfill(
    days: int = 7,
    bucket_minutes: int = 5,
    entity_classes: Iterable[str] | None = None,
) -> int:
    """Backfill historical feature values using NRQL TIMESERIES. Returns row count."""
    specs = load_feature_specs()
    nr_specs = [s for s in specs if s.source == "nr_metric"]
    log.info(
        "backfill_start",
        days=days,
        bucket_minutes=bucket_minutes,
        nr_metric_specs=len(nr_specs),
        filter_classes=list(entity_classes) if entity_classes else "all",
    )
    total = 0
    with NewRelicClient() as client:
        for spec in nr_specs:
            targets = spec.entity_classes
            if entity_classes is not None:
                targets = [c for c in targets if c in entity_classes]
            for klass in targets:
                entities = _entities_for_class(klass)
                log.info(
                    "backfill_spec",
                    spec=spec.name,
                    entity_class=klass,
                    entity_count=len(entities),
                )
                for guid, _ in entities:
                    rows = backfill_feature_for_entity(
                        client, spec, guid, klass, days, bucket_minutes
                    )
                    write_feature_rows(rows)
                    total += len(rows)
    log.info("backfill_complete", rows_written=total)
    return total


if __name__ == "__main__":
    run_pull()
