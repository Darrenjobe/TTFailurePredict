"""Build (duration, event_indicator, covariates) tuples for survival models.

For each (entity, time-window), emit a training sample:
  - duration  -- minutes until next qualifying event, or censor time
  - event     -- 1 if an event occurred within the horizon, else 0
  - X         -- feature vector at the window's end timestamp

Right-censoring is handled naturally: entities whose observation window ends
before any event contribute as censored observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.features.spec import FeatureSpec, load_feature_specs

log = get_logger(__name__)


@dataclass
class SurvivalDataset:
    X: pd.DataFrame
    durations: np.ndarray
    events: np.ndarray
    feature_columns: list[str]
    entity_class: str
    created_at: datetime

    def __len__(self) -> int:
        return len(self.X)


def build_dataset(
    entity_class: str,
    since: datetime,
    until: datetime,
    max_duration_minutes: int = 60,
    sample_every_minutes: int = 5,
    specs: list[FeatureSpec] | None = None,
) -> SurvivalDataset:
    """Assemble a survival training set for one entity class.

    For each entity in the class, emit one sample every `sample_every_minutes`
    across [since, until]. The event horizon is `max_duration_minutes`; if no
    event occurs within it, the sample is censored at that cap.
    """
    specs = specs if specs is not None else load_feature_specs()
    feature_names = [s.name for s in specs if entity_class in s.entity_classes]
    log.info(
        "build_dataset_start",
        entity_class=entity_class,
        feature_names=feature_names,
        since=since.isoformat(),
        until=until.isoformat(),
    )

    entity_events = _load_events(entity_class, since - timedelta(minutes=max_duration_minutes), until)
    log.info("loading_feature_pivot")
    pivot = _load_feature_pivot(entity_class, feature_names, since, until)
    log.info(
        "feature_pivot_loaded",
        rows=len(pivot),
        entities=int(pivot["entity_guid"].nunique()) if not pivot.empty else 0,
        cols=int(pivot.shape[1]) if not pivot.empty else 0,
    )

    if pivot.empty:
        log.warning("empty_feature_pivot", entity_class=entity_class)
        return SurvivalDataset(
            X=pivot,
            durations=np.array([]),
            events=np.array([]),
            feature_columns=[],
            entity_class=entity_class,
            created_at=datetime.now(timezone.utc),
        )

    samples: list[dict] = []
    durations: list[float] = []
    events: list[int] = []
    step = timedelta(minutes=sample_every_minutes)

    n_entities = pivot["entity_guid"].nunique()
    log.info(
        "sampling_grid_start",
        entity_class=entity_class,
        entities=n_entities,
        step_minutes=sample_every_minutes,
        horizon_minutes=max_duration_minutes,
    )

    for i, (guid, df) in enumerate(pivot.groupby("entity_guid"), start=1):
        df = df.sort_values("computed_at").set_index("computed_at")
        ev_times = sorted(entity_events.get(guid, []))
        first = df.index.min()
        last = df.index.max()
        t = max(since, first)  # don't walk before any data exists for this entity
        while t <= until and t <= last + timedelta(minutes=max_duration_minutes):
            row = _snapshot_at(df, t)
            if row is None:
                t += step
                continue
            next_event = _next_event_after(ev_times, t)
            if next_event is None:
                dur = max_duration_minutes
                ev = 0
            else:
                delta_min = (next_event - t).total_seconds() / 60.0
                if delta_min <= max_duration_minutes:
                    dur = max(0.0, delta_min)
                    ev = 1
                else:
                    dur = max_duration_minutes
                    ev = 0
            samples.append({"entity_guid": guid, **row})
            durations.append(dur)
            events.append(ev)
            t += step

        if i % 10 == 0 or i == n_entities:
            log.info(
                "sampling_progress",
                done=i,
                of=n_entities,
                samples=len(samples),
                events_so_far=int(sum(events)),
            )

    X = pd.DataFrame(samples)
    fc = [c for c in X.columns if c != "entity_guid"]
    return SurvivalDataset(
        X=X[fc].astype("float64").fillna(0.0),
        durations=np.asarray(durations, dtype="float64"),
        events=np.asarray(events, dtype="int32"),
        feature_columns=fc,
        entity_class=entity_class,
        created_at=datetime.now(timezone.utc),
    )


def _load_events(entity_class: str, since: datetime, until: datetime) -> dict[str, list[datetime]]:
    """Return events for the class, using the entities table as the source of truth.

    Joining through entities avoids depending on events.entity_class, which may
    be stale (ingested before the normalize_entity_class mapping existed) or
    inconsistent with how features are partitioned.
    """
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT e.entity_guid, e.occurred_at
            FROM events e
            JOIN entities ent ON ent.entity_guid = e.entity_guid
            WHERE ent.entity_class = %s
              AND e.occurred_at BETWEEN %s AND %s
              AND (e.label_status IS NULL OR e.label_status != 'false_positive')
            """,
            (entity_class, since, until),
        )
        rows = cur.fetchall()
    out: dict[str, list[datetime]] = {}
    for r in rows:
        out.setdefault(r["entity_guid"], []).append(r["occurred_at"])
    log.info(
        "labels_events_loaded",
        entity_class=entity_class,
        events=sum(len(v) for v in out.values()),
        distinct_entities=len(out),
    )
    return out


def _load_feature_pivot(
    entity_class: str, feature_names: list[str], since: datetime, until: datetime
) -> pd.DataFrame:
    """Load features and pivot to one row per (entity, timestamp)."""
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT entity_guid, computed_at, feature_name, window_seconds, value
            FROM features
            WHERE entity_class = %s
              AND feature_name = ANY(%s)
              AND computed_at BETWEEN %s AND %s
            """,
            (entity_class, feature_names, since, until),
        )
        long_df = pd.DataFrame(cur.fetchall())
    if long_df.empty:
        return long_df
    long_df["col"] = long_df["feature_name"] + "__w" + long_df["window_seconds"].astype(str)
    wide = long_df.pivot_table(
        index=["entity_guid", "computed_at"], columns="col", values="value", aggfunc="last"
    ).reset_index()
    wide.columns.name = None
    return wide


def _snapshot_at(df: pd.DataFrame, t: datetime) -> dict | None:
    """Return the most recent row at or before `t`, or None if none exists."""
    try:
        idx = df.index.searchsorted(t, side="right") - 1
    except TypeError as e:
        # Almost certainly a tz-awareness mismatch; loud so it doesn't silently
        # produce empty datasets the way it did in the first cut.
        log.warning("snapshot_at_tz_error", err=str(e), t=str(t), index_tz=str(getattr(df.index, "tz", None)))
        return None
    if idx < 0:
        return None
    return df.iloc[idx].to_dict()


def _next_event_after(event_times: list[datetime], t: datetime) -> datetime | None:
    for et in event_times:
        if et > t:
            return et
    return None
