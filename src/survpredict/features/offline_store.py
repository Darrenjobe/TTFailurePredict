"""Offline feature store (TimescaleDB).

Every materialized feature value lands here; training reads from here. Hot-path
inference reads from the Redis online store.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Sequence

import pandas as pd
from psycopg import sql

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger

log = get_logger(__name__)


def write_feature_rows(rows: Iterable) -> int:
    """Bulk-insert feature rows. Accepts FeatureRow-like duck-typed objects."""
    rows = list(rows)
    if not rows:
        return 0
    with pg_cursor() as cur:
        cur.executemany(
            """
            INSERT INTO features (entity_guid, entity_class, feature_name,
                                  window_seconds, value, computed_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_guid, feature_name, window_seconds, computed_at)
            DO UPDATE SET value = EXCLUDED.value
            """,
            [
                (r.entity_guid, r.entity_class, r.feature_name,
                 r.window_seconds, r.value, r.computed_at)
                for r in rows
            ],
        )
    return len(rows)


def load_feature_matrix(
    entity_class: str,
    feature_names: Sequence[str],
    since: datetime,
    until: datetime,
) -> pd.DataFrame:
    """Pull a long-format feature frame for training.

    Returns columns: entity_guid, computed_at, feature_name, window_seconds, value.
    The training dataset builder then pivots this into wide form.
    """
    query = sql.SQL(
        """
        SELECT entity_guid, computed_at, feature_name, window_seconds, value
        FROM features
        WHERE entity_class = %s
          AND feature_name = ANY(%s)
          AND computed_at BETWEEN %s AND %s
        ORDER BY entity_guid, computed_at
        """
    )
    with pg_cursor() as cur:
        cur.execute(query, (entity_class, list(feature_names), since, until))
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def recent_values(
    entity_guid: str, feature_name: str, window_seconds: int, lookback_seconds: int
) -> pd.DataFrame:
    """Latest raw samples for a (guid, feature, window) over the last `lookback_seconds`."""
    since = datetime.utcnow() - timedelta(seconds=lookback_seconds)
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT computed_at, value
            FROM features
            WHERE entity_guid = %s AND feature_name = %s AND window_seconds = %s
              AND computed_at >= %s
            ORDER BY computed_at
            """,
            (entity_guid, feature_name, window_seconds, since),
        )
        return pd.DataFrame(cur.fetchall())
