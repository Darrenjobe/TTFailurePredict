"""Evaluation metrics.

Design doc §7.4:
  - C-index (concordance)    -- ranking quality
  - IBS (integrated Brier)   -- probabilistic calibration
  - Time-dependent AUC       -- discrimination at specific horizons
  - Lead-time distribution   -- versus existing threshold alerts on the same events
  - Precision@K              -- operational relevance of the top-K risky entities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class EvalReport:
    c_index: float
    ibs: float | None
    time_auc: dict[int, float] = field(default_factory=dict)
    n: int = 0
    event_rate: float = 0.0

    def as_dict(self) -> dict:
        return {
            "c_index": self.c_index,
            "ibs": self.ibs,
            "time_auc": self.time_auc,
            "n": self.n,
            "event_rate": self.event_rate,
        }


def evaluate_rsf(trained, dataset, horizons_minutes=(5, 15, 30, 60)) -> EvalReport:
    from survpredict.training.rsf import predict_survival

    y = Surv.from_arrays(event=dataset.events.astype(bool), time=dataset.durations)

    risk = trained.model.predict(dataset.X.values)
    c_idx, _, _, _, _ = concordance_index_censored(
        event_indicator=dataset.events.astype(bool),
        event_time=dataset.durations,
        estimate=risk,
    )

    horizons = np.asarray([float(h) for h in horizons_minutes])
    try:
        pred = predict_survival(trained, dataset.X, list(horizons_minutes))
        # survival probabilities -> cumulative hazard surrogate for AUC
        estimates = np.column_stack([1 - pred[f"survival_{h}min"] for h in horizons_minutes])
        auc, _ = cumulative_dynamic_auc(y, y, estimates, horizons)
        time_auc = {int(h): float(auc[i]) for i, h in enumerate(horizons_minutes)}
    except Exception as e:  # valid for small evals where auc cannot be computed
        log.warning("time_auc_failed", err=str(e))
        time_auc = {}

    try:
        # Build a survival probability matrix aligned with horizons
        surv_matrix = np.column_stack([pred[f"survival_{h}min"] for h in horizons_minutes])
        ibs = float(integrated_brier_score(y, y, surv_matrix, horizons))
    except Exception as e:
        log.warning("ibs_failed", err=str(e))
        ibs = None

    return EvalReport(
        c_index=float(c_idx),
        ibs=ibs,
        time_auc=time_auc,
        n=len(dataset),
        event_rate=float(dataset.events.mean()),
    )


def lead_time_distribution(
    entity_class: str,
    hazard_threshold: float,
    lookback_hours: int = 168,
) -> pd.DataFrame:
    """For each real event, compute the earliest prior prediction that crossed threshold.

    Returns a DataFrame with columns: event_id, occurred_at, first_warning_at,
    lead_minutes. Events where no prior prediction crossed the threshold get
    NaN lead_minutes (interpreted as a miss).
    """
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT e.event_id, e.entity_guid, e.occurred_at,
                   MIN(p.predicted_at) FILTER (WHERE p.hazard_score >= %s) AS first_warning_at
            FROM events e
            LEFT JOIN predictions p
              ON p.entity_guid = e.entity_guid
             AND p.predicted_at BETWEEN e.occurred_at - interval '60 minutes' AND e.occurred_at
            WHERE e.entity_class = %s AND e.occurred_at >= %s
            GROUP BY e.event_id, e.entity_guid, e.occurred_at
            ORDER BY e.occurred_at DESC
            """,
            (hazard_threshold, entity_class, since),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["lead_minutes"] = (
        (df["occurred_at"] - df["first_warning_at"]).dt.total_seconds() / 60.0
    )
    return df


def precision_at_k(k: int = 10, horizon_minutes: int = 60) -> float:
    """Of the top-K highest-hazard entities, what fraction had an event within the horizon?"""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=horizon_minutes)
    with pg_cursor() as cur:
        cur.execute(
            """
            WITH ranked AS (
              SELECT DISTINCT ON (entity_guid) entity_guid, predicted_at, hazard_score
              FROM predictions
              WHERE predicted_at >= %s
              ORDER BY entity_guid, predicted_at DESC
            ),
            topk AS (
              SELECT entity_guid, predicted_at FROM ranked
              ORDER BY hazard_score DESC LIMIT %s
            )
            SELECT count(*) FILTER (WHERE e.event_id IS NOT NULL)::float / NULLIF(count(*), 0)
                AS precision
            FROM topk t
            LEFT JOIN events e
              ON e.entity_guid = t.entity_guid
             AND e.occurred_at BETWEEN t.predicted_at AND t.predicted_at + interval '%s minutes'
            """,
            (cutoff, k, horizon_minutes),
        )
        row = cur.fetchone()
    return float(row["precision"] or 0.0)
