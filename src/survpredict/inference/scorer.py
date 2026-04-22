"""Scoring core.

Single entry point: ``score_entity(entity_guid)``. Used by both the hot path
(FastAPI endpoint) and the warm sweep. Loads the production model for the
entity's class from the registry, resolves features from Redis, and writes
predictions to Postgres + Redis.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from survpredict.common.db import pg_cursor, redis_client
from survpredict.common.logging import get_logger
from survpredict.common.time import utcnow
from survpredict.features.online_store import read_feature_vector
from survpredict.training.registry import load_production
from survpredict.training.rsf import predict_survival

log = get_logger(__name__)

HORIZONS_MIN = [5, 15, 30, 60]

_model_cache: dict[str, dict[str, Any]] = {}


@dataclass
class ScoreResult:
    prediction_id: str
    entity_guid: str
    model_version: str
    predicted_at: datetime
    hazard_score: float
    survival: dict[int, float]
    top_features: list[tuple[str, float]]
    dep_risk_delta: float
    predicted_failure_mode: str | None


def _lookup_entity_class(entity_guid: str) -> str | None:
    with pg_cursor() as cur:
        cur.execute(
            "SELECT entity_class FROM entities WHERE entity_guid = %s", (entity_guid,)
        )
        row = cur.fetchone()
    return row["entity_class"] if row else None


def _get_model(entity_class: str) -> dict[str, Any] | None:
    cached = _model_cache.get(entity_class)
    if cached is not None:
        return cached
    loaded = load_production(entity_class)
    if loaded is not None:
        _model_cache[entity_class] = loaded
    return loaded


def invalidate_model_cache(entity_class: str | None = None) -> None:
    if entity_class is None:
        _model_cache.clear()
    else:
        _model_cache.pop(entity_class, None)


def score_entity(entity_guid: str, explain: bool = True) -> ScoreResult | None:
    klass = _lookup_entity_class(entity_guid)
    if not klass:
        log.warning("unknown_entity", guid=entity_guid)
        return None

    model_bundle = _get_model(klass)
    if model_bundle is None:
        log.warning("no_production_model", entity_class=klass)
        return None

    trained = model_bundle["trained"]
    feature_cols = model_bundle["feature_columns"]
    features = read_feature_vector(entity_guid)
    if not features:
        return None

    X = pd.DataFrame([{c: features.get(c, 0.0) for c in feature_cols}])
    preds = predict_survival(trained, X, HORIZONS_MIN)

    hazard = float(preds["hazard"][0])
    survival = {h: float(preds[f"survival_{h}min"][0]) for h in HORIZONS_MIN}

    top_features: list[tuple[str, float]] = []
    if explain:
        from survpredict.inference.explain import top_feature_attributions

        top_features = top_feature_attributions(trained, X, k=8)

    dep_risk = _read_dep_risk(entity_guid)

    result = ScoreResult(
        prediction_id=str(uuid.uuid4()),
        entity_guid=entity_guid,
        model_version=model_bundle.get("version", "unknown"),
        predicted_at=utcnow(),
        hazard_score=hazard,
        survival=survival,
        top_features=top_features,
        dep_risk_delta=dep_risk,
        predicted_failure_mode=None,
    )
    _persist(result)
    return result


def _read_dep_risk(entity_guid: str) -> float:
    raw = redis_client().get(f"deprisk:{entity_guid}")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _persist(r: ScoreResult) -> None:
    with pg_cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions (prediction_id, entity_guid, model_version, predicted_at,
                hazard_score, survival_5min, survival_15min, survival_30min, survival_60min,
                top_features, dep_risk_delta, predicted_failure_mode)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                r.prediction_id,
                r.entity_guid,
                r.model_version,
                r.predicted_at,
                r.hazard_score,
                r.survival.get(5),
                r.survival.get(15),
                r.survival.get(30),
                r.survival.get(60),
                json.dumps([[f, float(v)] for f, v in r.top_features]),
                r.dep_risk_delta,
                r.predicted_failure_mode,
            ),
        )
    redis_client().hset(
        f"pred:{r.entity_guid}",
        mapping={
            "hazard": str(r.hazard_score),
            "survival_5": str(r.survival.get(5, 0)),
            "survival_15": str(r.survival.get(15, 0)),
            "survival_30": str(r.survival.get(30, 0)),
            "survival_60": str(r.survival.get(60, 0)),
            "model_version": r.model_version,
            "predicted_at": r.predicted_at.isoformat(),
        },
    )
    redis_client().expire(f"pred:{r.entity_guid}", 900)
