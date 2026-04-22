"""FastAPI inference service.

Endpoints:
  GET  /healthz
  GET  /score/{entity_guid}
  POST /score/batch          body: {"guids": [...]}
  GET  /top                  query: ?k=25&window=60
  POST /invalidate-cache     body: {"entity_class": "apm.application"}
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from survpredict.common.db import pg_cursor
from survpredict.common.logging import get_logger
from survpredict.inference.propagation import propagate
from survpredict.inference.scorer import ScoreResult, invalidate_model_cache, score_entity
from survpredict.publish.newrelic_events import publish_prediction

log = get_logger(__name__)
app = FastAPI(title="survpredict inference", version="0.1.0")


class ScoreResponse(BaseModel):
    entity_guid: str
    model_version: str
    predicted_at: datetime
    hazard_score: float
    survival_5min: float
    survival_15min: float
    survival_30min: float
    survival_60min: float
    top_features: list[tuple[str, float]]
    dep_risk_delta: float


def _to_response(r: ScoreResult) -> ScoreResponse:
    return ScoreResponse(
        entity_guid=r.entity_guid,
        model_version=r.model_version,
        predicted_at=r.predicted_at,
        hazard_score=r.hazard_score,
        survival_5min=r.survival.get(5, 0.0),
        survival_15min=r.survival.get(15, 0.0),
        survival_30min=r.survival.get(30, 0.0),
        survival_60min=r.survival.get(60, 0.0),
        top_features=[(f, float(v)) for f, v in r.top_features],
        dep_risk_delta=r.dep_risk_delta,
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/score/{entity_guid}", response_model=ScoreResponse)
def score(entity_guid: str) -> ScoreResponse:
    r = score_entity(entity_guid)
    if r is None:
        raise HTTPException(404, f"cannot score entity {entity_guid}")
    propagate(entity_guid, r.hazard_score)
    try:
        publish_prediction(r)
    except Exception as e:
        log.warning("nr_publish_failed", err=str(e))
    return _to_response(r)


class BatchRequest(BaseModel):
    guids: list[str]


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(req: BatchRequest) -> list[ScoreResponse]:
    out: list[ScoreResponse] = []
    for g in req.guids:
        r = score_entity(g)
        if r is None:
            continue
        propagate(g, r.hazard_score)
        try:
            publish_prediction(r)
        except Exception as e:
            log.warning("nr_publish_failed", err=str(e))
        out.append(_to_response(r))
    return out


@app.get("/top")
def top(k: int = 25, window: int = 60) -> list[dict[str, Any]]:
    """Top-K currently riskiest entities within the last `window` minutes."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)
    with pg_cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (entity_guid) entity_guid, predicted_at, hazard_score,
                   survival_5min, survival_15min, survival_30min, survival_60min,
                   top_features, dep_risk_delta, model_version
            FROM predictions
            WHERE predicted_at >= %s
            ORDER BY entity_guid, predicted_at DESC
            """,
            (cutoff,),
        )
        rows = cur.fetchall()
    rows.sort(key=lambda r: r["hazard_score"] or 0.0, reverse=True)
    return rows[:k]


class InvalidateRequest(BaseModel):
    entity_class: str | None = None


@app.post("/invalidate-cache")
def invalidate(req: InvalidateRequest) -> dict[str, str]:
    invalidate_model_cache(req.entity_class)
    return {"status": "ok"}
