"""Publish ``PredictedRisk`` custom events to New Relic.

Uses the Event API (not NerdGraph) -- it is a simple HTTP POST with the
ingest license key. Predictions are batched where possible.
"""

from __future__ import annotations

import json
from typing import Iterable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from survpredict.common.logging import get_logger
from survpredict.config import nr_settings
from survpredict.inference.scorer import ScoreResult

log = get_logger(__name__)

EVENT_TYPE = "PredictedRisk"


def _to_payload(r: ScoreResult) -> dict:
    return {
        "eventType": EVENT_TYPE,
        "entityGuid": r.entity_guid,
        "modelVersion": r.model_version,
        "timestamp": int(r.predicted_at.timestamp()),
        "hazardScore": r.hazard_score,
        "survival5min": r.survival.get(5, 0.0),
        "survival15min": r.survival.get(15, 0.0),
        "survival30min": r.survival.get(30, 0.0),
        "survival60min": r.survival.get(60, 0.0),
        "topFeaturesJson": json.dumps([[f, float(v)] for f, v in r.top_features]),
        "depRiskDelta": r.dep_risk_delta,
        "predictedFailureMode": r.predicted_failure_mode or "unknown",
    }


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
def publish_prediction(r: ScoreResult) -> None:
    publish_batch([r])


def publish_batch(results: Iterable[ScoreResult]) -> int:
    s = nr_settings()
    if not s.ingest_license_key or not s.account_id:
        log.debug("publish_skipped_no_nr_key")
        return 0

    payloads = [_to_payload(r) for r in results]
    if not payloads:
        return 0

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(
            s.events_api_url,
            headers={
                "Api-Key": s.ingest_license_key,
                "Content-Type": "application/json",
            },
            json=payloads,
        )
        resp.raise_for_status()
    return len(payloads)
