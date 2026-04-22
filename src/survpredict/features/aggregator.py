"""Compile raw samples -> derived feature vectors per entity.

This is the bridge between "raw values from NRQL" and "features the model
eats". Runs on a schedule (e.g. every 60s) or ad-hoc when the puller writes
new samples.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np

from survpredict.common.logging import get_logger
from survpredict.common.time import utcnow
from survpredict.features.offline_store import recent_values
from survpredict.features.online_store import write_feature_vector
from survpredict.features.spec import FeatureSpec, load_feature_specs
from survpredict.features.transformations import (
    changepoint_distance,
    cyclical_dow,
    cyclical_hour,
    level,
    trend_slope,
    volatility_mad,
    volatility_stddev,
)

log = get_logger(__name__)


def aggregate_for_entity(
    entity_guid: str, specs: Iterable[FeatureSpec] | None = None
) -> dict[str, float]:
    specs = list(specs) if specs is not None else load_feature_specs()
    out: dict[str, float] = {}
    for spec in specs:
        for window in spec.windows:
            frame = recent_values(entity_guid, spec.name, window, lookback_seconds=window * 2)
            if frame.empty:
                continue
            ts = frame["computed_at"].astype("int64").to_numpy() // 10**9
            v = frame["value"].to_numpy(dtype="float64")
            for transform in spec.transformations:
                col = f"{spec.name}__w{window}__{transform}"
                out[col] = _apply(transform, ts, v)

    now = utcnow()
    s_h, c_h = cyclical_hour(now)
    s_d, c_d = cyclical_dow(now)
    out["time__hour_sin"] = s_h
    out["time__hour_cos"] = c_h
    out["time__dow_sin"] = s_d
    out["time__dow_cos"] = c_d

    write_feature_vector(entity_guid, out)
    return out


def _apply(transform: str, ts: np.ndarray, v: np.ndarray) -> float:
    if transform == "level":
        return level(v)
    if transform == "trend_slope":
        return trend_slope(ts, v)
    if transform == "volatility_stddev":
        return volatility_stddev(v)
    if transform == "volatility_mad":
        return volatility_mad(v)
    if transform == "changepoint_distance":
        return changepoint_distance(ts, v)
    return float("nan")
