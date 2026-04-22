"""Window-level transformations.

Given a time-ordered series of (timestamp, value) pairs for one (entity,
feature, window) triple, produce scalar feature values per transformation
declared in the spec.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np


def level(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values[-1])


def trend_slope(timestamps: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope per minute. NaN-safe."""
    if values.size < 2:
        return 0.0
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return 0.0
    t = timestamps[mask].astype("float64") / 60.0
    v = values[mask].astype("float64")
    t = t - t.mean()
    denom = (t * t).sum()
    if denom == 0:
        return 0.0
    return float((t * (v - v.mean())).sum() / denom)


def volatility_stddev(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.nanstd(values, ddof=1))


def volatility_mad(values: np.ndarray) -> float:
    """Median absolute deviation. Robust to outliers."""
    if values.size == 0:
        return 0.0
    med = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - med)))


def changepoint_distance(
    timestamps: np.ndarray, values: np.ndarray, k: float = 3.0
) -> float:
    """Seconds since the last CUSUM-style deviation exceeding k sigma.

    Simple, cheap approximation -- good enough as a training signal. Replace
    with ``ruptures`` or a proper Bayesian online change-point detector when
    the prototype matures.
    """
    if values.size < 5:
        return float(timestamps[-1] - timestamps[0]) if values.size else 0.0
    mu = np.nanmean(values)
    sigma = np.nanstd(values, ddof=1) or 1.0
    pos = 0.0
    neg = 0.0
    last_cp_idx = 0
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        pos = max(0.0, pos + (v - mu) - 0.5 * sigma)
        neg = min(0.0, neg + (v - mu) + 0.5 * sigma)
        if pos > k * sigma or neg < -k * sigma:
            last_cp_idx = i
            pos = neg = 0.0
    return float(timestamps[-1] - timestamps[last_cp_idx])


def time_since_last_deploy(now_ts: float, last_deploy_ts: float | None) -> float:
    if last_deploy_ts is None:
        return 86400.0 * 30  # 30 days, effectively "never" for feature purposes
    return max(0.0, now_ts - last_deploy_ts)


def cyclical_hour(ts: datetime) -> tuple[float, float]:
    angle = 2 * math.pi * (ts.hour + ts.minute / 60.0) / 24.0
    return math.sin(angle), math.cos(angle)


def cyclical_dow(ts: datetime) -> tuple[float, float]:
    angle = 2 * math.pi * ts.weekday() / 7.0
    return math.sin(angle), math.cos(angle)


TRANSFORMATIONS = {
    "level": level,
    "trend_slope": trend_slope,
    "volatility_stddev": volatility_stddev,
    "volatility_mad": volatility_mad,
    "changepoint_distance": changepoint_distance,
}
