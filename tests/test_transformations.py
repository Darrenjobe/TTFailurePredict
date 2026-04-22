"""Pure-function tests for window transformations."""

from __future__ import annotations

import numpy as np

from survpredict.features.transformations import (
    changepoint_distance,
    level,
    trend_slope,
    volatility_mad,
    volatility_stddev,
)


def test_level_returns_last_value():
    assert level(np.array([1.0, 2.0, 3.0])) == 3.0


def test_level_handles_empty():
    out = level(np.array([]))
    assert np.isnan(out)


def test_trend_slope_increasing():
    ts = np.arange(0, 600, 60, dtype="float64")  # 10 points over 10 minutes
    v = np.arange(10, dtype="float64")            # y = 0..9
    slope = trend_slope(ts, v)
    assert slope > 0


def test_trend_slope_flat():
    ts = np.arange(0, 600, 60, dtype="float64")
    v = np.full(10, 5.0)
    assert trend_slope(ts, v) == 0.0


def test_volatility_stddev_zero_for_constant():
    assert volatility_stddev(np.full(10, 3.0)) == 0.0


def test_volatility_mad_robust_to_outlier():
    v = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    assert volatility_mad(v) < 2.0


def test_changepoint_distance_detects_shift():
    ts = np.arange(60, dtype="float64")
    v = np.concatenate([np.zeros(30), np.full(30, 10.0)])
    d = changepoint_distance(ts, v, k=1.0)
    assert d < 60  # some distance less than the full window
