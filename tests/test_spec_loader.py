"""Feature spec loading tests."""

from __future__ import annotations

from pathlib import Path

from survpredict.features.spec import load_feature_specs


def test_load_repo_specs():
    specs = load_feature_specs(Path(__file__).resolve().parents[1] / "feature_specs")
    names = {s.name for s in specs}
    assert "error_rate" in names
    assert "cpu_utilization_pct" in names
    assert "invocation_error_rate" in names


def test_derived_columns_unique():
    specs = load_feature_specs(Path(__file__).resolve().parents[1] / "feature_specs")
    cols: list[str] = []
    for s in specs:
        cols.extend(s.derived_column_names())
    assert len(cols) == len(set(cols))
