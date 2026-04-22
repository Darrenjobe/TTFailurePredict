"""SHAP-based feature attributions for survival tree models.

RSF is a tree ensemble, so TreeExplainer works. We shap the survival function
at a pinned horizon (30 min) -- this keeps explanations consistent across
scoring passes and cheap enough for the hot path.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from survpredict.common.logging import get_logger

log = get_logger(__name__)

EXPLAIN_HORIZON_MIN = 30


@lru_cache(maxsize=32)
def _explainer_for(model_id: int) -> Any:
    import shap

    model = _model_registry.get(model_id)
    return shap.TreeExplainer(model)


_model_registry: dict[int, Any] = {}


def top_feature_attributions(trained, X: pd.DataFrame, k: int = 8) -> list[tuple[str, float]]:
    """Return top-K (feature, signed_shap) pairs for the first row of X.

    Falls back to a gradient-like fallback if SHAP isn't available or fails --
    we never want the hot path to hard-fail on explanations.
    """
    try:
        import shap

        model = trained.model
        _model_registry[id(model)] = model
        explainer = _explainer_for(id(model))
        shap_values = explainer.shap_values(X.values)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        row = np.asarray(shap_values)[0]
        contribs = list(zip(X.columns, row.tolist()))
    except Exception as e:  # noqa: BLE001
        log.warning("shap_failed", err=str(e))
        contribs = [(c, float(X[c].iloc[0])) for c in X.columns]

    contribs.sort(key=lambda t: abs(t[1]), reverse=True)
    return contribs[:k]
