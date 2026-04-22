# `survpredict.training`

Training pipeline: dataset → models → eval → MLflow registry →
production promotion.

> Design doc: [§7 Modeling](../../../docs/DESIGN.md#7-modeling),
> [§9.4 Retraining](../../../docs/DESIGN.md#94-retraining).

## Contents

| File | Role |
|------|------|
| `rsf.py` | Random Survival Forest (primary model). |
| `cox.py` | Cox PH with optional L2 penalization (interpretability). |
| `evaluate.py` | C-index, IBS, time-dependent AUC, lead time, precision@K. |
| `registry.py` | MLflow logging + Postgres metadata mirror + promote(). |
| `pipeline.py` | End-to-end orchestration per entity class. |

## Why RSF as primary?

- Handles **non-proportional hazards** (which we get in abundance when
  deploys reshape the baseline hazard).
- Captures **feature interactions** without hand-authored crosses.
- Plays well with the SHAP tree explainer for per-prediction attributions.
- Mature, documented implementation in `scikit-survival`.

Cox PH runs alongside it as the interpretability partner. When the PH
assumption is meaningfully violated (which we detect in eval), the Cox
coefficients are still useful *relative* signals for explanations even
if absolute interpretation is weaker.

## Starting hyperparameters

```python
RandomSurvivalForest(
    n_estimators=500,
    min_samples_leaf=15,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)
```

Justification: design doc §7.1. We tune from there per entity class
once the first real dataset lands.

## Evaluation

- **C-index** — primary ranking metric. ≥0.75 on held-out is the v1
  success criterion.
- **Integrated Brier Score** — probabilistic calibration.
- **Time-dependent AUC** at 5/15/30/60 min horizons.
- **Lead-time distribution vs. threshold alerts** (`lead_time_distribution`)
  — the business metric: median ≥10 minutes lead.
- **Precision@K** — operational relevance of the top-K risky list.

## Promotion

New model versions are logged with `is_production=FALSE`. Promotion is a
distinct step:

```bash
survpredict train promote <model_version>
```

Per design doc §7.5: promote only after a 7-day shadow run that shows
C-index improvement ≥1% and no regression in lead time. The shadow
mechanism is intentionally manual in v1 — revisit once we have the
canary comparison harness.

## Registry mirror

MLflow is the source of truth for artifacts. The Postgres
`model_registry_meta` table is a read-optimized mirror so the inference
service can resolve "which model is production for this class?" without
a cross-service call on the hot path.

## Extending

- **New model family:** add a trainer module, log via `registry.log_model`
  with a new `algorithm` string. The inference side reads algorithm
  metadata to pick the right prediction function.
- **Custom eval metric:** extend `EvalReport` and log to MLflow; the
  mirror layer will pick up the metric if you add the column.
