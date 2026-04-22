# `survpredict.inference`

Produces per-entity hazard scores and publishes them. Two paths:

- **Hot path** — FastAPI endpoints (`service.py`). Triggered by feature
  updates or on-demand requests. Target p99 < 500 ms.
- **Warm path** — periodic sweep (`warm_sweep.py`). Every
  `WARM_SWEEP_SECONDS` (default 60). Target: <30 s for 10k entities.

> Design doc: [§8 Inference Path](../../../docs/DESIGN.md#8-inference-path).

## Contents

| File | Role |
|------|------|
| `scorer.py` | Single `score_entity()` entry point used by both paths. |
| `service.py` | FastAPI app with `/score/*`, `/top`, `/healthz`, `/invalidate-cache`. |
| `warm_sweep.py` | Continuous fleet sweeper. |
| `explain.py` | SHAP feature attributions for tree survival models. |
| `propagation.py` | Graph-hop risk propagation with rate limiting. |

## Endpoints

```
GET  /healthz
GET  /score/{entity_guid}          # score a single entity
POST /score/batch                  # body: {"guids": [...]}
GET  /top?k=25&window=60           # top-K riskiest over the last `window` minutes
POST /invalidate-cache             # drop cached model
```

Every scored entity:

1. Reads its feature vector from Redis (`fvec:{guid}`).
2. Aligns columns to the production model for its class.
3. Predicts hazard and survival at `[5, 15, 30, 60]` minute horizons.
4. Generates SHAP top-K attributions (best-effort — never fails the path).
5. Writes a row into `predictions` and updates a Redis summary.
6. If above the propagation threshold, walks `entity_edges` up to N hops.
7. Publishes a `PredictedRisk` custom event back to New Relic.

## Model caching

Models are loaded lazily on first score for an entity class and cached
in-process (`_model_cache`). After a training promotion, call
`/invalidate-cache` (or let a restart do it) to pick up the new version.

## Graph propagation

`propagation.py` walks downstream edges (`src_guid = this_guid`) when
the current hazard ≥ `PROPAGATION_HAZARD_THRESHOLD` (default 0.6). It:

- Dampens contribution by `0.6^hop` to avoid turning the whole graph red.
- Caps total propagation to 500 events/minute via a Redis counter — this
  is the "prevent propagation storms" knob.
- Marks affected entities dirty so the next warm sweep re-scores them.

## Explanations

SHAP's `TreeExplainer` wraps the RSF ensemble. We shap the survival
function at a pinned 30-minute horizon for stability across sweeps.
Attribution failures log a warning and fall back to raw feature values —
the scorer does not raise.

## Performance notes

- FastAPI + uvicorn is fine for v1. The hot path allocates a single-row
  DataFrame and calls `predict` once — dominant cost is SHAP, which the
  LRU-cached explainer mitigates.
- If/when p99 pressure shows up, the move is to rewrite the hot path in
  Rust or Go and keep the warm sweep in Python.

## Extending

- **New horizon:** edit `HORIZONS_MIN` in `scorer.py` and add a column
  to `predictions` + the NR payload.
- **Different propagation policy:** swap the damping schedule in
  `propagation._bump_dep_risk`.
- **gRPC transport:** add a new server in `service.py` next to FastAPI;
  share the `score_entity` core.
