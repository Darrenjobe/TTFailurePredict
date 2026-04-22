# `survpredict.features`

Feature engineering. Takes raw values produced by ingestion and turns
them into the covariate vectors that feed the survival models.

> Design doc: [§6 Feature Engineering](../../../docs/DESIGN.md#6-feature-engineering),
> [§5.1 Feature schema](../../../docs/DESIGN.md#51-feature-schema-online--offline).

## Contents

| File | Role |
|------|------|
| `spec.py` | Loader for YAML feature specs in `feature_specs/`. |
| `transformations.py` | Pure window transforms: level, trend, volatility, change-point, cyclical time. |
| `offline_store.py` | Timescale-backed long-form feature rows (for training). |
| `online_store.py` | Redis-backed derived vectors + dirty-entity set (for hot-path scoring). |
| `aggregator.py` | Compiles raw samples → derived feature vector per entity. |

## Spec format

```yaml
- name: error_rate
  description: "Fraction of transactions with error=true"
  entity_classes: [apm.application]
  source: nr_metric
  nrql: "SELECT percentage(count(*), WHERE error IS true) FROM Transaction WHERE entity.guid = ? TIMESERIES 1 minute"
  windows: [60, 300, 900]
  transformations: [level, trend_slope, volatility_stddev]
```

- `?` is substituted with the current `entity_guid`.
- `source: nr_metric` → the NRQL puller materializes values.
- `source: derived` → the aggregator computes the value from other
  sources (e.g. upstream hazard weighted sum).

## Derived column naming

After aggregation, each feature expands to one column per
`(window_seconds, transformation)` pair. Example:

```
error_rate__w60__level
error_rate__w60__trend_slope
error_rate__w60__volatility_stddev
error_rate__w300__level
```

Training and inference agree on names by calling
`FeatureSpec.derived_column_names()`.

## Online vs. offline store

| | Online (Redis) | Offline (Timescale) |
|---|---|---|
| Latency target | sub-ms | seconds |
| Retention | hours | 90+ days |
| Readers | hot path inference | training, analytics, reconciliation |
| Writers | aggregator, propagation | NRQL puller |

## Why the two-layer design

Hot-path scoring can't afford a SQL roundtrip per feature. The aggregator
precomputes the derived vector once per sweep (or on a feature-change
event) and stashes it in a single Redis HASH. The scorer does one
`HGETALL`, pivots to a DataFrame, and predicts.

Training needs the long history, joins across entities, and analytic
queries that Redis can't serve — that's what the Timescale hypertable is
for.

## Extending

- **New transform:** add a pure function to `transformations.py` and map
  it in the `TRANSFORMATIONS` dict. Declare it in any spec.
- **New source type:** add a branch to the aggregator (or writer in
  ingestion) and document it in spec format.
- **New entity class:** add a YAML file, enumerate
  `entity_classes: [<new_class>]`, ingest entities of that type.
