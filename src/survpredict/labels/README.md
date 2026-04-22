# `survpredict.labels`

Turns events + feature history into **right-censored survival
observations**.

> Design doc: [§3.2 Survival formulation](../../../docs/DESIGN.md#32-survival-formulation),
> [§5.2 Events table](../../../docs/DESIGN.md#52-events-table-survival-labels).

## What the builder emits

For each `(entity, timestamp)` sampling grid point:

```python
SurvivalDataset(
    X=<feature vectors>,          # pandas DataFrame
    durations=<minutes_to_event>, # numpy array
    events=<1 if event, else 0>,  # numpy array
    feature_columns=[...],
    entity_class="apm.application",
)
```

- If no qualifying event occurs within `max_duration_minutes`, the row is
  **right-censored** at the cap (`event=0`, `duration=cap`).
- If an event occurs at time `t` > grid point `g`, `duration = t - g` and
  `event=1`.

## Why right-censored > binary classification

A binary classifier that says "will this entity fail in the next 15
minutes?" discards every observation that *hasn't* failed — which is
almost all of them. The survival framing keeps them as censored and
learns from both the timing of failures and the non-failures.
One fit gives you `S(t | X)` for any horizon, which is why you can read
`survival_5min`, `survival_15min`, `survival_30min`, `survival_60min`
off the same model.

## Sampling strategy

- `sample_every_minutes=5` by default: every entity contributes a sample
  once every 5 minutes over the lookback window. Overlap is fine — RSF
  handles correlated samples through bootstrapping.
- The grid is per-entity; we do not synchronize across entities.

## Event filtering

- Rows where `events.label_status = 'false_positive'` are excluded from
  the training set (they are feedback, not ground truth).
- Rows flagged `metadata ? 'miss'` (false negatives) remain but are
  up-weighted by the retraining replay buffer.

## Extending

- **New event types:** insert into `events` with a new `event_type`
  value; the builder consumes them by default.
- **Different horizon:** pass `max_duration_minutes` when calling
  `build_dataset`. Nothing else in the pipeline needs to change.
