# `survpredict.feedback`

The feedback loop. Postmortems in, structured labels + feature proposals
+ retrained models out.

> Design doc: [§9 Feedback Loop](../../../docs/DESIGN.md#9-feedback-loop).

## Contents

| File | Role |
|------|------|
| `structurer.py` | Claude-driven postmortem → JSON structurer (controlled vocab). |
| `reconcile.py` | TP / FP / FN / near-miss label reconciliation. |
| `proposer.py` | LLM-proposed new features (human-reviewed queue). |
| `retrain.py` | Weekly full + daily incremental retraining with replay buffer. |

## What the LLM does — and doesn't

| Action | Allowed? |
|---|---|
| Extract structured fields from a postmortem | ✅ |
| Propose new YAML feature specs | ✅ (into `feature_proposals` queue) |
| Draft human-readable risk explanations | ✅ |
| Modify model weights | ❌ |
| Edit production feature specs | ❌ (proposal only, humans merge) |
| Change prediction thresholds | ❌ |
| Mark events as TP/FP directly | ❌ (reconciliation is deterministic SQL) |

Design doc §9.5 is the definitive rule. Every LLM output passes through
a deterministic gate (a DB constraint, a human review, or a code PR)
before affecting production behavior.

## Structurer

`structurer.py` posts the raw postmortem text to Claude with a schema
and a pinned vocabulary. The returned JSON lands in the `postmortems`
table. Model version is pinned (`ANTHROPIC_MODEL` env var) so label
stability across training rounds is preserved — bumping the version is
a deliberate operation.

```bash
survpredict feedback structure            # process N pending postmortems
```

## Reconciliation

`reconcile.py` joins `events` ⨝ `predictions` ⨝ `postmortems` under
four rules:

- **Confirmed TP** — event with a prior `hazard >= threshold` prediction
  within horizon.
- **False positive** — prediction above threshold, no event within
  horizon. Inserted as a `false_positive_probe` row for training
  accounting.
- **False negative** — event with no prior threshold prediction. Flagged
  via `events.metadata.miss = true` for the proposer to consume.
- **Near miss** — high-hazard prediction followed by recovery (hazard
  drops below half-threshold within horizon) and no event. Tracked,
  not penalized.

```bash
survpredict feedback reconcile            # run the pass
```

## Feature proposer

For every false negative, `proposer.py`:

1. Pulls the event, the postmortem (if any), and recent telemetry.
2. Sends them to Claude with the current feature-name list (so it
   doesn't re-propose things we already have).
3. Receives back a YAML block.
4. Inserts the YAML into `feature_proposals` with `status=pending`.

A human merges approved proposals into `feature_specs/` via PR. The
operating discipline is: no production `feature_specs/` file is ever
written by a machine.

```bash
survpredict feedback propose <event_id>
```

## Retraining cadence

- **Weekly full retrain** (§9.4) with updated labels across the 90-day
  lookback.
- **Daily incremental** warm-starts on the last 7 days.
- **Replay buffer** up-samples recent false negatives and chronic false
  positives.

```bash
survpredict feedback retrain apm.application              # incremental
survpredict feedback retrain apm.application --mode full  # full
```

## Extending

- **More detailed structured fields:** extend the JSON schema in
  `structurer.py`. Mirror the change in the `postmortems` column list
  or widen `structured` (JSONB) with the new keys.
- **More nuanced TP/FP logic:** the reconciliation SQL is intentionally
  deterministic and inspectable. Add rules there, not in the LLM.
