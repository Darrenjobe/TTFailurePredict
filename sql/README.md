# `sql/`

The canonical schema. Loaded automatically on first `docker compose up`
via `/docker-entrypoint-initdb.d/`.

> Design doc: [§5 Data Model](../docs/DESIGN.md#5-data-model).

## Tables

| Table | Purpose |
|---|---|
| `entities` | Set of monitored objects (synced from NR). |
| `entity_edges` | Dependency graph with temporal validity. |
| `features` | Time-series feature values (Timescale hypertable). |
| `events` | Survival endpoints / labels. |
| `predictions` | Every scoring pass writes here (hypertable). |
| `postmortems` | Raw + structured narratives for label feedback. |
| `feature_proposals` | LLM-generated proposals awaiting human review. |
| `model_registry_meta` | MLflow mirror for fast production-model lookup. |
| `deployment_markers` | Change-point reference for "time since deploy" features. |

## Migrations

v1 uses a single idempotent `schema.sql`. When we start mutating schema
in production, we'll move to Alembic or sqitch and diff against the
last migration rather than editing `schema.sql` in place.

## Hypertables

- `features` partitioned on `computed_at`. Chunk interval defaults to
  7 days; revisit when we see real ingestion volume.
- `predictions` similarly partitioned. TTL/retention policy is not set
  yet — add it once we decide how far back we need evaluation replays.

## Constraints worth understanding

- `features` primary key includes `(entity_guid, feature_name,
  window_seconds, computed_at)`. Re-running an NRQL pull with the same
  timestamp is idempotent (upsert).
- `entity_edges` stores history via `(valid_from, valid_to)`. Current
  edges have `valid_to IS NULL`.
- `predictions.outcome_event_id` is a soft FK — populated by the
  reconciliation job, not at insert time.
