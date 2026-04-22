# `survpredict.ingestion`

The ingestion layer is the **front door** of the system. It pulls raw
telemetry, entity metadata, entity relationships, and operational
postmortems into our storage so everything downstream (features, labels,
models, feedback) can be deterministic.

> Design doc: [§4.2 Ingestion layer](../../../docs/DESIGN.md#ingestion-layer),
> [§11 New Relic integration](../../../docs/DESIGN.md#11-new-relic-integration-details).

## Contents

| File | Role |
|------|------|
| `newrelic_client.py` | Thin NerdGraph HTTP client with retry/backoff and NRQL helper. |
| `nrql_puller.py` | Scheduled NRQL puller that materializes feature raw values. |
| `entity_graph.py` | Entity search + relationship snapshotter (adjacency list). |
| `postmortem.py` | Markdown / Slack / Jira postmortem intake. |
| `events.py` | Pulls `NrAiIncident` into the `events` label table. |

## Data flow

```
NerdGraph ──► NewRelicClient ──► nrql_puller ──► features (Timescale)
                              ├─► entity_graph ─► entities + entity_edges
                              └─► events ───────► events (label rows)

local fs / slack / jira ──► postmortem ingester ──► postmortems (raw)
```

## v1 choices

- **NRQL polling** over streaming export for simplicity. The client is
  deliberately small so the streaming path can slot in later.
- **25-guid batches** for relationship pulls (NerdGraph cap).
- **SHA-256 fingerprint** on postmortem source_ref to avoid re-ingesting
  the same doc as its text changes but path remains.

## Usage

```bash
survpredict ingest entities APPLICATION      # sync APM apps
survpredict ingest nrql                      # pull feature raw values
survpredict ingest incidents                 # pull alert incidents as labels
survpredict ingest postmortems ./postmortems # pick up new .md files
```

## Extending

- **Add an entity class:** call `ingest entities <TYPE>` with a new NR
  entity type, add a YAML spec under `feature_specs/` scoped to the new
  `entity_classes` value.
- **Add a source:** implement an adapter in `postmortem.py` that writes
  to `postmortems` with a new `source` tag. The structurer in
  `feedback/` will pick it up without changes.
- **Move to streaming:** swap `nrql_puller.run_pull()` with a Kafka
  consumer that writes into the same `features` table. Nothing
  downstream needs to change.
