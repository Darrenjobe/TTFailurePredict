# `survpredict.publish`

Publishes predictions back into New Relic as `PredictedRisk` custom
events. This is what makes our output visible in NR dashboards,
workflows, and alerts.

> Design doc: [§11.2 Outputs back to New Relic](../../../docs/DESIGN.md#112-outputs-back-to-new-relic),
> [§11.3 Alert co-existence](../../../docs/DESIGN.md#113-alert-co-existence).

## Schema

```
PredictedRisk {
    entityGuid,
    modelVersion,
    timestamp,            // seconds since epoch
    hazardScore,
    survival5min, survival15min, survival30min, survival60min,
    topFeaturesJson,      // serialized list of [feature, shap_value]
    depRiskDelta,
    predictedFailureMode
}
```

## Why custom events (and not alerts)?

Custom events show up in NRDB and NR dashboards immediately, can be
queried with NRQL, and allow existing **alerting primitives** to be
built on top of our output without us owning the alert path:

```sql
SELECT count(*)
FROM PredictedRisk
WHERE survival15min < 0.7
FACET entityGuid
SINCE 5 minutes ago
```

That NRQL becomes an NR alert condition. We publish signal; NR owns
routing.

## Transport

HTTPS POST to the Insights Event API. Not NerdGraph.

- `Api-Key: <ingest license key>` — different from the NerdGraph user
  API key.
- Endpoint differs by region (US / EU) — resolved via
  `nr_settings().events_api_url`.
- Retries with exponential backoff on transient failures (3 attempts).

## Co-existence with threshold alerts (v1)

Predictions **do not replace** existing NR alert conditions in v1. They
run alongside. The explicit goal over time is that threshold alerts
graduate from *primary detection* to *confirmation signals* on top of
predictive risk — but that's an earned transition, not a flip.

## Extending

- **Batch publishing:** `publish_batch(results)` already exists; the
  warm sweep currently calls it per entity. If throughput matters,
  buffer per sweep and flush once.
- **Alternate sinks (Kafka, SNS):** add a new module in `publish/` that
  mirrors the function signature `publish_prediction(r)`. The scorer
  doesn't care which sink is wired.
