# Survival Analysis for Predictive Observability

## Design Document — Prototype Specification

**Status:** Draft v0.1
**Owner:** Darren
**Purpose:** Design doc for prototyping a survival-analysis-based predictive observability system driven by New Relic telemetry. Intended as initial context for Claude Code-assisted implementation.

---

## 1. Problem Statement

Traditional threshold-based alerting in observability platforms is fundamentally reactive: alerts fire *after* a metric crosses a line, meaning the failure condition has largely already arrived. For complex SaaS platforms running on heterogeneous infrastructure (Kubernetes, Lambda, cloud-native services, third-party APIs), this reactive posture produces:

- High MTTD (mean time to detect) for degradation-class incidents
- Alert fatigue from threshold tuning across hundreds of entity types
- Poor signal propagation across service dependencies
- No principled way to prioritize risk across the fleet

**Goal:** Build a system that uses survival analysis on streaming observability data to estimate the hazard rate (instantaneous failure probability) for each monitored entity, producing predictive risk scores that surface degradation-class failures *before* traditional threshold alerts fire.

**Target outcome:** 10–30 minute predictive lead time on degradation-class incidents with precision/recall that improves over time via postmortem feedback loops.

---

## 2. Scope

### In scope (v1 prototype)

- Ingestion of New Relic metrics, events, logs, and traces for a defined entity set
- Feature engineering pipeline producing time-varying covariates per entity
- Training pipeline for Random Survival Forests (primary) and Cox PH (interpretability)
- Streaming inference producing per-entity hazard scores
- Entity relationship graph ingestion from New Relic for dependency-aware risk propagation
- Postmortem ingestion pipeline with LLM-assisted structuring
- Feedback loop for model retraining based on postmortem labels
- Basic UI / API for surfacing top-risk entities with explanations

### Out of scope (v1)

- Automated remediation actions
- Multi-tenant isolation
- Replacement of existing alerting (predictions run alongside, not instead of)
- Real-time deep learning survival models (DeepSurv/DeepHit) — deferred to v2
- Cross-customer federated learning

---

## 3. Core Concepts

### 3.1 Entity

A monitored object with a stable identifier (New Relic GUID). Examples: a Kubernetes pod, a Lambda function, an APM application, a host, a database instance, a queue.

Each entity has a **class** (e.g., `k8s.pod`, `aws.lambda`, `apm.application`) that determines which model variant is used for scoring.

### 3.2 Survival formulation

For each entity, we model the time until a **failure event**. Failure events include:

- SLO violation (error rate, latency, availability breach)
- Resource saturation crossing a defined threshold (for labeled training)
- Alert-confirmed incident (validated via postmortem)
- Health-state transition (e.g., pod OOMKilled, Lambda throttled)

Observations are **right-censored**: an entity that has not yet failed by the end of its observation window contributes information up to that point.

We model the **hazard function** `h(t | X(t))` where `X(t)` is the vector of time-varying covariates (telemetry features) at time `t`.

### 3.3 Prediction output

For each entity, at each scoring interval, the system produces:

- `hazard_score` — instantaneous hazard rate at the current time
- `survival_prob_window` — P(survives next N minutes) for configurable N (5, 15, 30, 60 min)
- `top_contributing_features` — ranked feature attributions (SHAP for tree models, partial hazards for Cox)
- `dependency_risk_delta` — risk contribution from upstream entities via the service graph
- `predicted_failure_mode` — classification of likely failure type (from LLM + historical pattern match)

---

## 4. Architecture

### 4.1 System diagram (logical)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NEW RELIC PLATFORM                          │
│   (Metrics, Events, Logs, Traces, Entity Relationships, NRQL)       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ Streaming Export / NRDB API
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INGESTION LAYER                                │
│  - NR streaming consumer                                            │
│  - NRQL scheduled pullers (for aggregates)                          │
│  - Entity relationship snapshotter                                  │
│  - Postmortem ingester (Slack / Jira / markdown)                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  FEATURE STORE (Redis + Postgres)                   │
│  - Online: Redis, keyed by entity_guid, sliding windows             │
│  - Offline: Postgres (TimescaleDB), full history for training       │
└────────┬───────────────────────────────────┬────────────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────────┐            ┌─────────────────────────────────┐
│   TRAINING PIPELINE │            │    INFERENCE SERVICE            │
│   (batch, Python)   │            │    (FastAPI + gRPC)             │
│  - Label assembly   │            │  - Hot path: event-triggered    │
│  - Model fitting    │            │  - Warm path: periodic sweep    │
│  - Eval + canary    │            │  - Graph propagation            │
│  - Registry push    │            │  - Explanation generation       │
└──────────┬──────────┘            └──────────┬──────────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐            ┌─────────────────────────────────┐
│   MODEL REGISTRY    │────────────▶│    PREDICTION STORE             │
│   (MLflow / S3)     │            │    (Postgres + Redis)           │
└─────────────────────┘            └──────────┬──────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PRESENTATION + FEEDBACK LAYER                     │
│  - Risk dashboard (top-N entities by hazard)                        │
│  - Alert enrichment (sent back to NR as custom events)              │
│  - Postmortem capture UI                                            │
│  - LLM-assisted feature proposal + label structuring                │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component responsibilities

#### Ingestion layer
- **Primary source:** New Relic streaming export to Kafka, or direct NRDB polling via NerdGraph/NRQL for v1 simplicity
- **Entity relationship snapshotter:** runs every 5 minutes, pulls the full entity relationship graph, stores as a Postgres-backed adjacency list with temporal validity
- **Postmortem ingester:** watches a configured source (Slack channel, Jira project, markdown directory) and triggers LLM structuring on new entries

#### Feature store
- **Online (Redis):** last N=60 windowed aggregations per entity per feature, used for sub-second inference
- **Offline (TimescaleDB):** full historical feature table for training, partitioned by entity_class and time
- **Feature definitions:** declared as YAML specs, compiled into both NRQL (for backfill) and streaming aggregation logic

#### Training pipeline
- **Language:** Python 3.11+
- **Libraries:** `scikit-survival`, `lifelines`, `xgboost` (survival objective), `pycox` (deferred), `shap`
- **Execution:** runs on DGX Spark for RSF on large feature sets; CPU-fine for Cox PH
- **Cadence:** weekly full retrain, daily incremental on new events
- **Per-class models:** separate model artifacts for each entity class, stored in MLflow

#### Inference service
- **Language:** Python (FastAPI) for v1, with option to migrate hot path to Rust/Go in v2
- **Hot path:** triggered by feature-change events, target <100ms p99
- **Warm path:** full-fleet sweep every 60s, target <30s total for 10K entities
- **Graph propagation:** when an entity's hazard crosses a configurable threshold, recompute hazards for all entities within N hops downstream

#### Prediction store
- **Postgres:** long-term prediction history for evaluation and feedback
- **Redis:** current hazard scores, TTL-bounded

#### Presentation + feedback
- **Risk dashboard:** simple React or Streamlit UI showing top-N entities by hazard, explanations, and dependency context
- **NR integration:** predictions published back to New Relic as custom events, enabling NR-native dashboards and workflow integration
- **Postmortem capture:** structured form (LLM-assisted) producing training labels

---

## 5. Data Model

### 5.1 Feature schema (online + offline)

```sql
CREATE TABLE features (
    entity_guid      TEXT NOT NULL,
    entity_class     TEXT NOT NULL,
    feature_name     TEXT NOT NULL,
    window_seconds   INT  NOT NULL,
    value            DOUBLE PRECISION,
    computed_at      TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (entity_guid, feature_name, window_seconds, computed_at)
);
SELECT create_hypertable('features', 'computed_at');
```

### 5.2 Events table (survival labels)

```sql
CREATE TABLE events (
    event_id         UUID PRIMARY KEY,
    entity_guid      TEXT NOT NULL,
    entity_class     TEXT NOT NULL,
    event_type       TEXT NOT NULL,  -- 'slo_breach', 'oom', 'throttle', 'confirmed_incident', etc.
    severity         TEXT,
    occurred_at      TIMESTAMPTZ NOT NULL,
    detected_by      TEXT,           -- 'threshold_alert', 'postmortem', 'health_transition'
    postmortem_id    UUID,           -- nullable FK
    label_status     TEXT            -- 'pending', 'confirmed_tp', 'false_positive', 'near_miss'
);
```

### 5.3 Entity relationships

```sql
CREATE TABLE entity_edges (
    src_guid         TEXT NOT NULL,
    dst_guid         TEXT NOT NULL,
    relationship     TEXT NOT NULL,  -- 'calls', 'contains', 'depends_on', etc.
    valid_from       TIMESTAMPTZ NOT NULL,
    valid_to         TIMESTAMPTZ,
    PRIMARY KEY (src_guid, dst_guid, relationship, valid_from)
);
```

### 5.4 Predictions

```sql
CREATE TABLE predictions (
    prediction_id    UUID PRIMARY KEY,
    entity_guid      TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    predicted_at     TIMESTAMPTZ NOT NULL,
    hazard_score     DOUBLE PRECISION,
    survival_5min    DOUBLE PRECISION,
    survival_15min   DOUBLE PRECISION,
    survival_30min   DOUBLE PRECISION,
    survival_60min   DOUBLE PRECISION,
    top_features     JSONB,          -- ranked feature attributions
    dep_risk_delta   DOUBLE PRECISION,
    outcome_event_id UUID            -- populated later if an event occurs within horizon
);
```

### 5.5 Postmortems

```sql
CREATE TABLE postmortems (
    postmortem_id    UUID PRIMARY KEY,
    source           TEXT,           -- 'slack', 'jira', 'markdown'
    raw_text         TEXT,
    structured       JSONB,          -- LLM-produced structured form
    entities_affected TEXT[],
    root_cause_category TEXT,
    event_start      TIMESTAMPTZ,
    event_end        TIMESTAMPTZ,
    contributing_signals TEXT[],
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 6. Feature Engineering

### 6.1 Feature families (starting set)

For each entity class, features are computed over multiple windows (1m, 5m, 15m, 1h) and include:

**Level features:** current value of the metric (cpu %, error rate, latency p99, queue depth, etc.)

**Trend features:** slope of linear fit over window

**Volatility features:** stddev, IQR, MAD over window

**Change-point features:** distance since last significant change (CUSUM or Bayesian change-point)

**Cross-entity features:** hazard-weighted aggregate of upstream dependency risk

**Temporal features:** time-of-day, day-of-week encodings (cyclical)

**Deployment features:** time since last deploy (from NR deployment markers)

**Trace-derived features:** ratio of distributed trace error rate, upstream latency contribution

### 6.2 Feature spec format (YAML)

```yaml
- name: cpu_utilization_pct
  entity_classes: [k8s.pod, host, apm.application]
  source: nr_metric
  nrql: "SELECT average(cpuPercent) FROM SystemSample WHERE entityGuid = ? TIMESERIES 1 minute"
  windows: [60, 300, 900, 3600]
  transformations: [level, trend_slope, volatility_stddev]

- name: error_rate
  entity_classes: [apm.application, aws.lambda]
  source: nr_metric
  nrql: "SELECT percentage(count(*), WHERE error IS true) FROM Transaction WHERE entityGuid = ? TIMESERIES 1 minute"
  windows: [60, 300, 900]
  transformations: [level, trend_slope]

- name: upstream_hazard_weighted
  entity_classes: [apm.application]
  source: derived
  computation: |
    SUM over upstream entities of (edge_weight * upstream.hazard_score)
```

---

## 7. Modeling

### 7.1 Primary model: Random Survival Forest

- **Why:** handles non-proportional hazards, captures feature interactions, provides built-in feature importances, robust to mixed feature types
- **Library:** `scikit-survival` (`RandomSurvivalForest`)
- **Training:** per-entity-class, right-censored observations over rolling 90-day windows
- **Hyperparameters (starting):** `n_estimators=500`, `min_samples_leaf=15`, `max_features='sqrt'`

### 7.2 Interpretability model: Cox PH with time-varying covariates

- **Why:** explicit coefficient interpretation, well-understood by stats-literate stakeholders
- **Library:** `lifelines` (`CoxTimeVaryingFitter`)
- **Usage:** run in parallel with RSF on the same data; use for generating human-readable risk explanations

### 7.3 Deferred (v2): Neural survival models

- **DeepSurv / DeepHit** via `pycox` for entities with high-dimensional features (e.g., log/trace embeddings)
- Training on DGX Spark

### 7.4 Evaluation metrics

- **Concordance index (C-index)** — primary ranking metric
- **Integrated Brier Score (IBS)** — probabilistic calibration
- **Time-dependent AUC** at 5/15/30/60 min horizons
- **Lead time distribution** vs. threshold alerts on the same events
- **Precision@K** for top-K risky entities per time window

### 7.5 Canary + promotion

- New model versions run in shadow mode for 7 days
- Promoted only if C-index improves by ≥1% AND no regression in lead-time-vs-alert

---

## 8. Inference Path

### 8.1 Hot path (event-driven)

1. Feature-change event arrives (e.g., new metric sample materially shifts a window aggregate)
2. Feature vector assembled from Redis online store
3. Model loaded from in-process cache
4. Score computed, SHAP values computed
5. If hazard crosses propagation threshold, enqueue downstream entities for re-scoring
6. Result written to Redis + Postgres, published as NR custom event

### 8.2 Warm path (periodic)

- Runs every 60s
- Scores all entities whose features have been touched since last sweep
- Primary mechanism for catching slow-moving degradation in entities without recent events

### 8.3 Graph propagation

- When `hazard_score` crosses threshold (default: 0.6), walk downstream edges up to N hops (default: 3)
- Recompute `upstream_hazard_weighted` feature for each affected entity
- Re-score those entities (capped by a rate limiter to prevent propagation storms)

---

## 9. Feedback Loop

### 9.1 Postmortem pipeline

1. New postmortem ingested (raw text)
2. LLM pass produces structured form:
   - Event start / end timestamps
   - Affected entity GUIDs (resolved from NR entity names)
   - Root cause category (from a controlled vocabulary)
   - Contributing signals (feature names, if recognizable)
   - Human impact severity
3. Structured form written to `postmortems` table
4. Matching `events` records updated with `postmortem_id` and `label_status`

### 9.2 Label reconciliation

- **True positive:** prediction with hazard >threshold within lookback window AND confirmed event
- **False positive:** prediction with hazard >threshold AND no event within forward window AND postmortem confirms no latent issue
- **False negative:** confirmed event with no prior high-hazard prediction
- **Near miss:** high hazard that recovered without event (kept separately, not penalized)

### 9.3 LLM-assisted feature proposal

For every **false negative**, an LLM pass is triggered:
- Input: postmortem narrative, feature set available at prediction time, recent telemetry for affected entities
- Output: proposed new features (as YAML specs), proposed new entity-relationship edges, proposed new event types
- **Proposals are NOT auto-applied.** They land in a review queue.

### 9.4 Retraining

- **Weekly full retrain** with updated labels
- **Daily incremental retrain** on new events only (warm-start from latest model)
- **Replay buffer** over-samples recent false negatives and hard false positives
- **Per-class model refresh** — no global retrain across entity classes

### 9.5 What the LLM never does directly

- Never modifies model weights
- Never changes feature spec YAML in the production repo without PR review
- Never changes prediction thresholds
- All LLM outputs are proposals or enrichments with a human or deterministic gate before production impact

---

## 10. Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Ingestion | Python + Kafka (or direct NRQL polling v1) | Simplicity for prototype; migrate to streaming export for scale |
| Feature store (online) | Redis | Sub-ms lookup, widely understood |
| Feature store (offline) | TimescaleDB | Native time-series, SQL compatibility |
| Training | Python, scikit-survival, lifelines, xgboost | Mature survival analysis ecosystem |
| Inference service | Python FastAPI | Fastest prototype path; acceptable latency for v1 |
| Model registry | MLflow | Standard, integrates with S3 |
| Orchestration | Prefect or Dagster | Modern, Python-native |
| LLM integration | Claude via Anthropic API | Structured output, long context for postmortems |
| Presentation | Streamlit (v1), React (v2) | Fast iteration |
| Deployment | Docker Compose (v1), Kubernetes (v2) | Minimize prototype ops burden |

---

## 11. New Relic Integration Details

### 11.1 Data sources

- **NRQL** via NerdGraph API for historical backfill and scheduled aggregates
- **Streaming exports** (Kafka/Kinesis) for real-time where available
- **Entity relationship API** (NerdGraph) for dependency graph
- **Deployment markers API** for change-point features

### 11.2 Outputs back to New Relic

- Predictions published as **custom events** (`PredictedRisk` event type)
- Schema:
  ```
  PredictedRisk {
    entityGuid, modelVersion, hazardScore,
    survival5min, survival15min, survival30min, survival60min,
    topFeaturesJson, depRiskDelta, predictedFailureMode
  }
  ```
- Enables NR-native dashboards and integration with existing workflows/alerts

### 11.3 Alert co-existence

- Predictions do NOT replace existing NR alerts in v1
- A new NR alert condition can be built on `PredictedRisk` event stream (e.g., "alert when survival15min < 0.7")
- Eventual goal: threshold alerts graduate to confirmation signals rather than primary detection

---

## 12. Prototype Milestones

### Milestone 1 — Data path (weeks 1–2)
- NR ingestion for 1 entity class (start with `apm.application`)
- Feature store schemas up in Postgres + Redis
- First 10 features computed and backfilled for 30 days
- Event labels from existing NR threshold alerts

### Milestone 2 — First model (weeks 3–4)
- RSF trained on 1 entity class
- Baseline C-index measurement
- Lead-time comparison vs. threshold alerts on holdout incidents

### Milestone 3 — Inference + NR publishing (weeks 5–6)
- FastAPI inference service
- Warm-path scoring every 60s
- Predictions published as NR custom events
- Streamlit risk dashboard

### Milestone 4 — Graph + multi-class (weeks 7–8)
- Entity relationship ingestion
- Upstream-hazard-weighted feature
- Second entity class (`k8s.pod` or `aws.lambda`)

### Milestone 5 — Feedback loop (weeks 9–10)
- Postmortem ingester + LLM structuring
- Label reconciliation job
- First retraining cycle with postmortem labels
- LLM feature-proposal queue (not yet auto-applied)

---

## 13. Open Questions

1. **Event definition thresholds** — where do we draw the line on what counts as a "failure" for training labels? Needs per-class decisions.
2. **Cold-start entities** — how long must we observe a new entity before scoring it? Initial proposal: 24h, then class-level prior.
3. **Retraining frequency vs. drift** — weekly may be too slow for rapidly changing services; needs measurement.
4. **Graph propagation blast radius** — how do we prevent a single high-hazard entity from flipping the entire dependency graph red?
5. **LLM vendor strategy** — Claude for v1, but postmortem structuring must remain stable across model versions; consider pinning model versions for training-label stability.
6. **Privacy / data handling** — postmortems often contain customer references; need a redaction pass before LLM ingestion in regulated environments.

---

## 14. Known Limitations

- Will not meaningfully predict **deploy-triggered failures** without a strong deployment-marker feature
- Will struggle with **novel failure modes** until sufficient training events accumulate
- Proportional hazards assumption violations are addressed via RSF, but Cox explanations may be misleading in regimes where effects are highly time-varying
- Cross-entity-class risk propagation requires high-quality entity relationships — NR's automatic relationships are good but not exhaustive
- This system augments, does not replace, threshold alerting in v1

---

## 15. Success Criteria (v1)

- **Lead time:** median predictive lead time ≥10 minutes on degradation-class incidents vs. threshold alerts, measured on a holdout of 50+ incidents
- **Precision@10:** ≥60% of the top-10 highest-hazard entities at any given time have an actual event within 60 minutes, or a clear degradation signal
- **C-index:** ≥0.75 on held-out test set for the primary entity class
- **Operational:** warm-path sweep completes in <30s for 10K entities; hot-path p99 <500ms
- **Feedback:** ≥80% of postmortems successfully structured by LLM without manual correction; false-negative feature proposals land in queue within 1h of postmortem ingestion

---

## 16. Appendix: References for Implementation

- `scikit-survival` — https://scikit-survival.readthedocs.io
- `lifelines` — https://lifelines.readthedocs.io
- `pycox` (DeepSurv, DeepHit) — https://github.com/havakv/pycox
- New Relic NerdGraph API — https://docs.newrelic.com/docs/apis/nerdgraph/
- Random Survival Forests (Ishwaran et al., 2008)
- "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, SHAP)
