# TTFailurePredict — Survival Analysis for Predictive Observability

> Prototype v0.1. See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design doc.

`TTFailurePredict` is a predictive-observability system that takes New Relic telemetry
as input and produces per-entity **hazard scores** — the instantaneous
probability that an entity (APM app, k8s pod, Lambda, host, …) is about to
fail. Instead of waiting for a threshold to fire, the system forecasts
degradation-class incidents **10–30 minutes ahead** and surfaces them with
feature attributions and dependency context.

The model class is **survival analysis** — specifically Random Survival
Forests (primary) and Cox PH with time-varying covariates (interpretability).
Survival models natively handle right-censoring, time-varying covariates, and
produce calibrated probabilities over a horizon — all of which are properties
you want in an observability signal.

---

## Why survival analysis?

Most ML-for-ops work frames the problem as binary classification ("will this
fail in the next N minutes?"). That throws away two useful signals:

1. **Censoring**. Entities that *haven't failed yet* still contain
   information. Classifiers either label them negative (biasing the training
   set) or drop them.
2. **Horizons**. A survival model gives you `S(t | X)` — the probability of
   surviving past `t` — for any `t`, from one fit. No separate model per
   horizon.

Add the operational reality — telemetry is noisy, events are rare, feature
distributions shift after every deploy — and the survival framing earns its
keep.

---

## Architecture at a glance

```
New Relic (NRDB, NerdGraph, deployment markers)
          │
          ▼
Ingestion ──► Feature store ──► Training ──► MLflow registry
 (NRQL     (Redis + Timescale)  (RSF + Cox)       │
  puller,                                         ▼
  entity                                    Inference service
  graph,                                    (FastAPI + warm sweep
  postmortems)                               + graph propagation)
                                                  │
                                                  ▼
                              Predictions ──► Streamlit dashboard
                                          └─► New Relic custom events
                                                  │
Postmortems ──► LLM structurer ──► Label reconciliation ──► Retraining
                                                 │
                                                 └─► LLM feature proposal queue
                                                     (human-reviewed)
```

See the per-module READMEs under [`src/survpredict/`](src/survpredict/) for
deeper detail.

---

## Repo layout

```
TTFailurePredict/
├── docs/                       # design doc + runbooks
├── sql/                        # Postgres + TimescaleDB schema
├── feature_specs/              # YAML feature definitions, per entity class
├── src/survpredict/
│   ├── common/                 # config, db, logging, time helpers
│   ├── ingestion/              # NR client, NRQL puller, entity graph, postmortem intake
│   ├── features/               # spec loader, transformations, online/offline stores
│   ├── labels/                 # survival dataset assembly
│   ├── training/               # RSF, Cox, eval, MLflow registry
│   ├── inference/              # FastAPI service, warm-path sweeper, propagation, SHAP
│   ├── feedback/               # LLM structurer, label reconciliation, feature proposer
│   ├── publish/                # PredictedRisk custom events back to NR
│   ├── ui/                     # Streamlit dashboard
│   └── cli.py                  # `survpredict ...` entry point
├── scripts/bootstrap.sh        # one-shot prototype boot
├── tests/                      # unit tests (transformations, spec loader, …)
├── docker-compose.yml          # postgres + redis + mlflow + service + dashboard
├── Dockerfile
└── pyproject.toml
```

---

## Quickstart (prototype)

**Prereqs:** Docker, Python 3.11+, a New Relic account with a user API key
and an ingest license key.

```bash
git clone <repo> && cd TTFailurePredict
cp .env.example .env                 # fill in NR + Anthropic keys
./scripts/bootstrap.sh               # brings up infra, pulls initial data
survpredict train run apm.application
survpredict train promote <version_printed_above>
docker compose up -d inference warm-sweeper dashboard
open http://localhost:8501           # risk dashboard
open http://localhost:8080/docs      # inference API
```

Milestone 1–3 from the design doc are covered by the above. Milestones 4–5
(graph propagation, feedback loop) require additional entity classes and a
postmortem corpus; see below.

---

## Day-to-day CLI

```bash
# Ingestion
survpredict ingest entities APPLICATION          # sync APM apps from NR
survpredict ingest nrql                          # pull NRQL features
survpredict ingest incidents                     # pull NrAiIncident as labels
survpredict ingest postmortems ./postmortems     # pick up markdown postmortems

# Features
survpredict features list                        # show all loaded specs
survpredict features aggregate <entity_guid>     # compute derived vector now

# Training
survpredict train run apm.application            # full training run
survpredict train promote <version>              # mark as production

# Feedback loop
survpredict feedback structure                   # LLM-structure pending postmortems
survpredict feedback reconcile                   # TP/FP/FN/near-miss label pass
survpredict feedback propose <event_id>          # LLM-proposed features for a miss
survpredict feedback retrain apm.application     # incremental retrain
```

---

## Service endpoints

- `GET  /healthz`
- `GET  /score/{entity_guid}` — score on demand, returns hazard +
  survival + top features + dep risk
- `POST /score/batch` — body `{"guids":[...]}`
- `GET  /top?k=25&window=60` — top-K currently-riskiest entities
- `POST /invalidate-cache` — drop cached model (body:
  `{"entity_class":"apm.application"}`)

---

## Data flow

1. **Entity sync.** `survpredict ingest entities <TYPE>` pulls the NR
   entity search results for a type (e.g. `APPLICATION`) and upserts them
   into `entities`. Relationships land in `entity_edges`.
2. **NRQL puller.** For every feature spec whose `source` is `nr_metric`,
   the puller substitutes `?` with the entity GUID and walks windows
   declared in the spec, writing raw values into `features` (TimescaleDB).
3. **Feature aggregation.** The aggregator compiles raw values into a
   derived vector (level, trend slope, volatility, CUSUM-style change-point
   distance, cyclical time encodings). The vector is cached in Redis and
   used by the inference service.
4. **Label assembly.** For each entity class, the labels module walks
   `events` and feature history in parallel to produce (duration, event
   indicator, covariate vector) tuples. Right-censored cases (no event
   within the horizon) are preserved.
5. **Training.** Per entity class: fit a Random Survival Forest on a
   90-day rolling window; fit a Cox PH on the same data for
   interpretability. Evaluate (C-index, IBS, time-dependent AUC,
   precision@K, lead-time distribution). Log to MLflow. Mirror metadata
   to `model_registry_meta`.
6. **Canary + promotion.** Run in shadow for 7 days. Promote iff C-index
   improves ≥1% *and* lead time doesn't regress.
7. **Inference.** Warm-path sweeper runs every 60s. Hot path responds on
   feature-change events. Graph propagation re-scores downstream entities
   when an upstream crosses the threshold, rate-limited to prevent storms.
8. **Publishing.** Each prediction is posted to New Relic as a
   `PredictedRisk` custom event, enabling native NR dashboards and alerts.
9. **Feedback.** Postmortems are LLM-structured into a controlled
   vocabulary and joined back to predictions for label reconciliation
   (TP / FP / FN / near-miss). False negatives trigger an LLM pass that
   proposes new features; proposals land in a human-review queue and
   never auto-modify the spec repo.

---

## What the LLM does (and doesn't) do

**Does:** structure postmortems into JSON, propose new features and event
types, draft risk explanations.

**Never:** modify model weights, edit the production feature-spec YAML,
change prediction thresholds, or label events without a deterministic gate.
All proposals pass through `feature_proposals` (status: pending) or through
the human before taking effect. See design doc §9.5.

---

## Known limits

- Cold-start entities: 24h observation before first scoring; class-level
  priors after that.
- Novel failure modes: no predictive signal until events accumulate.
- Deploy-triggered failures: need deployment markers wired up.
- Threshold-alert co-existence: predictions run **alongside**, not instead
  of, existing NR alerts in v1.

See design doc §14 and §13 for the open-questions list.

---

## Further reading

- [`docs/DESIGN.md`](docs/DESIGN.md) — the authoritative spec.
- `src/survpredict/*/README.md` — per-module details.
- scikit-survival: <https://scikit-survival.readthedocs.io>
- lifelines: <https://lifelines.readthedocs.io>
- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model
  Predictions."
