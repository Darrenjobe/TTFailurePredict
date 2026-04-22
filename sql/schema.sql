-- ---------------------------------------------------------------------------
-- survpredict :: schema
-- Postgres + TimescaleDB. See design doc §5 for rationale.
-- This file is idempotent: safe to run on a fresh DB or re-run during dev.
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- features : time-varying covariates per entity (the training/inference fuel)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    entity_guid      TEXT NOT NULL,
    entity_class     TEXT NOT NULL,
    feature_name     TEXT NOT NULL,
    window_seconds   INT  NOT NULL,
    value            DOUBLE PRECISION,
    computed_at      TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (entity_guid, feature_name, window_seconds, computed_at)
);

SELECT create_hypertable('features', 'computed_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS ix_features_entity_time
    ON features (entity_guid, computed_at DESC);
CREATE INDEX IF NOT EXISTS ix_features_class_time
    ON features (entity_class, computed_at DESC);

-- ---------------------------------------------------------------------------
-- events : the "failure" observations used as survival endpoints / labels
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    event_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_guid      TEXT NOT NULL,
    entity_class     TEXT NOT NULL,
    event_type       TEXT NOT NULL,   -- slo_breach | oom | throttle | confirmed_incident | health_transition
    severity         TEXT,
    occurred_at      TIMESTAMPTZ NOT NULL,
    detected_by      TEXT,            -- threshold_alert | postmortem | health_transition | synthetic
    postmortem_id    UUID,
    label_status     TEXT DEFAULT 'pending',  -- pending | confirmed_tp | false_positive | near_miss
    metadata         JSONB
);

CREATE INDEX IF NOT EXISTS ix_events_entity_time
    ON events (entity_guid, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_events_class_time
    ON events (entity_class, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_events_postmortem
    ON events (postmortem_id) WHERE postmortem_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- entity_edges : dependency / relationship graph with temporal validity
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity_edges (
    src_guid         TEXT NOT NULL,
    dst_guid         TEXT NOT NULL,
    relationship     TEXT NOT NULL,   -- calls | contains | depends_on | hosts
    weight           DOUBLE PRECISION DEFAULT 1.0,
    valid_from       TIMESTAMPTZ NOT NULL,
    valid_to         TIMESTAMPTZ,
    PRIMARY KEY (src_guid, dst_guid, relationship, valid_from)
);

CREATE INDEX IF NOT EXISTS ix_edges_src ON entity_edges (src_guid) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS ix_edges_dst ON entity_edges (dst_guid) WHERE valid_to IS NULL;

-- ---------------------------------------------------------------------------
-- entities : the set we're scoring (synced from NR)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    entity_guid      TEXT PRIMARY KEY,
    entity_class     TEXT NOT NULL,
    name             TEXT,
    tags             JSONB,
    first_seen_at    TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_entities_class ON entities (entity_class);

-- ---------------------------------------------------------------------------
-- predictions : every scoring pass writes a row; evaluated later against events
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_guid       TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    predicted_at      TIMESTAMPTZ NOT NULL,
    hazard_score      DOUBLE PRECISION,
    survival_5min     DOUBLE PRECISION,
    survival_15min    DOUBLE PRECISION,
    survival_30min    DOUBLE PRECISION,
    survival_60min    DOUBLE PRECISION,
    top_features      JSONB,
    dep_risk_delta    DOUBLE PRECISION,
    predicted_failure_mode TEXT,
    outcome_event_id  UUID REFERENCES events(event_id) ON DELETE SET NULL
);

SELECT create_hypertable('predictions', 'predicted_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS ix_predictions_entity_time
    ON predictions (entity_guid, predicted_at DESC);
CREATE INDEX IF NOT EXISTS ix_predictions_hazard
    ON predictions (predicted_at DESC, hazard_score DESC);

-- ---------------------------------------------------------------------------
-- postmortems : structured narrative + labels fed back to training
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS postmortems (
    postmortem_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source               TEXT,            -- slack | jira | markdown
    source_ref           TEXT,            -- channel+ts, jira key, or file path
    raw_text             TEXT,
    structured           JSONB,
    entities_affected    TEXT[],
    root_cause_category  TEXT,
    event_start          TIMESTAMPTZ,
    event_end            TIMESTAMPTZ,
    contributing_signals TEXT[],
    severity             TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_postmortems_created ON postmortems (created_at DESC);

-- ---------------------------------------------------------------------------
-- feature_proposals : LLM-generated suggestions that await human review
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_proposals (
    proposal_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    triggered_by     UUID REFERENCES postmortems(postmortem_id) ON DELETE SET NULL,
    yaml_spec        TEXT NOT NULL,
    rationale        TEXT,
    status           TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected | merged
    reviewer         TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    decided_at       TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- model_registry_meta : mirror of MLflow metadata we care about for scoring
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry_meta (
    model_version    TEXT PRIMARY KEY,
    entity_class     TEXT NOT NULL,
    algorithm        TEXT NOT NULL,   -- rsf | cox | xgb_surv
    c_index          DOUBLE PRECISION,
    ibs              DOUBLE PRECISION,
    trained_at       TIMESTAMPTZ,
    promoted_at      TIMESTAMPTZ,
    is_production    BOOLEAN DEFAULT FALSE,
    artifact_uri     TEXT,
    feature_list     JSONB
);

CREATE INDEX IF NOT EXISTS ix_models_class_prod
    ON model_registry_meta (entity_class)
    WHERE is_production;

-- ---------------------------------------------------------------------------
-- deployment_markers : change-point reference for "time since deploy" features
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS deployment_markers (
    marker_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_guid      TEXT NOT NULL,
    revision         TEXT,
    changelog        TEXT,
    deployed_at      TIMESTAMPTZ NOT NULL,
    source           TEXT
);

CREATE INDEX IF NOT EXISTS ix_deploys_entity_time
    ON deployment_markers (entity_guid, deployed_at DESC);
