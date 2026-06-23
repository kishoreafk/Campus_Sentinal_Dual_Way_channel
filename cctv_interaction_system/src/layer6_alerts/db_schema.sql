-- PostgreSQL schema for CCTV alert storage
-- Run via: psql -U cctv -d cctv_alerts -f db_schema.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS alerts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id       VARCHAR(64) NOT NULL,
    timestamp       DOUBLE PRECISION NOT NULL,
    action_type     VARCHAR(64) NOT NULL,
    confidence      REAL NOT NULL,
    track_ids       JSONB NOT NULL DEFAULT '[]'::jsonb,
    bbox_coords     JSONB NOT NULL DEFAULT '[]'::jsonb,
    clip_path       TEXT,
    processed       BOOLEAN NOT NULL DEFAULT FALSE,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_camera_time
    ON alerts (camera_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_action
    ON alerts (action_type);
CREATE INDEX IF NOT EXISTS idx_alerts_unprocessed
    ON alerts (processed) WHERE processed = FALSE;
CREATE INDEX IF NOT EXISTS idx_alerts_created
    ON alerts (created_at DESC);

CREATE TABLE IF NOT EXISTS cameras (
    camera_id       VARCHAR(64) PRIMARY KEY,
    name            VARCHAR(256),
    location        VARCHAR(512),
    rtsp_url        TEXT,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS system_events (
    id              BIGSERIAL PRIMARY KEY,
    event_type      VARCHAR(64) NOT NULL,
    payload         JSONB NOT NULL,
    timestamp       DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON system_events (event_type, timestamp DESC);
