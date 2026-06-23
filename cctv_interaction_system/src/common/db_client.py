"""PostgreSQL client for Layer 6 (alerts persistence)."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional

import psycopg2
import psycopg2.extras

from .logger import get_logger

logger = get_logger()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id              UUID PRIMARY KEY,
    camera_id       VARCHAR(64) NOT NULL,
    timestamp       DOUBLE PRECISION NOT NULL,
    action_type     VARCHAR(64) NOT NULL,
    confidence      REAL NOT NULL,
    track_ids       JSONB NOT NULL DEFAULT '[]',
    bbox_coords     JSONB NOT NULL DEFAULT '[]',
    clip_path       TEXT,
    processed       BOOLEAN NOT NULL DEFAULT FALSE,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_camera_time
    ON alerts (camera_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_action
    ON alerts (action_type);
CREATE INDEX IF NOT EXISTS idx_alerts_unprocessed
    ON alerts (processed) WHERE processed = FALSE;

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
"""


class PostgresClient:
    """Wrapper around psycopg2 with helpers for alert persistence."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "cctv",
        password: str = "cctv",
        database: str = "cctv_alerts",
    ):
        self.conn_params = dict(
            host=host, port=port, user=user, password=password, dbname=database,
        )
        self._conn = None

    def _connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.conn_params)
            self._conn.autocommit = False
        return self._conn

    @contextmanager
    def cursor(self) -> Iterator[psycopg2.extras.RealDictCursor]:
        conn = self._connect()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def init_schema(self) -> None:
        with self.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        logger.info("Postgres schema initialised")

    def insert_alert(
        self,
        alert_id: str,
        camera_id: str,
        timestamp: float,
        action_type: str,
        confidence: float,
        track_ids: List[int],
        bbox_coords: List[List[float]],
        clip_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO alerts
                  (id, camera_id, timestamp, action_type, confidence,
                   track_ids, bbox_coords, clip_path, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    alert_id, camera_id, timestamp, action_type, confidence,
                    json.dumps(track_ids), json.dumps(bbox_coords),
                    clip_path, json.dumps(metadata or {}),
                ),
            )

    def list_alerts(
        self,
        camera_id: Optional[str] = None,
        action_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        since: Optional[float] = None,
    ) -> List[dict]:
        query = "SELECT * FROM alerts WHERE 1=1"
        params: list = []
        if camera_id:
            query += " AND camera_id = %s"
            params.append(camera_id)
        if action_type:
            query += " AND action_type = %s"
            params.append(action_type)
        if since is not None:
            query += " AND timestamp >= %s"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        with self.cursor() as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def mark_processed(self, alert_id: str) -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET processed = TRUE WHERE id = %s",
                (alert_id,),
            )

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()


# Module-level singleton
_client: Optional[PostgresClient] = None


def get_postgres() -> PostgresClient:
    global _client
    if _client is None:
        from config.settings import get_settings
        cfg = get_settings().postgres
        _client = PostgresClient(cfg.host, cfg.port, cfg.user, cfg.password, cfg.database)
    return _client


def reset_postgres() -> None:
    global _client
    if _client is not None:
        _client.close()
    _client = None
