"""Alert manager — handles alert lifecycle: receive, dedup, persist, clip write."""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

from config.settings import get_settings
from src.common.logger import get_logger
from src.common.metrics import ALERT_LATENCY
from src.common.schemas import ActionEvent, Alert

from .clip_writer import ClipWriter

logger = get_logger()


class AlertManager:
    """Persists alerts to PostgreSQL and writes event clips.

    Uses a fallback in-memory store when Postgres is unavailable (e.g. tests).
    """

    def __init__(
        self,
        clip_writer: Optional[ClipWriter] = None,
        use_db: bool = False,
    ):
        self.clip_writer = clip_writer or ClipWriter()
        self.use_db = use_db
        self._db = None
        self._in_memory: List[Alert] = []
        if use_db:
            try:
                from src.common.db_client import get_postgres
                self._db = get_postgres()
                self._db.init_schema()
                logger.info("AlertManager connected to PostgreSQL")
            except Exception as e:
                logger.warning(f"Postgres unavailable ({e}), using in-memory store")
                self._db = None
                self.use_db = False

    def handle_event(self, event: ActionEvent) -> Optional[Alert]:
        """Convert an ActionEvent into an Alert (with clip).

        Returns None if the event is not in 'alert' state.
        """
        if event.state != "alert":
            return None

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            camera_id=event.camera_id,
            timestamp=event.timestamp,
            action_type=event.action_type,
            confidence=event.confidence,
            track_ids=event.track_ids,
            bbox_coords=event.bbox_coords,
            metadata={"event_id": event.event_id, "is_interaction": event.is_interaction},
        )

        # Write clip (best-effort — don't fail the alert if clip write fails)
        try:
            if self.clip_writer is not None:
                clip_path = self.clip_writer.write_clip(
                    camera_id=event.camera_id,
                    alert_timestamp=event.timestamp,
                    alert_id=alert.alert_id,
                    action_label=event.action_type,
                )
                alert.clip_path = clip_path
        except Exception as e:
            logger.error(f"Clip write failed for alert {alert.alert_id}: {e}")

        # Persist
        if self._db is not None:
            try:
                self._db.insert_alert(
                    alert_id=alert.alert_id,
                    camera_id=alert.camera_id,
                    timestamp=alert.timestamp,
                    action_type=alert.action_type,
                    confidence=alert.confidence,
                    track_ids=alert.track_ids,
                    bbox_coords=alert.bbox_coords,
                    clip_path=alert.clip_path,
                    metadata=alert.metadata,
                )
            except Exception as e:
                logger.error(f"DB insert failed for alert {alert.alert_id}: {e}")
        else:
            self._in_memory.append(alert)

        # End-to-end alert latency metric
        ALERT_LATENCY.observe(time.time() - event.timestamp)

        logger.info(
            f"[{alert.camera_id}] ALERT {alert.action_type} "
            f"(conf={alert.confidence:.2f}, tracks={alert.track_ids}) -> {alert.alert_id}"
        )
        return alert

    def list_alerts(self, limit: int = 100, **filters) -> List[Alert]:
        if self._db is not None:
            rows = self._db.list_alerts(limit=limit, **filters)
            return [Alert(**{
                "alert_id": str(r["id"]),
                "camera_id": r["camera_id"],
                "timestamp": r["timestamp"],
                "action_type": r["action_type"],
                "confidence": r["confidence"],
                "track_ids": r["track_ids"],
                "bbox_coords": r["bbox_coords"],
                "clip_path": r["clip_path"],
                "processed": r["processed"],
                "metadata": r["metadata"],
            }) for r in rows]
        # In-memory
        out = list(self._in_memory)
        if "camera_id" in filters:
            out = [a for a in out if a.camera_id == filters["camera_id"]]
        if "action_type" in filters:
            out = [a for a in out if a.action_type == filters["action_type"]]
        return out[-limit:]

    def push_frame(self, camera_id: str, timestamp: float, frame) -> None:
        """Push a frame into the ring buffer for clip generation."""
        if self.clip_writer is not None:
            self.clip_writer.push_frame(camera_id, timestamp, frame)
