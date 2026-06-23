"""Alert deduplication.

Suppresses duplicate alerts for the same (camera_id, track_ids, action_label)
within a configurable window (default 3 seconds).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Set, Tuple

from src.common.metrics import ALERTS_GENERATED, ALERTS_SUPPRESSED


class AlertDeduplicator:
    """Time-windowed alert deduplication."""

    def __init__(self, window_s: float = 3.0):
        self.window_s = window_s
        # key -> last alert timestamp
        self._last_alert: Dict[Tuple[str, str, str], float] = {}

    def _make_key(
        self,
        camera_id: str,
        track_ids: tuple[int, ...],
        action_label: str,
    ) -> Tuple[str, str, str]:
        # Normalise track_ids order so (5,7) and (7,5) dedup together
        sorted_ids = tuple(sorted(track_ids))
        return (camera_id, str(sorted_ids), action_label)

    def should_alert(
        self,
        camera_id: str,
        track_ids: tuple[int, ...],
        action_label: str,
        now: float | None = None,
    ) -> bool:
        """Return True if this alert should fire (not a duplicate)."""
        if now is None:
            now = time.time()
        key = self._make_key(camera_id, track_ids, action_label)
        last = self._last_alert.get(key, 0.0)
        if now - last < self.window_s:
            ALERTS_SUPPRESSED.labels(camera_id, action_label).inc()
            return False
        self._last_alert[key] = now
        ALERTS_GENERATED.labels(camera_id, action_label).inc()
        return True

    def cleanup(self, now: float | None = None) -> None:
        """Drop entries older than window — call periodically."""
        if now is None:
            now = time.time()
        stale = [k for k, t in self._last_alert.items() if now - t > self.window_s * 10]
        for k in stale:
            del self._last_alert[k]
