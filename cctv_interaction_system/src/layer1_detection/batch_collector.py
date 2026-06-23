"""Batch collector for detection.

Accumulates frames from multiple cameras until either:
  - max_batch is reached, OR
  - timeout elapses since the first frame in the buffer.

This maximises GPU utilisation by always running the largest possible batch.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional


class BatchCollector:
    """Thread-safe batch collector with timeout-based flush."""

    def __init__(self, max_batch: int = 32, timeout_ms: int = 50):
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000.0
        self._buf: List[dict] = []
        self._first_added_at: Optional[float] = None
        self._lock = threading.Lock()

    def add(self, packet: dict) -> None:
        with self._lock:
            if not self._buf:
                self._first_added_at = time.time()
            self._buf.append(packet)

    def should_flush(self) -> bool:
        with self._lock:
            if not self._buf:
                return False
            if len(self._buf) >= self.max_batch:
                return True
            if self._first_added_at is not None and \
                    time.time() - self._first_added_at > self.timeout:
                return True
            return False

    def flush(self) -> List[dict]:
        with self._lock:
            batch = self._buf
            self._buf = []
            self._first_added_at = None
            return batch

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def try_get_batch(self) -> Optional[List[dict]]:
        """Return a batch if ready, else None."""
        if self.should_flush():
            return self.flush()
        return None
