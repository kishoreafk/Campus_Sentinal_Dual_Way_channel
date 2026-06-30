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
    """Thread-safe batch collector with timeout-based flush.

    Uses a single lock and a threading.Event for efficient wakeup.
    """

    def __init__(self, max_batch: int = 32, timeout_ms: int = 50, min_batch: int = 1):
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000.0
        self.min_batch = min_batch
        self._buf: List[dict] = []
        self._first_added_at: Optional[float] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def add(self, packet: dict) -> None:
        with self._lock:
            is_first = not self._buf
            self._buf.append(packet)
            if is_first:
                self._first_added_at = time.time()
            if len(self._buf) >= self.max_batch:
                self._ready.set()

    def get_batch(self) -> Optional[List[dict]]:
        """Block until a batch is ready, then return it (single locked op).

        Returns None when the collector is empty and would block forever
        (caller should poll again).
        """
        with self._lock:
            if not self._buf:
                self._ready.clear()
                return None
            if len(self._buf) >= self.max_batch:
                return self._do_flush()
            if self._first_added_at is not None and \
                    time.time() - self._first_added_at > self.timeout:
                if len(self._buf) >= self.min_batch:
                    return self._do_flush()
            return None

    def _do_flush(self) -> List[dict]:
        batch = self._buf
        self._buf = []
        self._first_added_at = None
        self._ready.clear()
        return batch

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def ready_event(self) -> threading.Event:
        return self._ready

    def try_get_batch(self) -> Optional[List[dict]]:
        """Return a batch if ready, else None."""
        return self.get_batch()

    # -- backward-compat aliases (used by tests) -----------------------
    def should_flush(self) -> bool:
        with self._lock:
            return bool(self._buf) and (
                len(self._buf) >= self.max_batch or
                (self._first_added_at is not None and time.time() - self._first_added_at > self.timeout)
            )

    def flush(self) -> List[dict]:
        with self._lock:
            return self._do_flush()
