"""Frame router / load balancer for detection workers.

In a multi-GPU deployment, this routes frames from many cameras across
detection worker queues in round-robin fashion, with dynamic load balancing
based on queue depth.

In single-node mode this is a simple multiplexer.
"""

from __future__ import annotations

import queue
import threading
from typing import Dict, List

from src.common.logger import get_logger

logger = get_logger()


class FrameRouter:
    """Round-robin router with optional depth-aware load balancing."""

    def __init__(self, worker_queues: List["queue.Queue"], strategy: str = "round_robin"):
        self.worker_queues = worker_queues
        self.strategy = strategy
        self._lock = threading.Lock()
        self._idx = 0

    def route(self, packet: dict) -> bool:
        if not self.worker_queues:
            return False
        target = self._pick()
        try:
            target.put_nowait(packet)
            return True
        except queue.Full:
            # Try others
            for _ in range(len(self.worker_queues)):
                target = self._pick()
                try:
                    target.put_nowait(packet)
                    return True
                except queue.Full:
                    continue
            return False

    def _pick(self) -> "queue.Queue":
        if self.strategy == "least_loaded":
            return min(self.worker_queues, key=lambda q: q.qsize())
        # round robin
        with self._lock:
            q = self.worker_queues[self._idx % len(self.worker_queues)]
            self._idx += 1
            return q


class FanoutRouter:
    """Routes packets to multiple downstream queues (broadcast)."""

    def __init__(self, downstream: List["queue.Queue"]):
        self.downstream = downstream

    def route(self, packet: dict) -> None:
        for q in self.downstream:
            try:
                q.put_nowait(packet)
            except queue.Full:
                logger.warning(f"FanoutRouter: queue full, dropping packet")
