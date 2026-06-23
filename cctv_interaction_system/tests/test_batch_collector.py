"""Tests for Layer 1 batch collector."""

from __future__ import annotations

import time

from src.layer1_detection.batch_collector import BatchCollector


def test_batch_collector_flushes_on_size():
    bc = BatchCollector(max_batch=4, timeout_ms=10000)
    for i in range(4):
        bc.add({"i": i})
    assert bc.should_flush() is True
    batch = bc.flush()
    assert len(batch) == 4
    assert bc.should_flush() is False


def test_batch_collector_flushes_on_timeout():
    bc = BatchCollector(max_batch=32, timeout_ms=50)
    bc.add({"i": 1})
    assert bc.should_flush() is False
    time.sleep(0.08)
    assert bc.should_flush() is True
    batch = bc.flush()
    assert len(batch) == 1


def test_batch_collector_empty():
    bc = BatchCollector(max_batch=4, timeout_ms=10)
    assert bc.should_flush() is False
    assert bc.flush() == []
    assert bc.try_get_batch() is None


def test_batch_collector_thread_safe():
    import threading
    bc = BatchCollector(max_batch=100, timeout_ms=10000)

    def adder(start, end):
        for i in range(start, end):
            bc.add({"i": i})

    threads = [threading.Thread(target=adder, args=(i * 100, (i + 1) * 100)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(bc) == 400
    batch = bc.flush()
    assert len(batch) == 400
