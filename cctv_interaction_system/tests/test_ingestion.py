"""Tests for ingestion (Layer 0)."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from src.layer0_ingestion.ffmpeg_pipeline import MockSource, mock_frame_generator
from src.layer0_ingestion.frame_router import FanoutRouter, FrameRouter
from src.layer0_ingestion.ingestion_worker import IngestionWorker


def test_mock_frame_generator_yields_frames():
    gen = mock_frame_generator(width=64, height=64, fps=10, seed=42)
    f1 = next(gen)
    f2 = next(gen)
    assert f1.shape == (64, 64, 3)
    assert f1.dtype == np.uint8
    # Frames should differ (noise + motion)
    assert not np.array_equal(f1, f2)


def test_mock_source_read_frame():
    src = MockSource("cam_001", width=64, height=64, fps=10, seed=1)
    src.start()
    f = src.read_frame()
    assert f is not None
    assert f.shape == (64, 64, 3)


def test_frame_router_round_robin():
    q1 = queue.Queue(maxsize=10)
    q2 = queue.Queue(maxsize=10)
    router = FrameRouter([q1, q2], strategy="round_robin")
    router.route({"i": 1})
    router.route({"i": 2})
    router.route({"i": 3})
    # Round robin: q1, q2, q1
    assert q1.qsize() == 2
    assert q2.qsize() == 1


def test_frame_router_least_loaded():
    q1 = queue.Queue(maxsize=10)
    q2 = queue.Queue(maxsize=10)
    router = FrameRouter([q1, q2], strategy="least_loaded")
    # Pre-fill q1
    for i in range(5):
        q1.put({"i": i})
    router.route({"i": 100})
    # Should go to q2 (less loaded)
    assert q2.qsize() == 1
    assert q1.qsize() == 5


def test_fanout_router_broadcasts():
    q1 = queue.Queue(maxsize=10)
    q2 = queue.Queue(maxsize=10)
    router = FanoutRouter([q1, q2])
    router.route({"i": 1})
    assert q1.qsize() == 1
    assert q2.qsize() == 1


def test_ingestion_worker_produces_frames():
    """Ingestion worker should produce frames on tracking queue."""
    det_q = queue.Queue(maxsize=4096)
    track_q = queue.Queue(maxsize=4096)
    worker = IngestionWorker(
        camera_id="cam_001",
        rtsp_url="rtsp://test",
        detection_queue=det_q,
        tracking_queue=track_q,
        detection_stride=5,
        width=64,
        height=64,
        fps=30,
        use_mock=True,
    )
    worker.start()
    # Wait for at least 6 frames (to get at least 1 detection)
    time.sleep(0.3)
    worker.stop()
    # Tracking queue should have frames
    assert track_q.qsize() > 0
    # Detection queue should have at least 1 frame (every 5th)
    assert det_q.qsize() > 0
    # Detection queue should be smaller (1/5th rate)
    assert det_q.qsize() < track_q.qsize()


def test_ingestion_worker_detection_stride():
    """Verify that only every Nth frame goes to detection queue."""
    det_q = queue.Queue(maxsize=8192)
    track_q = queue.Queue(maxsize=8192)
    worker = IngestionWorker(
        camera_id="cam_001",
        rtsp_url="rtsp://test",
        detection_queue=det_q,
        tracking_queue=track_q,
        detection_stride=5,
        width=64,
        height=64,
        fps=30,
        use_mock=True,
    )
    worker.start()
    time.sleep(0.5)
    worker.stop()
    # Detection should be ~1/5 of tracking
    n_track = track_q.qsize()
    n_det = det_q.qsize()
    if n_track > 10:  # only assert if we got enough frames
        ratio = n_track / max(1, n_det)
        # Should be roughly 5 (allow some tolerance)
        assert 3 <= ratio <= 7, f"expected ratio ~5, got {ratio} (track={n_track}, det={n_det})"
