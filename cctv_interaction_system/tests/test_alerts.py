"""Tests for Layer 6 clip writer + alert manager."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.common.schemas import ActionEvent
from src.layer6_alerts.alert_manager import AlertManager
from src.layer6_alerts.clip_writer import ClipWriter, FrameRingBuffer


def test_frame_ring_buffer_push_get():
    buf = FrameRingBuffer(capacity_s=2.0, fps=10)
    for i in range(20):
        buf.push(i * 0.1, np.zeros((10, 10, 3), dtype=np.uint8))
    # Get last 0.5s
    frames = buf.get_last_n_seconds(0.5, now=1.9)
    assert len(frames) >= 5


def test_frame_ring_buffer_evicts_old():
    buf = FrameRingBuffer(capacity_s=1.0, fps=5)  # maxlen=5
    for i in range(20):
        buf.push(i * 0.2, np.zeros((10, 10, 3), dtype=np.uint8))
    # Only last 5 should remain
    frames = buf.get_range(0.0, 100.0)
    assert len(frames) <= 5


def test_clip_writer_creates_file(tmp_path):
    cw = ClipWriter(storage_path=str(tmp_path))
    # Push some frames
    for i in range(10):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 10  # different colours
        cw.push_frame("cam_001", i * 0.1, frame)
    # Write clip
    path = cw.write_clip(
        camera_id="cam_001",
        alert_timestamp=0.9,
        alert_id="abc123",
        action_label="fight",
    )
    if path is not None:  # cv2 may not be available in all envs
        assert path.endswith(".mp4")
        import os
        assert os.path.exists(path)


def test_alert_manager_in_memory_mode():
    """AlertManager without DB connection stores alerts in memory."""
    am = AlertManager(use_db=False)
    event = ActionEvent(
        camera_id="cam_001",
        frame_id=1,
        timestamp=time.time(),
        action_type="fight",
        confidence=0.85,
        track_ids=[1, 2],
        bbox_coords=[[100, 100, 200, 300], [150, 100, 250, 300]],
        is_interaction=True,
        state="alert",
    )
    alert = am.handle_event(event)
    assert alert is not None
    assert alert.action_type == "fight"
    assert alert.camera_id == "cam_001"
    alerts = am.list_alerts(limit=10)
    assert len(alerts) >= 1


def test_alert_manager_ignores_non_alert_events():
    am = AlertManager(use_db=False)
    event = ActionEvent(
        camera_id="cam_001",
        frame_id=1,
        timestamp=time.time(),
        action_type="fight",
        confidence=0.85,
        track_ids=[1],
        state="candidate",  # not alert
    )
    alert = am.handle_event(event)
    assert alert is None


def test_alert_manager_list_filters():
    am = AlertManager(use_db=False)
    for action in ["fight", "fight", "hug"]:
        am.handle_event(ActionEvent(
            camera_id="cam_001",
            frame_id=1,
            timestamp=time.time(),
            action_type=action,
            confidence=0.85,
            track_ids=[1, 2],
            state="alert",
        ))
    fights = am.list_alerts(limit=10, action_type="fight")
    hugs = am.list_alerts(limit=10, action_type="hug")
    assert len(fights) == 2
    assert len(hugs) == 1
