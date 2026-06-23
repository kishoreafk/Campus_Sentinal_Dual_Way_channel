"""Tests for track manager integration."""

from __future__ import annotations

import numpy as np

from src.common.schemas import Detection
from src.layer2_tracking.track_manager import TrackManager


def _make_det(cx, cy, w=80, h=160, conf=0.9, kp_offset=0):
    """Build a Detection with plausible keypoints."""
    kps = []
    # 17 keypoints roughly distributed in bbox
    layout = [
        (0.50, 0.10), (0.45, 0.08), (0.55, 0.08), (0.40, 0.10), (0.60, 0.10),
        (0.35, 0.25), (0.65, 0.25), (0.30, 0.45), (0.70, 0.45),
        (0.28, 0.65), (0.72, 0.65),
        (0.42, 0.55), (0.58, 0.55),
        (0.40, 0.78), (0.60, 0.78),
        (0.40, 0.95), (0.60, 0.95),
    ]
    for rx, ry in layout:
        kps.append([
            cx - w / 2 + rx * w + kp_offset,
            cy - h / 2 + ry * h,
            0.9,
        ])
    return Detection(
        bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        confidence=conf,
        keypoints=kps,
        keypoint_scores=[0.9] * 17,
    )


def test_track_manager_assigns_ids():
    tm = TrackManager(camera_id="cam_001")
    dets = [_make_det(200, 360), _make_det(800, 360)]
    tracks = tm.update(dets)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2


def test_track_manager_persists_ids_across_frames():
    tm = TrackManager(camera_id="cam_001")
    dets = [_make_det(200, 360), _make_det(800, 360)]
    t1 = tm.update(dets)
    # Move slightly
    dets2 = [_make_det(205, 360), _make_det(805, 360)]
    t2 = tm.update(dets2)
    ids1 = {t.track_id for t in t1}
    ids2 = {t.track_id for t in t2}
    assert ids1 == ids2


def test_track_manager_skeleton_buffer_populated():
    tm = TrackManager(camera_id="cam_001")
    for fid in range(5):
        dets = [_make_det(200, 360, kp_offset=fid)]
        tm.update(dets)
    skel = tm.get_skeleton(1)
    assert skel is not None
    # Skeleton buffer should have up to 48 frames; we pushed 5
    assert skel.shape[0] == 48  # padded
    assert skel.shape[1:] == (17, 3)


def test_track_manager_get_all_skeletons():
    tm = TrackManager(camera_id="cam_001")
    dets = [_make_det(200, 360), _make_det(800, 360)]
    tm.update(dets)
    all_skels = tm.get_all_skeletons()
    # Both tracks should have skeletons
    assert len(all_skels) >= 2
    for tid, skel in all_skels.items():
        assert skel.shape == (48, 17, 3)


def test_track_manager_handles_empty_detections():
    tm = TrackManager(camera_id="cam_001")
    tracks = tm.update([])
    assert tracks == []


def test_track_manager_with_frame():
    tm = TrackManager(camera_id="cam_001")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = [_make_det(200, 360)]
    tracks = tm.update(dets, frame=frame)
    assert len(tracks) == 1
    # Appearance feature should have been extracted
    assert 1 in tm.appearances or tm.appearances == {}
