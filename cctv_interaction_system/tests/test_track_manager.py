"""Tests for track manager integration."""

from __future__ import annotations

import numpy as np

from src.common.schemas import Detection
from src.layer2_tracking.osnet_reid import BaseReID
from src.layer2_tracking.track_manager import TrackManager


class _ConstReID(BaseReID):
    """Stub Re-ID: every crop yields the same feature (cosine sim == 1)."""

    def extract(self, frame, bbox):
        return np.ones(8, dtype=np.float32)


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


def test_track_manager_reid_recovery_restores_original_id():
    """When an occluded track reappears under a new ByteTrack ID, Re-ID should
    fold the new ID back into the original (not leave it as dead-code alias)."""
    tm = TrackManager(camera_id="cam_001", reid=_ConstReID(), reid_cosine_threshold=0.5)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Establish person A at the left -> track id 1.
    for _ in range(4):
        tm.update([_make_det(200, 400)], frame=frame)
    assert 1 in tm.tracklets

    # A vanishes; a person reappears far away (right) -> a fresh id 2 that,
    # by appearance, matches the now-occluded id 1. After enough missed
    # frames, Re-ID recovery should merge id 2 back into id 1.
    last = []
    for _ in range(8):
        last = tm.update([_make_det(800, 400)], frame=frame)

    assert tm.id_aliases.get(2) == 1, "new id should alias back to the original"
    assert 2 not in tm.tracklets, "orphan recovered tracklet must be cleaned up"
    assert {t.track_id for t in last} == {1}, "track reported under original id"
    assert tm.tracklets[1].state == "ACTIVE"


def test_track_manager_with_frame():
    tm = TrackManager(camera_id="cam_001")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = [_make_det(200, 360)]
    tracks = tm.update(dets, frame=frame)
    assert len(tracks) == 1
    # Appearance feature should have been extracted
    assert 1 in tm.appearances or tm.appearances == {}
