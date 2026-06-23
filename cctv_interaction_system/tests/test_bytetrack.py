"""Tests for Layer 2 ByteTrack."""

from __future__ import annotations

from src.common.schemas import Detection
from src.layer2_tracking.bytetrack import ByteTrack, bbox_iou_matrix


def _make_det(cx, cy, w=80, h=160, conf=0.9):
    return Detection(
        bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        confidence=conf,
        keypoints=[[cx, cy, 0.9]] * 17,
        keypoint_scores=[0.9] * 17,
    )


def test_iou_matrix():
    import numpy as np
    boxes_a = np.array([[0, 0, 100, 100]], dtype=np.float32)
    boxes_b = np.array([[50, 50, 150, 150]], dtype=np.float32)
    iou = bbox_iou_matrix(boxes_a, boxes_b)
    assert iou.shape == (1, 1)
    assert 0.1 < iou[0, 0] < 0.4  # roughly 2500/15000 = 0.167


def test_bytetrack_assigns_consistent_ids():
    tracker = ByteTrack()
    # Frame 1: 2 detections
    dets1 = [_make_det(200, 400), _make_det(800, 400)]
    tracks1 = tracker.update(dets1)
    assert len(tracks1) == 2
    ids1 = sorted([t.track_id for t in tracks1])
    assert ids1 == [1, 2]

    # Frame 2: same detections, slightly moved
    dets2 = [_make_det(205, 400), _make_det(805, 400)]
    tracks2 = tracker.update(dets2)
    ids2 = sorted([t.track_id for t in tracks2])
    assert ids2 == ids1, "IDs should persist across frames"


def test_bytetrack_handles_missing_detections():
    tracker = ByteTrack()
    dets1 = [_make_det(200, 400)]
    tracker.update(dets1)
    # Empty frame
    tracks2 = tracker.update([])
    # Should mark track as lost
    assert len(tracker.lost_tracks) >= 1


def test_bytetrack_initialises_new_track():
    tracker = ByteTrack()
    dets1 = [_make_det(200, 400, conf=0.95)]
    tracks1 = tracker.update(dets1)
    assert len(tracks1) == 1
    assert tracks1[0].track_id == 1
    assert tracks1[0].state == "NEW"


def test_bytetrack_reactivated_track_removed_from_lost():
    """A track recovered from the lost pool must not linger there (no dupes)."""
    tracker = ByteTrack(track_buffer=30)
    # Frame 1: establish a track.
    tracker.update([_make_det(200, 400)])
    # Frame 2: detection disappears -> track goes lost.
    tracker.update([])
    assert len(tracker.lost_tracks) == 1
    lost_obj = tracker.lost_tracks[0]
    # Frame 3: detection reappears at the same place -> reactivated from lost.
    tracks3 = tracker.update([_make_det(200, 400)])
    active_ids = [t.track_id for t in tracks3]
    # Track is active again and no longer sitting in the lost pool.
    assert lost_obj.track_id in active_ids
    assert lost_obj not in tracker.lost_tracks
    assert len(tracker.lost_tracks) == 0
    # tracked_tracks must not contain duplicate references to the same object.
    assert len(tracker.tracked_tracks) == len(set(id(t) for t in tracker.tracked_tracks))


def test_bytetrack_low_confidence_filtered():
    tracker = ByteTrack(track_thresh=0.5, track_high_thresh=0.6)
    # Low-confidence detection should not start a new track
    dets = [_make_det(200, 400, conf=0.3)]
    tracks = tracker.update(dets)
    assert len(tracks) == 0
