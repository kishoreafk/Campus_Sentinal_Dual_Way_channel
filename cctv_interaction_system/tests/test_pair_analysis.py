"""Tests for Layer 3 pair analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.schemas import Detection, Tracklet
from src.layer3_pair_analysis.distance_matrix import (
    bbox_center,
    bbox_iou,
    bbox_union,
    face_to_face_dot,
    face_vector,
)
from src.layer3_pair_analysis.pair_analyzer import PairAnalyzer
from src.layer3_pair_analysis.router import Router


def test_bbox_center():
    assert bbox_center((0, 0, 100, 100)) == (50.0, 50.0)
    assert bbox_center((10, 20, 30, 40)) == (20.0, 30.0)


def test_bbox_iou_full_overlap():
    iou = bbox_iou((0, 0, 100, 100), (0, 0, 100, 100))
    assert iou == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    iou = bbox_iou((0, 0, 100, 100), (200, 200, 300, 300))
    assert iou == 0.0


def test_bbox_iou_partial():
    iou = bbox_iou((0, 0, 100, 100), (50, 50, 150, 150))
    # Intersection = 50x50 = 2500, union = 10000 + 10000 - 2500 = 17500
    assert iou == pytest.approx(2500 / 17500, rel=1e-3)


def test_bbox_union():
    u = bbox_union((0, 0, 100, 100), (50, 50, 200, 200))
    assert u == (0, 0, 200, 200)


def test_face_vector_basic():
    # Person facing right: nose to right of shoulders
    kps = [[0, 0, 0]] * 17
    kps[0] = [55, 50, 0.9]  # nose — to the right
    kps[5] = [40, 60, 0.9]  # left shoulder
    kps[6] = [60, 60, 0.9]  # right shoulder
    vec = face_vector(kps)
    assert vec is not None
    # Should be pointing right
    assert vec[0] > 0


def test_face_to_face_dot_opposite():
    # Two people facing each other -> dot product < 0
    kps_a = [[0, 0, 0]] * 17
    kps_a[0] = [55, 50, 0.9]
    kps_a[5] = [40, 60, 0.9]
    kps_a[6] = [60, 60, 0.9]
    kps_b = [[0, 0, 0]] * 17
    kps_b[0] = [45, 50, 0.9]  # facing left
    kps_b[5] = [60, 60, 0.9]
    kps_b[6] = [40, 60, 0.9]
    dot = face_to_face_dot(kps_a, kps_b)
    assert dot is not None
    assert dot < 0


def test_pair_analyzer_skips_distant_pairs():
    analyzer = PairAnalyzer(camera_id="cam_001")
    # Two far-apart tracklets
    t1 = Tracklet(track_id=1, camera_id="cam_001",
                  bbox=(100, 100, 200, 300), confidence=0.9,
                  keypoints=[[150, 200, 0.9]] * 17,
                  keypoint_scores=[0.9] * 17)
    t2 = Tracklet(track_id=2, camera_id="cam_001",
                  bbox=(800, 100, 900, 300), confidence=0.9,
                  keypoints=[[850, 200, 0.9]] * 17,
                  keypoint_scores=[0.9] * 17)
    pairs = analyzer.update([t1, t2], frame_id=1, timestamp=0.0)
    assert pairs == []


def test_pair_analyzer_finds_close_pair():
    """Use the close_pair_detections fixture."""
    # We need tracklets with proper keypoints for face-to-face
    # Reuse fixture pattern inline
    def make_kps(cx, facing_right=True):
        nose_offset = 5 if facing_right else -5
        return [
            [cx + nose_offset, 110.0, 0.95],
            [cx, 105.0, 0.9], [cx, 105.0, 0.9],
            [cx, 110.0, 0.8], [cx, 110.0, 0.8],
            [cx - 30, 150.0, 0.9], [cx + 30, 150.0, 0.9],
            [cx, 200.0, 0.85], [cx, 200.0, 0.85],
            [cx, 250.0, 0.8], [cx, 250.0, 0.8],
            [cx - 20, 230.0, 0.85], [cx + 20, 230.0, 0.85],
            [cx, 270.0, 0.8], [cx, 270.0, 0.8],
            [cx, 300.0, 0.75], [cx, 300.0, 0.75],
        ]

    t1 = Tracklet(track_id=1, camera_id="cam_001",
                  bbox=(100.0, 100.0, 200.0, 300.0), confidence=0.92,
                  keypoints=make_kps(150, facing_right=True),
                  keypoint_scores=[0.9] * 17)
    t2 = Tracklet(track_id=2, camera_id="cam_001",
                  bbox=(180.0, 100.0, 280.0, 300.0), confidence=0.88,
                  keypoints=make_kps(230, facing_right=False),
                  keypoint_scores=[0.85] * 17)

    analyzer = PairAnalyzer(
        camera_id="cam_001",
        sustained_proximity_frames=1,  # Trigger after just 1 frame
        iou_threshold=0.0,  # Make it easy
    )
    pairs = analyzer.update([t1, t2], frame_id=1, timestamp=0.0)
    assert len(pairs) >= 1
    p = pairs[0]
    assert p.track_id_a == 1
    assert p.track_id_b == 2
    assert p.face_to_face_dot < -0.3


def test_pair_analyzer_sustained_proximity():
    """Pairs only emitted after sustained_proximity_frames."""
    t1 = Tracklet(track_id=1, camera_id="cam_001",
                  bbox=(100.0, 100.0, 200.0, 300.0), confidence=0.92,
                  keypoints=[[150, 200, 0.9]] * 17,
                  keypoint_scores=[0.9] * 17)
    t2 = Tracklet(track_id=2, camera_id="cam_001",
                  bbox=(150.0, 100.0, 250.0, 300.0), confidence=0.88,
                  keypoints=[[200, 200, 0.9]] * 17,
                  keypoint_scores=[0.85] * 17)
    # face-to-face: need proper keypoints
    def make_kps(cx, facing_right=True):
        nose_offset = 5 if facing_right else -5
        return [
            [cx + nose_offset, 110.0, 0.95],
            [cx, 105.0, 0.9], [cx, 105.0, 0.9],
            [cx, 110.0, 0.8], [cx, 110.0, 0.8],
            [cx - 30, 150.0, 0.9], [cx + 30, 150.0, 0.9],
            [cx, 200.0, 0.85], [cx, 200.0, 0.85],
            [cx, 250.0, 0.8], [cx, 250.0, 0.8],
            [cx - 20, 230.0, 0.85], [cx + 20, 230.0, 0.85],
            [cx, 270.0, 0.8], [cx, 270.0, 0.8],
            [cx, 300.0, 0.75], [cx, 300.0, 0.75],
        ]
    t1.keypoints = make_kps(150, True)
    t2.keypoints = make_kps(200, False)

    analyzer = PairAnalyzer(
        camera_id="cam_001",
        sustained_proximity_frames=3,
        iou_threshold=0.0,
    )
    # Frame 1: should not yet emit (sustained=1)
    p1 = analyzer.update([t1, t2], frame_id=1, timestamp=0.0)
    assert p1 == []
    # Frame 2: should not yet emit (sustained=2)
    p2 = analyzer.update([t1, t2], frame_id=2, timestamp=0.1)
    assert p2 == []
    # Frame 3: should emit (sustained=3)
    p3 = analyzer.update([t1, t2], frame_id=3, timestamp=0.2)
    assert len(p3) >= 1


def test_router_splits_pairs_and_singles():
    from src.common.schemas import PersonPair
    tracklets = [
        Tracklet(track_id=1, camera_id="c", bbox=(0, 0, 10, 10), confidence=0.9),
        Tracklet(track_id=2, camera_id="c", bbox=(0, 0, 10, 10), confidence=0.9),
        Tracklet(track_id=3, camera_id="c", bbox=(0, 0, 10, 10), confidence=0.9),
    ]
    pairs = [
        PersonPair(camera_id="c", frame_id=1, timestamp=0.0,
                   track_id_a=1, track_id_b=2,
                   bbox_a=(0, 0, 10, 10), bbox_b=(0, 0, 10, 10),
                   distance=5, iou=0.5, face_to_face_dot=-0.5,
                   sustained_frames=20, union_bbox=(0, 0, 10, 10)),
    ]
    router = Router()
    pair_out, singles = router.route(tracklets, pairs)
    assert len(pair_out) == 1
    assert len(singles) == 1
    assert singles[0].track_id == 3
