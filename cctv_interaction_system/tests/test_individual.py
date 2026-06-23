"""Tests for Layer 4B individual recognition."""

from __future__ import annotations

import numpy as np

from src.common.schemas import Tracklet
from src.layer4b_individual.individual_recognizer import make_individual_recognizer
from src.layer4b_individual.skeleton_preprocess import preprocess_skeleton


def test_preprocess_skeleton_shape():
    skel = np.random.rand(20, 17, 3).astype(np.float32)
    out = preprocess_skeleton(skel)
    assert out.shape == (3, 48, 17, 1)


def test_preprocess_skeleton_full_length():
    skel = np.random.rand(48, 17, 3).astype(np.float32)
    out = preprocess_skeleton(skel)
    assert out.shape == (3, 48, 17, 1)


def test_preprocess_skeleton_overlong():
    skel = np.random.rand(60, 17, 3).astype(np.float32)
    out = preprocess_skeleton(skel)
    assert out.shape == (3, 48, 17, 1)


def test_preprocess_skeleton_centers_to_hip():
    # Build a skeleton with hip at known position
    skel = np.zeros((10, 17, 3), dtype=np.float32)
    skel[:, :, 2] = 0.9  # all confidence
    skel[:, 11, 0] = 100.0  # left_hip x
    skel[:, 11, 1] = 200.0  # left_hip y
    skel[:, 12, 0] = 110.0  # right_hip x
    skel[:, 12, 1] = 200.0  # right_hip y
    out = preprocess_skeleton(skel)
    # After centering, hip should be at origin (approx)
    hip_x = out[0, -1, 11, 0]
    hip_y = out[1, -1, 11, 0]
    assert abs(hip_x) < 0.1
    assert abs(hip_y) < 0.1


def test_individual_recognizer_returns_predictions():
    rec = make_individual_recognizer()
    skel = np.random.rand(48, 17, 3).astype(np.float32)
    skel[:, :, 2] = 0.9
    tracklets = [Tracklet(track_id=1, camera_id="cam_001",
                          bbox=(100, 100, 200, 300), confidence=0.9,
                          keypoints=[[150, 200, 0.9]] * 17,
                          keypoint_scores=[0.9] * 17)]
    skeleton_buffers = {1: skel}
    preds = rec.recognize(tracklets, skeleton_buffers,
                          frame_id=1, timestamp=0.0, camera_id="cam_001")
    assert len(preds) == 1
    p = preds[0]
    assert p.label in rec.labels
    assert 0.0 <= p.confidence <= 1.0
    assert p.track_id == 1


def test_individual_recognizer_empty_tracklets():
    rec = make_individual_recognizer()
    preds = rec.recognize([], {}, frame_id=1, timestamp=0.0, camera_id="c")
    assert preds == []


def test_individual_recognizer_missing_skeletons():
    rec = make_individual_recognizer()
    tracklets = [Tracklet(track_id=1, camera_id="c",
                          bbox=(100, 100, 200, 300), confidence=0.9)]
    preds = rec.recognize(tracklets, {}, frame_id=1, timestamp=0.0, camera_id="c")
    assert preds == []
