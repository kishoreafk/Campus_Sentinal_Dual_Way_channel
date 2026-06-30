"""Tests for Layer 4A cascade filter and PoseConv3D."""

from __future__ import annotations

import numpy as np

from src.common.schemas import PersonPair
from src.layer4a_interaction.cascade_filter import CascadeFilter, make_cascade_filter
from src.layer4a_interaction.poseconv3d import MockPoseConv3D, PoseConv3D
from src.layer4a_interaction.roi_extractor import (
    ar_pad,
    extract_paired_skeletons,
    extract_roi_clip,
)



def test_poseconv3d_mock_pair_returns_probs():
    model = MockPoseConv3D(num_classes=8, mode="pair")
    # (B=2, 3, T=48, V=17, M=2)
    skel = np.random.rand(2, 3, 48, 17, 2).astype(np.float32)
    probs = model.infer_batch(skel)
    assert probs.shape == (2, 8)
    # Each row should sum to ~1 (softmax)
    for row in probs:
        assert abs(row.sum() - 1.0) < 1e-3


def test_poseconv3d_mock_single_returns_probs():
    model = MockPoseConv3D(num_classes=7, mode="single")
    skel = np.random.rand(3, 3, 48, 17, 1).astype(np.float32)
    probs = model.infer_batch(skel)
    assert probs.shape == (3, 7)
    for row in probs:
        assert abs(row.sum() - 1.0) < 1e-3


def test_poseconv3d_factory_returns_mock():
    model = PoseConv3D(mode="pair")
    assert isinstance(model, MockPoseConv3D)


def test_ar_pad_widens():
    p = ar_pad((0, 0, 100, 200), target_ratio=1.0)
    # w=100, h=200, target=1:1 -> w=200, h=200
    x1, y1, x2, y2 = p
    assert (x2 - x1) == (y2 - y1)


def test_ar_pad_tallens():
    p = ar_pad((0, 0, 200, 100), target_ratio=1.0)
    x1, y1, x2, y2 = p
    assert (x2 - x1) == (y2 - y1)


def test_extract_paired_skeletons_shape():
    skel_a = np.random.rand(48, 17, 3).astype(np.float32)
    skel_b = np.random.rand(48, 17, 3).astype(np.float32)
    out = extract_paired_skeletons(skel_a, skel_b, clip_len=48)
    assert out.shape == (3, 48, 17, 2)


def test_extract_paired_skeletons_pads_short():
    skel_a = np.random.rand(10, 17, 3).astype(np.float32)
    skel_b = np.random.rand(10, 17, 3).astype(np.float32)
    out = extract_paired_skeletons(skel_a, skel_b, clip_len=48)
    assert out.shape == (3, 48, 17, 2)


def test_extract_roi_clip_shape():
    frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(32)]
    clip = extract_roi_clip(frames, (100, 100, 300, 400),
                            clip_len=32, img_size=224, margin_ratio=0.2)
    assert clip.shape == (32, 3, 224, 224)
    # Values should be in [0, 1]
    assert clip.min() >= 0.0
    assert clip.max() <= 1.0


def test_extract_roi_clip_empty_buffer():
    clip = extract_roi_clip([], (100, 100, 300, 400), clip_len=32, img_size=224)
    assert clip is None


def test_weighted_fusion():
    pose = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    rgb = np.array([0.3, 0.7, 0.0], dtype=np.float32)
    cf = make_cascade_filter()
    fused = cf._fuse(pose, rgb)
    expected = cf.fusion_pose_weight * pose + cf.fusion_rgb_weight * rgb
    np.testing.assert_allclose(fused, expected)


def test_cascade_filter_recognize_pairs():
    """End-to-end cascade filter on synthetic pairs."""
    cf = make_cascade_filter()
    # Build two fake skeleton buffers and a pair
    skel_a = np.random.rand(48, 17, 3).astype(np.float32)
    skel_b = np.random.rand(48, 17, 3).astype(np.float32)
    skeleton_buffers = {1: skel_a, 2: skel_b}
    pairs = [
        PersonPair(
            camera_id="cam_001", frame_id=1, timestamp=0.0,
            track_id_a=1, track_id_b=2,
            bbox_a=(100, 100, 200, 300), bbox_b=(150, 100, 250, 300),
            distance=50, iou=0.3, face_to_face_dot=-0.5,
            sustained_frames=20, union_bbox=(100, 100, 250, 300),
        )
    ]
    frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(32)]
    results = cf.recognize(pairs, skeleton_buffers, frame_buffer=frames)
    assert len(results) == 1
    r = results[0]
    assert r.label in cf.labels
    assert 0.0 <= r.confidence <= 1.0


def test_cascade_filter_empty_pairs():
    cf = make_cascade_filter()
    results = cf.recognize([], {}, None)
    assert results == []


def test_cascade_filter_missing_skeletons():
    """If a track's skeleton is missing, return 'none' prediction."""
    cf = make_cascade_filter()
    pairs = [
        PersonPair(
            camera_id="cam_001", frame_id=1, timestamp=0.0,
            track_id_a=1, track_id_b=2,
            bbox_a=(100, 100, 200, 300), bbox_b=(150, 100, 250, 300),
            distance=50, iou=0.3, face_to_face_dot=-0.5,
            sustained_frames=20, union_bbox=(100, 100, 250, 300),
        )
    ]
    results = cf.recognize(pairs, {}, frame_buffer=[])
    assert len(results) == 1
    assert results[0].label == "none"
