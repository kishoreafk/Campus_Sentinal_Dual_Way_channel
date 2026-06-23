"""Tests for Layer 1 detector (mock)."""

from __future__ import annotations

import numpy as np

from src.layer1_detection.detector import MockDetector


def test_mock_detector_returns_detections():
    det = MockDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    results = det.detect_batch(
        [frame],
        [{"camera_id": "cam_001", "frame_id": 1, "timestamp": 0.0}],
    )
    assert len(results) == 1
    r = results[0]
    assert r.camera_id == "cam_001"
    assert r.frame_id == 1
    assert len(r.detections) >= 1
    for d in r.detections:
        assert len(d.bbox) == 4
        assert d.bbox[0] < d.bbox[2]
        assert d.bbox[1] < d.bbox[3]
        assert len(d.keypoints) == 17
        assert len(d.keypoint_scores) == 17


def test_mock_detector_batch():
    det = MockDetector()
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    metas = [{"camera_id": "cam_001", "frame_id": i, "timestamp": float(i)}
             for i in range(5)]
    results = det.detect_batch(frames, metas)
    assert len(results) == 5
    for i, r in enumerate(results):
        assert r.frame_id == i


def test_mock_detector_deterministic():
    det = MockDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    meta = {"camera_id": "cam_001", "frame_id": 42, "timestamp": 0.0}
    r1 = det.detect_batch([frame], [meta])[0]
    r2 = det.detect_batch([frame], [meta])[0]
    assert len(r1.detections) == len(r2.detections)
    for d1, d2 in zip(r1.detections, r2.detections):
        assert d1.bbox == d2.bbox
        assert d1.confidence == d2.confidence
