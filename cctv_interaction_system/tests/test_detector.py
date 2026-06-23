"""Tests for Layer 1 detector (mock)."""

from __future__ import annotations

import numpy as np

from src.layer1_detection.detector import MockDetector
from src.layer1_detection.postprocess import parse_yolov8_pose_output


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


def test_parse_yolov8_pose_letterbox_mapping():
    """Boxes/keypoints must be mapped out of the centre-padded letterbox.

    On a non-square (1280x720) frame the short side is padded, so the inverse
    transform must subtract the pad offset before rescaling — not just multiply
    by max(H, W) / input_size.
    """
    input_size = 640
    orig_shape = (720, 1280)  # (H, W)
    ratio = input_size / max(orig_shape)  # 0.5
    pad_x = (input_size - orig_shape[1] * ratio) / 2.0  # 0
    pad_y = (input_size - orig_shape[0] * ratio) / 2.0  # 140

    # Person centred at original (640, 360), size 100x300.
    out = np.zeros((56, 8400), dtype=np.float32)
    out[0, 0] = 640 * ratio + pad_x      # cx in letterbox space
    out[1, 0] = 360 * ratio + pad_y      # cy
    out[2, 0] = 100 * ratio              # w
    out[3, 0] = 300 * ratio              # h
    out[4, 0] = 0.9                      # confidence
    # Nose keypoint at original (640, 220).
    out[5, 0] = 640 * ratio + pad_x
    out[6, 0] = 220 * ratio + pad_y
    out[7, 0] = 0.8

    res = parse_yolov8_pose_output(
        out, conf_threshold=0.5, orig_shape=orig_shape, input_size=input_size,
    )
    assert len(res) == 1
    dets = res[0]
    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0]["bbox"]
    assert abs(x1 - 590) < 1e-3
    assert abs(x2 - 690) < 1e-3
    assert abs(y1 - 210) < 1e-3
    assert abs(y2 - 510) < 1e-3
    nose = dets[0]["keypoints"][0]
    assert abs(nose[0] - 640) < 1e-3
    assert abs(nose[1] - 220) < 1e-3
