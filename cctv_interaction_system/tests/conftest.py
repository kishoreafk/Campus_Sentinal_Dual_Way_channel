"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `config` / `src` import cleanly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force mock mode for all tests (no GPU required)
os.environ.setdefault("CCTV_MOCK_MODE", "true")
os.environ.setdefault("CCTV_LOG_LEVEL", "WARNING")

import pytest  # noqa: E402


@pytest.fixture
def sample_frame():
    """1280x720 BGR frame."""
    import numpy as np
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """A list of plausible Detection objects."""
    from src.common.schemas import Detection
    return [
        Detection(
            bbox=(100.0, 100.0, 200.0, 300.0),
            confidence=0.92,
            keypoints=[[150.0, 130.0, 0.9]] * 17,
            keypoint_scores=[0.9] * 17,
        ),
        Detection(
            bbox=(300.0, 100.0, 400.0, 300.0),
            confidence=0.88,
            keypoints=[[350.0, 130.0, 0.85]] * 17,
            keypoint_scores=[0.85] * 17,
        ),
    ]


@pytest.fixture
def close_pair_detections():
    """Two detections that satisfy all pair-analysis criteria."""
    from src.common.schemas import Detection

    # Two people facing each other, ~100px apart, IoU > 0.15
    # Each bbox: 100w x 200h, overlapping
    def make_kps(cx, facing_right=True):
        # Face vector: from shoulder midpoint to nose
        # If facing right: nose x > shoulder midpoint x
        nose_offset = 5 if facing_right else -5
        return [
            [cx + nose_offset, 110.0, 0.95],  # 0: nose
            [cx - 5 if facing_right else cx + 5, 105.0, 0.9],  # 1: left_eye
            [cx + 5 if facing_right else cx - 5, 105.0, 0.9],  # 2: right_eye
            [cx - 15, 110.0, 0.8],  # 3: left_ear
            [cx + 15, 110.0, 0.8],  # 4: right_ear
            [cx - 30, 150.0, 0.9],  # 5: left_shoulder
            [cx + 30, 150.0, 0.9],  # 6: right_shoulder
            [cx - 40, 200.0, 0.85],  # 7: left_elbow
            [cx + 40, 200.0, 0.85],  # 8: right_elbow
            [cx - 50, 250.0, 0.8],  # 9: left_wrist
            [cx + 50, 250.0, 0.8],  # 10: right_wrist
            [cx - 20, 230.0, 0.85],  # 11: left_hip
            [cx + 20, 230.0, 0.85],  # 12: right_hip
            [cx - 25, 270.0, 0.8],  # 13: left_knee
            [cx + 25, 270.0, 0.8],  # 14: right_knee
            [cx - 25, 300.0, 0.75],  # 15: left_ankle
            [cx + 25, 300.0, 0.75],  # 16: right_ankle
        ]

    return [
        Detection(
            bbox=(100.0, 100.0, 200.0, 300.0),
            confidence=0.92,
            keypoints=make_kps(150, facing_right=True),  # facing right (towards person 2)
            keypoint_scores=[0.9] * 17,
        ),
        Detection(
            bbox=(180.0, 100.0, 280.0, 300.0),  # overlaps with person 1
            confidence=0.88,
            keypoints=make_kps(230, facing_right=False),  # facing left (towards person 1)
            keypoint_scores=[0.85] * 17,
        ),
    ]
