"""Distance matrix and pair-scoring utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.common.schemas import Tracklet


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou(b1: Tuple[float, float, float, float],
             b2: Tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def bbox_union(b1: Tuple[float, float, float, float],
               b2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (min(b1[0], b2[0]), min(b1[1], b2[1]),
            max(b1[2], b2[2]), max(b1[3], b2[3]))


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def person_height(tracklet: Tracklet) -> float:
    """Estimate height in pixels from bbox."""
    return tracklet.height


def face_vector(keypoints: List[List[float]]) -> Optional[np.ndarray]:
    """Compute face orientation vector from keypoints.

    For upright people in CCTV, the shoulder line is roughly horizontal and
    the nose's horizontal offset from the shoulder midpoint indicates which
    way the person is facing (left / right in the image plane). We use only
    the x-component to avoid contamination from the nose being above the
    shoulders (which is always true anatomically).

    Returns:
        np.ndarray of shape (2,) — the (x, 0) direction unit vector, or
        None if the direction cannot be determined.
    """
    if len(keypoints) < 13:
        return None
    try:
        nose_x = float(keypoints[0][0])
        l_sh_x = float(keypoints[5][0])
        r_sh_x = float(keypoints[6][0])
        # Skip if any keypoint has low confidence
        if (keypoints[0][2] < 0.3 or keypoints[5][2] < 0.3 or keypoints[6][2] < 0.3):
            return None
        midpoint_x = (l_sh_x + r_sh_x) / 2.0
        dx = nose_x - midpoint_x
        # If the horizontal offset is too small, we can't determine facing direction
        if abs(dx) < 1.0:
            return None
        # Return unit vector along x-axis
        sign = 1.0 if dx > 0 else -1.0
        return np.array([sign, 0.0], dtype=np.float32)
    except (IndexError, TypeError):
        return None


def face_to_face_dot(kp_a: List[List[float]], kp_b: List[List[float]]) -> Optional[float]:
    """Dot product of two face orientation vectors.

    A value near -1 means the two people are facing each other; near +1 means
    facing the same direction. Returns None if either vector is undefined.
    """
    v_a = face_vector(kp_a)
    v_b = face_vector(kp_b)
    if v_a is None or v_b is None:
        return None
    return float(np.dot(v_a, v_b))
