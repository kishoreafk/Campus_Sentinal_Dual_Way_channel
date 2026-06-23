"""Skeleton preprocessing for individual action recognition.

Operations:
  1. Center to hip joint (mean of left_hip and right_hip)
  2. Scale by body height (max distance across keypoints)
  3. Interpolate missing joints (confidence < 0.3) using temporal neighbours
  4. Pad / trim to 48 frames
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def preprocess_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Preprocess a single-person skeleton clip.

    Args:
        skeleton: (T, 17, 3) float32 — [x, y, conf] per keypoint per frame

    Returns:
        (3, T, 17, 1) float32 — [x, y, conf], padded to 48 frames
    """
    T, V, _ = skeleton.shape
    skel = skeleton.copy().astype(np.float32)

    # 1. Interpolate missing joints (conf < 0.3) using temporal neighbours
    conf = skel[:, :, 2]
    mask = conf < 0.3
    if mask.any():
        for v in range(V):
            for t in range(T):
                if mask[t, v]:
                    # Find nearest valid frame
                    left = right = None
                    for k in range(1, T):
                        if t - k >= 0 and not mask[t - k, v]:
                            left = t - k
                            break
                    for k in range(1, T):
                        if t + k < T and not mask[t + k, v]:
                            right = t + k
                            break
                    if left is not None and right is not None:
                        # Linear interp
                        alpha = (t - left) / (right - left)
                        skel[t, v, 0] = (1 - alpha) * skel[left, v, 0] + alpha * skel[right, v, 0]
                        skel[t, v, 1] = (1 - alpha) * skel[left, v, 1] + alpha * skel[right, v, 1]
                        skel[t, v, 2] = 0.5  # interpolated confidence
                    elif left is not None:
                        skel[t, v, 0] = skel[left, v, 0]
                        skel[t, v, 1] = skel[left, v, 1]
                        skel[t, v, 2] = 0.4
                    elif right is not None:
                        skel[t, v, 0] = skel[right, v, 0]
                        skel[t, v, 1] = skel[right, v, 1]
                        skel[t, v, 2] = 0.4

    # 2. Center to hip joint
    left_hip = skel[:, 11, :2]
    right_hip = skel[:, 12, :2]
    hip_center = (left_hip + right_hip) / 2.0  # (T, 2)
    skel[:, :, :2] = skel[:, :, :2] - hip_center[:, None, :]

    # 3. Scale by body height
    # Body height = max y - min y across shoulders/hips/knees/ankles
    kps_indices = [5, 6, 11, 12, 13, 14, 15, 16]
    y_vals = skel[:, kps_indices, 1]
    body_h = float(y_vals.max() - y_vals.min())
    if body_h > 1e-3:
        skel[:, :, :2] = skel[:, :, :2] / body_h

    # 4. Pad to target T
    target_T = 48
    if T < target_T:
        pad = np.zeros((target_T - T, V, 3), dtype=np.float32)
        skel = np.concatenate([pad, skel], axis=0)
    elif T > target_T:
        skel = skel[-target_T:]

    # Reshape to (3, T, V, M=1)
    out = skel.transpose(2, 0, 1)  # (3, T, V)
    out = out[:, :, :, None]  # (3, T, V, 1)
    return out
