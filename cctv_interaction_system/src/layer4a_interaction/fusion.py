"""Fusion module — re-exported from cascade_filter.

Kept as a separate file per the architecture diagram for clarity. The
actual fusion logic lives inside `CascadeFilter._fuse` because it's
tightly coupled with the cascade flow.
"""

from __future__ import annotations

import numpy as np


def weighted_fusion(
    pose_probs: np.ndarray,
    rgb_probs: np.ndarray,
    pose_weight: float = 0.6,
    rgb_weight: float = 0.4,
) -> np.ndarray:
    """Weighted average fusion of PoseConv3D + SlowFast probabilities.

    PoseConv3D is weighted higher for close interactions because skeletons
    are more reliable than RGB when bodies occlude each other.
    """
    if pose_probs.shape != rgb_probs.shape:
        raise ValueError(
            f"shape mismatch: pose {pose_probs.shape} vs rgb {rgb_probs.shape}"
        )
    return pose_weight * pose_probs + rgb_weight * rgb_probs
