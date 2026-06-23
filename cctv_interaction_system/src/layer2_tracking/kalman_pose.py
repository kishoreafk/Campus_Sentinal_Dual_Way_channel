"""Kalman filter for pose keypoint interpolation.

When detection runs at 6 FPS but tracking runs at 30 FPS, we need to fill
in the 4/5 frames between detections. A simple constant-velocity Kalman
filter on each keypoint (x, y, vx, vy) suffices.

We use filterpy's KalmanFilter under the hood, but fall back to a simple
constant-velocity predictor if filterpy is unavailable.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    from filterpy.kalman import KalmanFilter
    _HAS_FILTERPY = True
except ImportError:
    _HAS_FILTERPY = False


class KeypointKalman:
    """Per-keypoint constant-velocity Kalman filter.

    State: [x, y, vx, vy]
    Observation: [x, y]
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.05):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        if _HAS_FILTERPY:
            self._kf = KalmanFilter(dim_x=4, dim_z=2)
            self._kf.F = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            self._kf.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ], dtype=np.float32)
            self._kf.Q *= process_noise
            self._kf.R *= measurement_noise
            # Initialise P with high velocity uncertainty and position-velocity
            # correlation so velocity can be inferred from a single position update.
            # Default P = I is too restrictive — it makes the filter unable to
            # estimate velocity until many frames are observed.
            self._kf.P = np.array([
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 10.0, 0.0],
                [0.0, 1.0, 0.0, 10.0],
            ], dtype=np.float32)
        else:
            # Manual fallback state
            self._state = np.zeros(4, dtype=np.float32)
            self._initialised = False
        self._initialised_filterpy = False

    def initialise(self, x: float, y: float) -> None:
        if _HAS_FILTERPY:
            self._kf.x = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self._initialised_filterpy = True
        else:
            self._state = np.array([x, y, 0, 0], dtype=np.float32)
            self._initialised = True

    def update(self, x: float, y: float) -> None:
        if _HAS_FILTERPY:
            if not self._initialised_filterpy:
                self.initialise(x, y)
                return
            self._kf.update(np.array([[x], [y]], dtype=np.float32))
        else:
            if not getattr(self, "_initialised", False):
                self.initialise(x, y)
                return
            # Naive update: blend observation with prediction
            pred_x, pred_y = self._state[0], self._state[1]
            alpha = 0.5
            self._state[0] = alpha * x + (1 - alpha) * pred_x
            self._state[1] = alpha * y + (1 - alpha) * pred_y
            # Update velocity
            self._state[2] = self._state[0] - pred_x
            self._state[3] = self._state[1] - pred_y

    def predict(self) -> tuple[float, float]:
        if _HAS_FILTERPY:
            self._kf.predict()
            # filterpy uses (dim_x, 1) shape — index with [i, 0] or .flat[i]
            return float(self._kf.x[0, 0]), float(self._kf.x[1, 0])
        else:
            # Constant velocity predict
            self._state[0] += self._state[2]
            self._state[1] += self._state[3]
            return float(self._state[0]), float(self._state[1])


class PoseInterpolator:
    """Maintains one Kalman filter per keypoint (17 total)."""

    def __init__(
        self,
        num_keypoints: int = 17,
        process_noise: float = 0.01,
        measurement_noise: float = 0.05,
    ):
        self.num_keypoints = num_keypoints
        self.filters: List[KeypointKalman] = [
            KeypointKalman(process_noise, measurement_noise)
            for _ in range(num_keypoints)
        ]
        self.last_observed: Optional[np.ndarray] = None  # (17, 3)

    def update(self, keypoints: np.ndarray) -> None:
        """Update with observed keypoints (17, 3) — [x, y, conf]."""
        if keypoints.shape != (self.num_keypoints, 3):
            raise ValueError(
                f"Expected ({self.num_keypoints}, 3), got {keypoints.shape}"
            )
        for i, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > 0.3:
                self.filters[i].update(float(x), float(y))
        self.last_observed = keypoints.copy()

    def predict(self) -> np.ndarray:
        """Predict keypoints for next (skipped) frame. Returns (17, 3) with conf=0.5."""
        out = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        for i, f in enumerate(self.filters):
            x, y = f.predict()
            out[i, 0] = x
            out[i, 1] = y
            out[i, 2] = 0.5  # Lower confidence for interpolated
        return out
