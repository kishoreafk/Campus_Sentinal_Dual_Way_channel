"""Tests for Layer 2 Kalman pose interpolation."""

from __future__ import annotations

import numpy as np

from src.layer2_tracking.kalman_pose import KeypointKalman, PoseInterpolator


def test_keypoint_kalman_initialise():
    kf = KeypointKalman()
    kf.initialise(100.0, 200.0)
    x, y = kf.predict()
    assert abs(x - 100.0) < 5.0
    assert abs(y - 200.0) < 5.0


def test_keypoint_kalman_predict_advances():
    """After multiple updates with consistent motion, the Kalman filter
    should estimate velocity and advance the prediction."""
    kf = KeypointKalman(process_noise=1.0, measurement_noise=0.5)
    kf.initialise(100.0, 200.0)
    # Multiple updates showing consistent rightward motion
    for x in [105, 110, 115, 120]:
        kf.update(float(x), 200.0)
    # After 4 updates, velocity should be roughly estimated (15 px/frame)
    x_pred, y_pred = kf.predict()
    # The prediction should be at least past the last measurement (120)
    assert x_pred > 120.0, f"expected x_pred > 120 (last measurement), got {x_pred}"
    # And ideally advanced by some velocity
    x_pred2, _ = kf.predict()
    assert x_pred2 > x_pred, "second predict should advance further"


def test_pose_interpolator_update_predict():
    interp = PoseInterpolator(num_keypoints=17)
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:, 0] = np.arange(17) * 10.0
    kps[:, 1] = np.arange(17) * 5.0
    kps[:, 2] = 0.9
    interp.update(kps)
    pred = interp.predict()
    assert pred.shape == (17, 3)
    # Predicted positions should be near the last observed
    assert np.allclose(pred[:, :2], kps[:, :2], atol=20.0)


def test_pose_interpolator_low_confidence_ignored():
    interp = PoseInterpolator(num_keypoints=17)
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:, 2] = 0.1  # all low confidence
    interp.update(kps)
    # Should not crash, prediction should be zeros (or close)
    pred = interp.predict()
    assert pred.shape == (17, 3)
