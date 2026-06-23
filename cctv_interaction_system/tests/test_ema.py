"""Tests for Layer 5 EMA smoother."""

from __future__ import annotations

import numpy as np
import pytest

from src.layer5_postprocess.ema_smoother import EMASmoother


def test_ema_basic_update():
    ema = EMASmoother(alpha=0.7)
    s1 = ema.update("cam_1", "tid_1", "fight", 0.8)
    # First update: smoothed = 0.7*0 + 0.3*0.8 = 0.24
    assert s1 == pytest.approx(0.24, rel=1e-3)
    s2 = ema.update("cam_1", "tid_1", "fight", 0.8)
    # Second: 0.7*0.24 + 0.3*0.8 = 0.408
    assert s2 == pytest.approx(0.408, rel=1e-3)


def test_ema_different_tracks_independent():
    ema = EMASmoother(alpha=0.5)
    s1 = ema.update("cam_1", "tid_1", "fight", 1.0)
    s2 = ema.update("cam_1", "tid_2", "fight", 0.0)
    assert s1 == pytest.approx(0.5)
    assert s2 == pytest.approx(0.0)


def test_ema_probs_vector():
    ema = EMASmoother(alpha=0.5)
    labels = ["a", "b", "c"]
    probs1 = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    out1 = ema.update_probs("cam_1", "tid_1", labels, probs1)
    assert out1.shape == (3,)
    # First update: smoothed = alpha*0 + (1-alpha)*p
    np.testing.assert_allclose(out1, 0.5 * probs1, atol=1e-3)


def test_ema_reset():
    ema = EMASmoother(alpha=0.7)
    ema.update("cam_1", "tid_1", "fight", 0.9)
    ema.reset("cam_1", "tid_1")
    assert ema.get("cam_1", "tid_1", "fight") == 0.0


def test_ema_invalid_alpha():
    with pytest.raises(ValueError):
        EMASmoother(alpha=1.5)
    with pytest.raises(ValueError):
        EMASmoother(alpha=-0.1)
