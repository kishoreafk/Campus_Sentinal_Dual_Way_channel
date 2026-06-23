"""SlowFast RGB action recognition wrapper.

Real implementation uses a TensorRT-exported SlowFast model (e.g.
slowfast_r50_8xb8-8x8x256-256e_kinetics400). Mock returns deterministic
scores based on clip motion.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger

logger = get_logger()


class BaseSlowFast(abc.ABC):
    @abc.abstractmethod
    def infer_batch(self, clips: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            clips: (B, 3, T, H, W) float32 normalised [0, 1]

        Returns:
            (B, num_classes) softmax probs
        """


class MockSlowFast(BaseSlowFast):
    """Mock SlowFast — uses clip motion + colour statistics."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def infer_batch(self, clips: np.ndarray) -> np.ndarray:
        B = clips.shape[0]
        out = np.zeros((B, self.num_classes), dtype=np.float32)
        for b in range(B):
            out[b] = self._infer_one(clips[b])
        return out

    def _infer_one(self, clip: np.ndarray) -> np.ndarray:
        # clip shape (3, T, H, W)
        T = clip.shape[1]
        # Frame-to-frame pixel difference
        if T > 1:
            diffs = np.diff(clip, axis=1)
            motion = float(np.abs(diffs).mean())
        else:
            motion = 0.0
        # Mean colour saturation
        mean_rgb = clip.mean(axis=(1, 2))  # (3, T) -> mean over H,W
        saturation = float(mean_rgb.std())  # higher std = more colourful

        probs = np.zeros(self.num_classes, dtype=np.float32)
        # labels: [hug, kiss, fight, push, handshake, high-five, other, none]
        if motion > 0.15:
            probs[2] = 0.6  # fight
            probs[3] = 0.2  # push
        elif motion > 0.05:
            probs[5] = 0.5  # high-five
            probs[4] = 0.3  # handshake
        elif motion < 0.01:
            if saturation > 0.05:
                probs[0] = 0.5  # hug
                probs[1] = 0.3  # kiss
            else:
                probs[7] = 0.7  # none
        else:
            probs[6] = 0.5  # other
            probs[7] = 0.3  # none
        probs = probs + 0.02
        probs = probs / probs.sum()
        return probs


class TensorRTSlowFast(BaseSlowFast):
    """Real TensorRT SlowFast — stub."""

    def __init__(self, engine_path: str, num_classes: int):
        self.engine_path = engine_path
        self.num_classes = num_classes
        raise NotImplementedError(
            "TensorRTSlowFast is a stub. Use MockSlowFast for testing."
        )

    def infer_batch(self, clips: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def SlowFast(use_mock: Optional[bool] = None):
    if use_mock is None:
        use_mock = get_settings().mock_mode
    cfg = get_settings().layer4a
    if use_mock or cfg.use_mock:
        return MockSlowFast(len(cfg.interaction_labels))
    return TensorRTSlowFast(cfg.slowfast_engine_path, len(cfg.interaction_labels))
